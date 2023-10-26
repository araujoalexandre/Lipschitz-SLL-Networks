import os, sys
import re
import shutil
import json
import logging
import glob
import copy
import subprocess
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import lr_scheduler
from advertorch import attacks
# from autoattack import AutoAttack
import pytorch_warmup as warmup


def get_epochs_from_ckpt(filename):
  regex = "(?<=ckpt-)[0-9]+"
  return int(re.findall(regex, filename)[-1])

def get_list_checkpoints(train_dir):
  files = glob.glob(join(train_dir, "checkpoints", "model.ckpt-*.pth"))
  files = sorted(files, key=get_epochs_from_ckpt)
  return [filename for filename in files]


class MessageBuilder:

  def __init__(self):
    self.msg = []

  def add(self, name, values, align=">", width=0, format=None):
    if name:
      metric_str = "{}: ".format(name)
    else:
      metric_str = ""
    values_str = []
    if type(values) != list:
      values = [values]
    for value in values:
      if format:
        values_str.append("{value:{align}{width}{format}}".format(
          value=value, align=align, width=width, format=format))
      else:
        values_str.append("{value:{align}{width}}".format(
          value=value, align=align, width=width))
    metric_str += '/'.join(values_str)
    self.msg.append(metric_str)

  def get_message(self):
    message = " | ".join(self.msg)
    self.clear()
    return message

  def clear(self):
    self.msg = []


def setup_logging(config, rank):
  level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
    'INFO': 20, 'WARN': 30
  }[config.logging_verbosity]
  format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
  filename = '{}/log_{}_{}.logs'.format(config.train_dir, config.mode, rank)
  f = open(filename, "a")
  logging.basicConfig(filename=filename, level=level, format=format_, datefmt='%H:%M:%S')


def setup_distributed_training(world_size, rank):
  """ find a common host name on all nodes and setup distributed training """
  # make sure http proxy are unset, in order for the nodes to communicate
  for var in ['http_proxy', 'https_proxy']:
    if var in os.environ:
      del os.environ[var]
    if var.upper() in os.environ:
      del os.environ[var.upper()]
  # get distributed url 
  cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
  stdout = subprocess.check_output(cmd.split())
  host_name = stdout.decode().splitlines()[0]
  dist_url = f'tcp://{host_name}:9000'
  # setup dist.init_process_group
  dist.init_process_group(backend='nccl', init_method=dist_url,
    world_size=world_size, rank=rank)


class LossMargin(nn.Module):

  def __init__(self, config):
    super(LossMargin, self).__init__()
    self.criterion = nn.MultiMarginLoss(margin=config.margin)

  def __call__(self, outputs, labels):
    loss = self.criterion(outputs.cpu(), labels.cpu()).cuda()
    return loss


class LossXent(nn.Module):

  def __init__(self, config):
    super(LossXent, self).__init__()
    self.criterion = nn.CrossEntropyLoss()
    self.n_classes = {
      'cifar10': 10,
      'cifar100': 100,
      'tiny-imagenet': 200
    }[config.dataset]
    self.offset = config.offset
    self.temperature = config.temperature

  def __call__(self, outputs, labels):
    one_hot_labels = F.one_hot(labels, num_classes=self.n_classes)
    offset_outputs = outputs - self.offset * one_hot_labels
    offset_outputs /= self.temperature
    loss = self.criterion(offset_outputs, labels) * self.temperature
    return loss


class LossXentMargin(nn.Module):

  def __init__(self, config):
    super(LossXentMargin, self).__init__()
    self.config = config
    self.criterion1 = LossMargin(config)
    self.criterion2 = LossXent(config)

  def __call__(self, outputs, labels):
    loss1 = self.criterion1(outputs, labels)
    loss2 = self.criterion2(outputs, labels)
    return loss1 + loss2


class MarginBoosting(nn.Module):

  def __init__(self, config):
    super(MarginBoosting, self).__init__()
    self.config = config
    self.criterion = nn.CrossEntropyLoss()
    self.n_classes = 10 if config.dataset == 'cifar10' else 100

  def __call__(self, outputs, labels):
    loss1 = self.criterion(outputs, labels)
    loss2 = - F.log_softmax(-outputs, dim=1)
    loss2 *= (1 - F.one_hot(labels, self.n_classes))
    loss2 = loss2.mean() / (self.n_classes - 1)
    loss = loss1 + loss2
    return loss


def get_loss(config):
  if config.loss == 'xent':
    criterion = LossXent(config)
  elif config.loss == 'margin':
    criterion = LossMargin(config)
  elif config.loss == 'xent+margin':
    criterion = LossXentMargin(config)
  elif config.loss == 'margin_boosting':
    criterion = MarginBoosting(config)
  return criterion

def get_scheduler(optimizer, config, num_steps):
  """Return a learning rate scheduler schedulers."""
  if config.scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=num_steps)
  elif config.scheduler == 'interp':
    scheduler = TriangularLRScheduler(
      optimizer, num_steps, config.lr)
  elif config.scheduler == 'multi_step_lr':
    if config.decay is not None:
      steps_by_epochs = num_steps / config.epochs
      milestones = np.array(list(map(int, config.decay.split('-'))))
      milestones = list(np.int32(milestones * steps_by_epochs))
    else:
      milestones = list(map(int, [1/10 * num_steps, 5/10 * num_steps, 8.5/10 * num_steps]))
    scheduler = lr_scheduler.MultiStepLR(
      optimizer, milestones=milestones, gamma=config.gamma)
  else:
    ValueError("Scheduler not reconized")
  warmup_scheduler = None
  if config.warmup_scheduler > 0:
    warmup_period = int(num_steps * config.warmup_scheduler)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
  return scheduler, warmup_scheduler


def get_optimizer(config, params):
  """Returns the optimizer that should be used based on params."""
  lr, wd = config.lr, config.wd
  betas = (config.beta1, config.beta2)
  if config.optimizer == 'sgd':
    opt = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=config.nesterov)
  elif config.optimizer == 'adam':
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
  elif config.optimizer == 'adamw':
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
  else:
    raise ValueError("Optimizer was not recognized")
  return opt

 
def get_attack_adv_training(model, config):
  eps_float = config.adv_training / 255
  norm = config.adv_training_norm
  attack_params = {
    'eps': eps_float,
    'nb_iter': 4,
    'eps_iter': 2.*eps_float/10 
  }
  if norm == 'l2':
    attack = attacks.L2PGDAttack(model, **attack_params)
  elif norm == 'linf':
    attack = attacks.LinfPGDAttack(model, **attack_params)
  else:
    raise ValueError("Norm not recognized for PGD attack.")
  return attack


def get_attack_eval(model, attack_name, eps, batch_size):
  if attack_name == 'autoattack':
    attack = AutoAttack(model, norm='L2', eps=eps, version='standard')
    attack.perturb = lambda x, y: attack.run_standard_evaluation(x, y, bs=batch_size)
  elif attack_name == 'pgd':
    nb_iter = 100
    attack = attacks.L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                 eps=eps, nb_iter=nb_iter, eps_iter=2.*eps/nb_iter, 
                                 rand_init=True, clip_min=0, clip_max=1, targeted=False)
  else:
    raise ValueError("Attack name not recognized for adv training.")
  return attack



class TriangularLRScheduler:

  def __init__(self, optimizer, num_steps, lr):
    self.optimizer = optimizer
    self.num_steps = num_steps
    self.lr = lr

  def step(self, t):
    lr = np.interp([t],
      [0, self.num_steps * 2 // 5, self.num_steps * 4 // 5, self.num_steps],
      [0, self.lr, self.lr / 20.0, 0])[0]
    self.optimizer.param_groups[0].update(lr=lr)


