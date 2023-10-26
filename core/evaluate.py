import json
import time
import os
import re
import glob
import socket
import pprint
import logging
import shutil
from os.path import join, exists, basename

from core import utils
from core.models.model import NormalizedModel, LipschitzNetwork
from core.data.readers import readers_config

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class Evaluator:
  """Evaluate a Pytorch Model."""

  def __init__(self, config):
    self.config = config

  def load_ckpt(self, ckpt_path=None):
    if ckpt_path is None:
      checkpoints = glob.glob(join(self.config.train_dir, "checkpoints", "model.ckpt-*.pth"))
      get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
      ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
      ckpt_path = join(self.config.train_dir, "checkpoints", ckpt_name)
    checkpoint = torch.load(ckpt_path)
    new_checkpoint = {}
    for k, v in checkpoint['model_state_dict'].items():
      if 'alpha' not in k:
        new_checkpoint[k] = v
    self.model.load_state_dict(new_checkpoint)
    epoch = checkpoint['epoch']
    return epoch

  def __call__(self):
    """Run evaluation of model or eval under attack"""

    cudnn.benchmark = True

    # create a mesage builder for logging
    self.message = utils.MessageBuilder()
    # Setup logging & log the version.
    utils.setup_logging(self.config, 0)

    ngpus = torch.cuda.device_count()
    if ngpus:
      self.batch_size = self.config.batch_size * ngpus
    else:
      self.batch_size = self.config.batch_size

    # load reader
    Reader = readers_config[self.config.dataset]
    self.reader = Reader(self.config, self.batch_size, False, is_training=False)
    self.config.means = self.reader.means

    # load model
    self.model = LipschitzNetwork(self.config, self.reader.n_classes)
    self.model = NormalizedModel(self.model, self.reader.means, self.reader.stds)
    self.model = torch.nn.DataParallel(self.model)
    self.model = self.model.cuda()

    self.load_ckpt()
    if self.config.mode == "certified":
      for eps in [36, 72, 108, 255]:
        self.eval_certified(eps)
    elif self.config.mode == "attack":
      self.eval_attack()

    logging.info("Done with batched inference.")

  @torch.no_grad()
  def eval_certified(self, eps):
    eps_float = eps / 255
    self.model.eval()
    running_accuracy = 0
    running_certified = 0
    running_inputs = 0
    lip_cst = 1.
    data_loader, _ = self.reader.load_dataset()
    last_weight = self.model.module.model.model[-1].weight
    normalized_weight = F.normalize(last_weight, p=2, dim=1)
    for batch_n, data in enumerate(data_loader):
      inputs, labels = data
      inputs, labels = inputs.cuda(), labels.cuda()
      outputs = self.model(inputs)
      predicted = outputs.argmax(axis=1)
      correct = outputs.max(1)[1] == labels
      margins, indices = torch.sort(outputs, 1)
      margins = margins[:, -1][:, None] - margins[: , 0:-1]
      for idx in range(margins.shape[0]):
        margins[idx] /= torch.norm(
          normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]], dim=1, p=2)
      margins, _ = torch.sort(margins, 1)
      certified = margins[:, 0] > eps_float * lip_cst
      running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
      running_certified += torch.sum(correct & certified).item()
      running_inputs += inputs.size(0)
    accuracy = running_accuracy / running_inputs
    certified = running_certified / running_inputs
    self.message.add('eps', [eps, 255], format='.0f')
    self.message.add('eps', eps_float, format='.5f')
    self.message.add('accuracy', accuracy, format='.5f')
    self.message.add('certified accuracy', certified, format='.5f')
    logging.info(self.message.get_message())
    return accuracy, certified

  def eval_attack(self):
    """Run evaluation under attack."""

    attack = utils.get_attack_eval(
                    self.model,
                    self.config.attack,
                    self.config.eps/255,
                    self.batch_size)

    running_accuracy = 0
    running_accuracy_adv = 0
    running_inputs = 0
    data_loader, _ = self.reader.load_dataset()
    for batch_n, data in enumerate(data_loader):

      inputs, labels = data
      inputs, labels = inputs.cuda(), labels.cuda()
      inputs_adv = attack.perturb(inputs, labels)

      outputs = self.model(inputs)
      outputs_adv = self.model(inputs_adv)
      _, predicted = torch.max(outputs.data, 1)
      _, predicted_adv = torch.max(outputs_adv.data, 1)

      running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
      running_accuracy_adv += predicted_adv.eq(labels.data).cpu().sum().numpy()
      running_inputs += inputs.size(0)

    accuracy = running_accuracy / running_inputs
    accuracy_adv = running_accuracy_adv / running_inputs
    self.message.add(f'attack: {self.config.attack} - eps', self.config.eps, format='.0f')
    self.message.add('Accuracy', accuracy, format='.5f')
    self.message.add('Accuracy attack', accuracy_adv, format='.5f')
    logging.info(self.message.get_message())
