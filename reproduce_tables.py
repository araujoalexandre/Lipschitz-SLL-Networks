import glob
import natsort
from os.path import exists, realpath

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from core import utils
from core.models.model import NormalizedModel, LipschitzNetwork
from core.data.readers import readers_config

class Config:

  def __init__(self, dataset, size):
    self.mode = 'certified'
    self.dataset = dataset
    self.data_dir = '/gpfsscratch/rech/yxj/uuc79vj/data:/gpfswork/rech/yxj/uuc79vj/data:/gpfswork/rech/yxj/uuc79vj/data/vision_datasets:/gpfsdswork/dataset'
    self.ngpus = 4
    self.batch_size = 500
    self.shift_data = True
    self.first_layer = 'padding_channels'
    if dataset in ['cifar100', 'tiny-imagenet']:
      self.last_layer = 'lln' 
    else:
      self.last_layer = 'pooling_linear'

    if size == 'small':
      config = self.set_archi(20, 45, 7, 2048)
    elif size == 'medium':
      config = self.set_archi(30, 60, 10, 2048)
    elif size == 'large':
      config = self.set_archi(50, 90, 10, 2048)
    elif size == 'xlarge':
      config = self.set_archi(70, 120, 15, 2048)

  def set_archi(self, depth, num_channels, depth_linear, n_features):
    self.depth = depth
    self.num_channels = num_channels
    self.depth_linear = depth_linear
    self.n_features = n_features
    self.conv_size = 5


def load_ckpt(model, ckpt_path):
  checkpoint = torch.load(ckpt_path)['model_state_dict']
  model.load_state_dict(checkpoint)


@torch.no_grad()
def eval_certified(model, reader):
  model.eval()
  running_accuracy = 0
  running_certified = torch.zeros(4)
  running_inputs = 0
  lip_cst = 1.
  data_loader, _ = reader.load_dataset()
  for batch_n, data in enumerate(data_loader):
    inputs, labels = data
    inputs, labels = inputs.cuda(), labels.cuda()
    outputs = model(inputs)
    predicted = outputs.argmax(axis=1)
    correct = outputs.max(1)[1] == labels
    margins = torch.sort(outputs, 1)[0]
    running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
    for i, eps in enumerate([36, 72, 108, 255]):
      eps_float = eps / 255
      certified = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * lip_cst * eps_float
      running_certified[i] += torch.sum(correct & certified).item()
    running_inputs += inputs.size(0)
  accuracy = running_accuracy / running_inputs
  certified = running_certified / running_inputs
  return accuracy, certified

@torch.no_grad()
def eval_certified_lln(model, reader):
  model.eval()
  running_accuracy = 0
  running_certified = torch.zeros(4)
  running_inputs = 0
  lip_cst = 1.
  data_loader, _ = reader.load_dataset()
  last_weight = model.module.model.last_last.weight
  normalized_weight = F.normalize(last_weight, p=2, dim=1)
  for batch_n, data in enumerate(data_loader):
    inputs, labels = data
    inputs, labels = inputs.cuda(), labels.cuda()
    outputs = model(inputs)
    predicted = outputs.argmax(axis=1)
    correct = outputs.max(1)[1] == labels
    margins, indices = torch.sort(outputs, 1)
    margins = margins[:, -1][:, None] - margins[: , 0:-1]
    for idx in range(margins.shape[0]):
      margins[idx] /= torch.norm(
        normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]], dim=1, p=2)
    margins, _ = torch.sort(margins, 1)
    running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
    for i, eps in enumerate([36, 72, 108, 255]):
      eps_float = eps / 255
      certified = margins[:, 0] > eps_float * lip_cst
      running_certified[i] += torch.sum(correct & certified).item()
    running_inputs += inputs.size(0)
  accuracy = running_accuracy / running_inputs
  certified = running_certified / running_inputs
  return accuracy, certified




def main():
  
  cudnn.benchmark = True

  datasets = ['cifar10', 'cifar100']
  model_sizes = ['small', 'medium', 'large', 'xlarge']

  # create a mesage builder for logging
  message = utils.MessageBuilder()

  for dataset in datasets:
    print(f'-- Evaluation on {dataset} --')
    print('+-----------------------------------------------+')
    print('| {:>5}  |  {:>5}  |  {:>5}  |  {:>5}  |  {:>5} |'.format(
      *['Nat', 36, 72, 108, 255]))
    print('+-----------------------------------------------+')
    for size in model_sizes:
      paths = natsort.natsorted(glob.glob(f'./ckpts/{dataset}/{size}/model**'))

      config = Config(dataset, size)
      batch_size = config.batch_size * config.ngpus

      # load reader
      Reader = readers_config[dataset]
      reader = Reader(config, batch_size, False, is_training=False)
      config.means = reader.means

      # load model
      model = LipschitzNetwork(config, reader.n_classes)
      model = NormalizedModel(model, reader.means, reader.stds)
      model = torch.nn.DataParallel(model)
      model = model.cuda()

      avg_accuracy = 0.
      avg_certified = torch.zeros(4)
      for ckpt_path in paths:
        load_ckpt(model, ckpt_path)
        if not config.last_layer == 'lln':
          accuracy, certified = eval_certified(model, reader)
        else:
          accuracy, certified = eval_certified_lln(model, reader)
        avg_accuracy += accuracy
        avg_certified += certified

      avg_accuracy = avg_accuracy / len(paths)
      avg_certified = avg_certified / len(paths)

      print('| {:>5.1f}  |  {:>5.1f}  |  {:>5.1f}  |  {:>5.1f}  |  {:>5.1f} |'.format(
        avg_accuracy*100, *(avg_certified*100)))
    print('+-----------------------------------------------+\n')




if __name__ == '__main__':
  main()





