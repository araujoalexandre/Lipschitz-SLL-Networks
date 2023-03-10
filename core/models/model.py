import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from core.models.layers import LinearNormalized, PoolingLinear, PaddingChannels
from core.models.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer


class NormalizedModel(nn.Module):

  def __init__(self, model, mean, std):
    super(NormalizedModel, self).__init__()
    self.model = model
    self.normalize = Normalize(mean, std)

  def forward(self, x):
    return self.model(self.normalize(x))


class LipschitzNetwork(nn.Module):

  def __init__(self, config, n_classes):
    super(LipschitzNetwork, self).__init__()

    self.depth = config.depth
    self.num_channels = config.num_channels
    self.depth_linear = config.depth_linear
    self.n_features = config.n_features
    self.conv_size = config.conv_size
    self.n_classes = n_classes

    if config.dataset == 'tiny-imagenet':
      imsize = 64
    else:
      imsize = 32

    self.conv1 = PaddingChannels(self.num_channels, 3, "zero")

    layers = []
    block_conv = SDPBasedLipschitzConvLayer
    block_lin = SDPBasedLipschitzLinearLayer

    for _ in range(self.depth):
      layers.append(block_conv(config, (1, self.num_channels, imsize, imsize), self.num_channels, self.conv_size))
    

    layers.append(nn.AvgPool2d(4, divisor_override=4))
    self.stable_block = nn.Sequential(*layers)

    layers_linear = [nn.Flatten()]
    if config.dataset in ['cifar10', 'cifar100']:
      in_channels = self.num_channels * 8 * 8
    elif config.dataset == 'tiny-imagenet':
      in_channels = self.num_channels * 16 * 16

    for _ in range(self.depth_linear):
      layers_linear.append(block_lin(config, in_channels, self.n_features))

    if config.last_layer == 'pooling_linear':
      self.last_last = PoolingLinear(in_channels, self.n_classes, agg="trunc")
    elif config.last_layer == 'lln':
      self.last_last = LinearNormalized(in_channels, self.n_classes)
    else:
      raise ValueError("Last layer not recognized")


    self.layers_linear = nn.Sequential(*layers_linear)
    self.base = nn.Sequential(*[self.conv1, self.stable_block, self.layers_linear])

  def forward(self, x):
    return self.last_last(self.base(x))


