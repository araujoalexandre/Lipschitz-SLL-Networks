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

    self.n_conv= config.n_conv
    self.n_dense = config.n_dense
    self.cin = config.w
    self.conv_inner_dim = config.conv_inner_dim
    self.dense_inner_dim = config.dense_inner_dim
    self.n_classes = n_classes

    if config.dataset == 'tiny-imagenet':
      imsize = 64
    else:
      imsize = 32

    self.model = []
    self.model.append(
      PaddingChannels(self.cin, 3, "zero")
    )

    for _ in range(self.n_conv):
      self.model.append(SDPBasedLipschitzConvLayer(self.cin, self.conv_inner_dim))
    
    self.model.append(nn.AvgPool2d(4, divisor_override=4))

    self.model.append(nn.Flatten())
    if config.dataset in ['cifar10', 'cifar100']:
      in_channels = self.cin * 8 * 8
    elif config.dataset == 'tiny-imagenet':
      in_channels = self.cin * 16 * 16

    for _ in range(self.n_dense):
      self.model.append(SDPBasedLipschitzLinearLayer(in_channels, self.dense_inner_dim))

    self.model.append(LinearNormalized(in_channels, self.n_classes))
    self.model = nn.Sequential(*self.model)

  def forward(self, x):
    return self.model(x)


