import glob
import natsort
from os.path import exists, realpath

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from core import utils
from torch import autograd
from torch.autograd import Variable, Function

from core.models.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer


def _norm_gradient_sq(linear_fun, v):
    v = Variable(v, requires_grad=True)
    loss = torch.norm(linear_fun(v))**2
    loss.backward()
    return v.grad.data


def generic_power_method(affine_fun, input_size, eps=1e-8, max_iter=500, use_cuda=False):
    """ Return the highest singular value of the linear part of
    `affine_fun` and it's associated left / right singular vectors.
    INPUT:
        * `affine_fun`: an affine function
        * `input_size`: size of the input
        * `eps`: stop condition for power iteration
        * `max_iter`: maximum number of iterations
        * `use_cuda`: set to True if CUDA is present
    OUTPUT:
        * `eigenvalue`: maximum singular value of `affine_fun`
        * `v`: the associated left singular vector
        * `u`: the associated right singular vector
    NOTE:
        This algorithm is not deterministic, depending of the random
        initialisation, the returned eigenvectors are defined up to the sign.
        If affine_fun is a PyTorch model, beware of setting to `False` all
        parameters.requires_grad.
    TEST::
        >>> conv = nn.Conv2d(3, 8, 5)
        >>> for p in conv.parameters(): p.requires_grad = False
        >>> s, u, v = generic_power_method(conv, [1, 3, 28, 28])
        >>> bias = conv(torch.zeros([1, 3, 28, 28]))
        >>> linear_fun = lambda x: conv(x) - bias
        >>> torch.norm(linear_fun(v) - s * u) # should be very small
    """
    zeros = torch.zeros(input_size)
    if use_cuda:
        zeros = zeros.cuda()
    bias = affine_fun(Variable(zeros))
    linear_fun = lambda x: affine_fun(x) - bias

    def norm(x, p=2):
      """ Norm for each batch
      """
      norms = Variable(torch.zeros(x.shape[0]))
      if use_cuda:
          norms = norms.cuda()
      for i in range(x.shape[0]):
          norms[i] = x[i].norm(p=p)
      return norms

    # Initialise with random values
    v = torch.randn(input_size)
    v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
    if use_cuda:
        v = v.cuda()

    stop_criterion = False
    it = 0
    while not stop_criterion:
        previous = v
        v = _norm_gradient_sq(linear_fun, v)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    # Compute Rayleigh product to get eivenvalue
    u = linear_fun(Variable(v))  # unormalized left singular vector
    eigenvalue = norm(u)
    u = u.div(eigenvalue)
    return eigenvalue.item()




def main():
  
  cudnn.benchmark = True

  datasets = ['cifar10', 'cifar100']
  model_sizes = ['small', 'medium', 'large', 'xlarge']

  # create a mesage builder for logging
  message = utils.MessageBuilder()

  for dataset in datasets:
    print(dataset)
    for size in model_sizes:
      paths = natsort.natsorted(glob.glob(f'./ckpts/{dataset}/{size}/model**'))

      for ckpt_path in paths:

        data = torch.load(ckpt_path)['model_state_dict']

        n_conv_layers = len(list(filter(
          lambda x: 'module.model.model' in x and 'kernel' in x, data.keys())))
        n_dense_layers = len(list(filter(
          lambda x: 'module.model.model' in x and 'weight' in x, data.keys())))

        for i in range(1, n_conv_layers+1):
          layer_ckpt = {
            'kernel': data[f'module.model.model.{i}.kernel'],
            'bias': data[f'module.model.model.{i}.bias'],
            'q': data[f'module.model.model.{i}.q']
          }
          cout, cin = data[f'module.model.model.{i}.kernel'].shape[:2]
          layer = SDPBasedLipschitzConvLayer(cin, cout)
          layer.load_state_dict(layer_ckpt)
          layer = layer.cuda()
          layer.kernel.requires_grad = False
          layer.bias.requires_grad = False
          layer.q.requires_grad = False

          input_size = (1, cin, 32, 32)
          sv = generic_power_method(layer, input_size, eps=1e-8, max_iter=500, use_cuda=True)
          print(f'SLL Conv {i}: {sv}')

        for i in range(n_conv_layers+3, n_dense_layers+n_conv_layers+2):
          layer_ckpt = {
            'weight': data[f'module.model.model.{i}.weight'],
            'bias': data[f'module.model.model.{i}.bias'],
            'q': data[f'module.model.model.{i}.q'],
          }
          cout, cin = data[f'module.model.model.{i}.weight'].shape
          layer = SDPBasedLipschitzLinearLayer(cin, cout)
          layer.load_state_dict(layer_ckpt)
          layer = layer.cuda()
          layer.weight.requires_grad = False
          layer.bias.requires_grad = False
          layer.q.requires_grad = False

          input_size = (1, cout, cin)
          sv = generic_power_method(layer, input_size, eps=1e-8, max_iter=500, use_cuda=True)
          print(f'SLL Dense {i}: {sv}')



if __name__ == '__main__':
  main()





