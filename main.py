
import os
import sys
import warnings
import argparse
import shutil
import submitit
from os.path import exists, realpath
from datetime import datetime
import numpy as np
from core.trainer import Trainer
from core.evaluate import Evaluator

warnings.filterwarnings("ignore")

def main(config):
  folder = config.train_dir.split('/')[-1]
  if config.mode == 'train':
    trainer = Trainer(config)
    trainer()
  elif config.mode in ['certified', 'attack']:
    evaluate = Evaluator(config)
    evaluate()


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Train or Evaluate Lipschitz Networks.')

  # parameters training or eval
  parser.add_argument("--mode", type=str, default="train", choices=['train', 'certified', 'attack'])
  parser.add_argument("--train_dir", type=str, help="Name of the training directory.")
  parser.add_argument("--data_dir", type=str, help="Name of the data directory.")
  parser.add_argument("--dataset", type=str,  default='cifar10', help="Dataset to use")

  parser.add_argument("--shift_data", type=bool, default=True, help="Shift dataset with mean.")

  parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
  parser.add_argument("--loss", type=str, default="xent", help="Define the loss to use for training.")
  parser.add_argument("--margin", type=float, default=0.7, help="Define margin")
  parser.add_argument("--offset", type=float, default=np.sqrt(2)*3/2)
  parser.add_argument("--temperature", type=float, default=0.25)
  parser.add_argument("--optimizer", type=str, default='adam')
  parser.add_argument("--scheduler", type=str, default="interp")
  parser.add_argument("--lr", type=float, default=0.01)
  parser.add_argument("--beta1", type=float, default=0.5)
  parser.add_argument("--beta2", type=float, default=0.9)
  parser.add_argument("--wd", type=float, default=0, help="Weight decay to use for training.")
  parser.add_argument("--nesterov", action='store_true', default=False)
  parser.add_argument("--warmup_scheduler", type=float, default=0., help="Percentage of training.")
  parser.add_argument("--decay", type=str, help="Milestones for MultiStepLR")
  parser.add_argument("--gamma", type=float, help="Gamma for MultiStepLR")
  parser.add_argument("--gradient_clip_by_norm", type=float, default=None)
  parser.add_argument("--gradient_clip_by_value", type=float, default=None)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--seed", type=int, help="Make the training deterministic.")
  parser.add_argument("--print_grad_norm", action='store_true', help="Print of the norm of the gradients")
  parser.add_argument("--frequency_log_steps", type=int, default=1000, help="Print log for every step.")
  parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")
  parser.add_argument("--save_checkpoint_epochs", type=int, default=5, help="Save checkpoint every epoch.")

  # specific parameters for eval
  parser.add_argument("--attack", type=str, choices=['pgd', 'autoattack'], help="Choose the attack.")
  parser.add_argument("--eps", type=float, default=36)

  # parameters of the architectures
  parser.add_argument("--model-name", type=str)
  parser.add_argument("--w", type=int, default=1)
  parser.add_argument("--conv_inner_dim", type=int, default=5)
  parser.add_argument("--dense_inner_dim", type=int, default=-1)
  parser.add_argument("--n_conv", type=int, default=30, help="The depth of the Stable Resnet")
  parser.add_argument("--n_dense", type=int, default=5, help="The number of linear layers.")

  # parse all arguments 
  config = parser.parse_args()
  config.cmd = f"python3 {' '.join(sys.argv)}"

  def override_args(config, n_conv, n_dense, w, dense_inner_dim):
    config.n_conv = n_conv
    config.n_dense = n_dense
    config.w = w
    config.dense_inner_dim = dense_inner_dim
    return config

  if config.model_name == 'small':
    config = override_args(config, 20, 7, 45, 2048)
  elif config.model_name == 'medium':
    config = override_args(config, 30, 10, 60, 2048)
  elif config.model_name == 'large':
    config = override_args(config, 50, 10, 90, 2048)
  elif config.model_name == 'xlarge':
    config = override_args(config, 70, 15, 120, 2048)
  elif config.model_name is None and \
      not all([config.n_conv, config.n_dense, config.w, config.dense_inner_dim]):
    ValueError("Choose --model-name 'small' 'medium' 'large' 'xlarge'")

  # process argments
  eval_mode = ['certified', 'attack']
  if config.data_dir is None:
    config.data_dir = os.environ.get('DATADIR', None)
  if config.data_dir is None:
    ValueError("the following arguments are required: --data_dir")
  os.makedirs('./trained_models', exist_ok=True)
  path = realpath('./trained_models')
  if config.train_dir is None:
    ValueError("--train_dir must be defined.")
  elif config.mode == 'train' and config.train_dir is not None:
    config.train_dir = f'{path}/{config.train_dir}'
    os.makedirs(config.train_dir, exist_ok=True)
    os.makedirs(f'{config.train_dir}/checkpoints', exist_ok=True)
  elif config.mode in eval_mode and config.train_dir is not None:
    config.train_dir = f'{path}/{config.train_dir}'
  elif config.mode in eval_mode and config.train_dir is None:
    ValueError("--train_dir must be defined.")


  if config.mode == 'attack' and config.attack is None:
    ValueError('With mode=attack, the following arguments are required: --attack')

  main(config)


