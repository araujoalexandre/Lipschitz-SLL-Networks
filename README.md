# Lipschitz SLL Networks - ICLR 2023


The official implementation of the paper "[A Unified Algebraic Perspective on Lipschitz Neural Networks](https://openreview.net/forum?id=k71IGLC8cfc)"

## Abstract
Important research efforts have focused on the design and training of neural networks with a controlled Lipschitz constant. The goal is to increase and sometimes guarantee the robustness against adversarial attacks. Recent promising techniques draw inspirations from different backgrounds to design 1-Lipschitz neural networks, just to name a few: convex potential layers derive from the discretization of continuous dynamical systems, Almost-Orthogonal-Layer proposes a tailored method for matrix rescaling.  However, it is today important to consider the recent and promising contributions in the field under a common theoretical lens to better design new and improved layers. This paper introduces a novel algebraic perspective unifying various types of 1-Lipschitz neural networks, including the ones previously mentioned, along with methods based on orthogonality and spectral methods. Interestingly, we show that many existing techniques can be derived and generalized via finding analytical solutions of a common semidefinite programming (SDP) condition.  We also prove that AOL biases the scaled weight to the ones which are close to the set of orthogonal matrices in a certain mathematical manner. Moreover, our algebraic condition, combined with the Gershgorin circle theorem, readily leads to new and diverse parameterizations for 1-Lipschitz network layers. Our approach, called SDP-based Lipschitz Layers (SLL), allows us to design non-trivial yet efficient generalization of convex potential layers.  Finally, the comprehensive set of experiments on image classification shows that SLLs outperforms previous approaches on natural and certified accuracy.


## Experiments

Experiments done with Pytorch 1.10

#### train & evaluation
```
# training
torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset cifar10 --model-name small --train_dir small_model 

# Evaluation
python3 main.py --mode certified --dataset cifar10 --model-name small --train_dir small_model
```

#### Reproducing CIFAR10/100 results from the paper
- Download the checkpoints [here](https://drive.google.com/file/d/1fUamevS89mn0gDlTIY40N8SfRtWAWlif/view?usp=share_link)

```
python3 reproduce_tables.py
```

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{araujo2023a,
  title={A Unified Algebraic Perspective on Lipschitz Neural Networks},
  author={Alexandre Araujo and Aaron J Havens and Blaise Delattre and Alexandre Allauzen and Bin Hu},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=k71IGLC8cfc}
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


