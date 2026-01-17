"""
Deep Learning from Scratch
==========================

A pedagogical implementation of neural networks built entirely from scratch
using NumPy. This package provides core components for understanding deep
learning fundamentals.

Modules
-------
neural_net : Core neural network layers (Linear, ReLU, Sigmoid, Tanh)
losses : Loss functions (MSELoss, CrossEntropyLoss)
mnist_mlp : Example 2-layer MLP for MNIST classification

Example
-------
>>> from deeplearning.neural_net import Linear, ReLU
>>> from deeplearning.losses import CrossEntropyLoss
>>> from deeplearning.mnist_mlp import TwoLayerMLP
>>>
>>> model = TwoLayerMLP()
>>> loss = model.train_step(x_batch, y_batch, lr=0.01)
"""

from deeplearning.neural_net import Linear, ReLU, Sigmoid, Tanh
from deeplearning.losses import MSELoss, CrossEntropyLoss
from deeplearning.mnist_mlp import TwoLayerMLP

__version__ = "0.1.0"
__author__ = "Prem Chand"

__all__ = [
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "MSELoss",
    "CrossEntropyLoss",
    "TwoLayerMLP",
]
