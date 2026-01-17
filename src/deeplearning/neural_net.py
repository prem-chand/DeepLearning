"""
Neural Network Core Layers Module
=================================

This module implements the fundamental building blocks for neural networks from scratch
using only NumPy. It provides fully connected (Linear) layers and common activation
functions with both forward and backward passes for manual backpropagation.

The implementation follows the standard deep learning convention where each layer:
1. Computes output during forward pass and caches necessary values
2. Computes gradients during backward pass using cached values
3. Stores parameter gradients for optimizer updates

Mathematical Background
-----------------------
Each layer implements the chain rule for backpropagation:
dL/dx = dL/dy * dy/dx
where dL/dy is the upstream gradient and dy/dx is the local gradient.

Example Usage
-------------
.. code-block:: python

    import numpy as np
    from deeplearning.neural_net import Linear, ReLU

    # Create a simple two-layer network
    linear1 = Linear(784, 128)
    relu = ReLU()
    linear2 = Linear(128, 10)

    # Forward pass
    x = np.random.randn(32, 784)  # Batch of 32 samples
    h1 = linear1.forward(x)
    h1_activated = relu.forward(h1)
    output = linear2.forward(h1_activated)

    # Backward pass (assuming dL_doutput is the gradient from loss)
    # dL_doutput needs to be defined elsewhere in a full example
    # dL_dh1_activated = linear2.backward(dL_doutput)
    # dL_dh1 = relu.backward(dL_dh1_activated)
    # dL_dx = linear1.backward(dL_dh1)

    # Update parameters using stored gradients
    # learning_rate needs to be defined
    # linear1._w -= learning_rate * linear1.dL_dW
    # linear1._b -= learning_rate * linear1.dL_db

References
----------
* He et al., "Delving Deep into Rectifiers", https://arxiv.org/pdf/1502.01852
* CS231n: Convolutional Neural Networks for Visual Recognition

Author: Prem Chand
"""
import numpy as np
from typing import Tuple, Optional


class Linear:
    """
    Fully connected (dense) layer implementing y = Wx + b.

    This layer performs a linear transformation of the input by multiplying it
    with a weight matrix and adding a bias vector. It is the fundamental building
    block for multi-layer perceptrons (MLPs) and many other neural network architectures.

    Mathematical Formulation
    ------------------------
    Forward pass:
        y = x @ W^T + b
    where:
        - x: input tensor of shape (N, in_features)
        - W: weight matrix of shape (out_features, in_features)
        - b: bias vector of shape (out_features,)
        - y: output tensor of shape (N, out_features)

    Backward pass (given upstream gradient dL/dy):
        dL/dx = dL/dy @ W           (for propagating gradients backward)
        dL/dW = dL/dy^T @ x         (for updating weights)
        dL/db = sum(dL/dy, axis=0)  (for updating biases)

    Weight Initialization
    ---------------------
    Uses Kaiming He initialization: W ~ N(0, sqrt(2/in_features))
    This initialization is specifically designed for layers followed by ReLU
    activations, helping to maintain gradient magnitude through deep networks.

    Attributes
    ----------
    _w : np.ndarray
        Weight matrix of shape (out_features, in_features).
    _b : np.ndarray
        Bias vector of shape (out_features,).
    _x : np.ndarray or None
        Cached input from forward pass, used in backward pass.
    dL_dW : np.ndarray or None
        Gradient of loss w.r.t. weights, computed during backward pass.
    dL_db : np.ndarray or None
        Gradient of loss w.r.t. biases, computed during backward pass.

    Examples
    --------
    >>> layer = Linear(in_features=784, out_features=128)
    >>> x = np.random.randn(32, 784)  # Batch of 32 samples
    >>> y = layer.forward(x)
    >>> y.shape
    (32, 128)
    >>> dL_dy = np.random.randn(32, 128)  # Upstream gradient
    >>> dL_dx = layer.backward(dL_dy)
    >>> dL_dx.shape
    (32, 784)
    >>> layer.dL_dW.shape
    (128, 784)
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize the linear layer with He initialization.

        Parameters
        ----------
        in_features : int
            Number of input features (dimension of input vectors).
        out_features : int
            Number of output features (dimension of output vectors).

        Notes
        -----
        He initialization sets weights to random values drawn from:
            W ~ N(0, sqrt(2/in_features))
        This helps prevent vanishing/exploding gradients when using ReLU.
        Reference: https://arxiv.org/pdf/1502.01852
        """
        self._w = np.random.randn(
            out_features, in_features) * np.sqrt(2. / in_features)
        self._b = np.zeros((out_features,))
        self._x: Optional[np.ndarray] = None
        self.dL_dW: Optional[np.ndarray] = None
        self.dL_db: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass: y = x @ W^T + b.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (N, in_features), where N is the batch size.

        Returns
        -------
        np.ndarray
            Output tensor of shape (N, out_features).

        Notes
        -----
        The input is cached for use in the backward pass. This is necessary
        because the gradient dL/dW depends on the input values.
        """
        self._x = x
        return x @ self._w.T + self._b

    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass using the chain rule.

        Parameters
        ----------
        dL_dy : np.ndarray
            Upstream gradient (gradient of loss w.r.t. layer output),
            shape (N, out_features).

        Returns
        -------
        np.ndarray
            Downstream gradient (gradient of loss w.r.t. layer input),
            shape (N, in_features).

        Side Effects
        ------------
        Stores the following gradients as instance attributes:
        - dL_dW : gradient w.r.t. weights, shape (out_features, in_features)
        - dL_db : gradient w.r.t. biases, shape (out_features,)

        Mathematical Derivation
        -----------------------
        Given y = x @ W^T + b and upstream gradient dL/dy:

        1. dL/dx = dL/dy @ W
           (Chain rule: gradient flows backward through W^T)

        2. dL/dW = dL/dy^T @ x
           (Each element W[i,j] affects y[:, i] through x[:, j])

        3. dL/db = sum(dL/dy, axis=0)
           (Bias affects all samples equally, so gradients are summed)
        """
        dL_dx = dL_dy @ self._w
        self.dL_dW = dL_dy.T @ self._x  # Shape: (out_features, in_features) to match _w
        self.dL_db = np.sum(dL_dy, axis=0)

        return dL_dx


class ReLU:
    """
    Rectified Linear Unit activation function: y = max(0, x).

    ReLU is the most commonly used activation function in modern deep learning.
    It introduces non-linearity while being computationally efficient and helping
    to mitigate the vanishing gradient problem.

    Mathematical Formulation
    ------------------------
    Forward pass:
        y = max(0, x)

    Backward pass:
        dL/dx = dL/dy * (x > 0)

    The gradient is 1 where x > 0 and 0 where x <= 0.

    Attributes
    ----------
    _x : np.ndarray
        Cached input from forward pass, used to compute gradient mask.

    Notes
    -----
    - ReLU has a "dying ReLU" problem: neurons can become inactive if they
      consistently receive negative inputs, as their gradients become zero.
    - At x = 0, the gradient is technically undefined, but we use 0 by convention.
    - Variants like Leaky ReLU and ELU address the dying ReLU problem.

    Examples
    --------
    >>> relu = ReLU()
    >>> x = np.array([[-1, 0, 1], [2, -2, 3]])
    >>> y = relu.forward(x)
    >>> y
    array([[0, 0, 1],
           [2, 0, 3]])
    >>> dL_dy = np.ones_like(y)
    >>> dL_dx = relu.backward(dL_dy)
    >>> dL_dx
    array([[0, 0, 1],
           [1, 0, 1]])
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass: y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Input tensor of any shape.

        Returns
        -------
        np.ndarray
            Output tensor of same shape as input, with negative values zeroed.
        """
        self._x = x
        return np.maximum(0, x)

    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        """
        Compute backward pass: gradient is 1 where x > 0, else 0.

        Parameters
        ----------
        dL_dy : np.ndarray
            Upstream gradient of same shape as forward pass output.

        Returns
        -------
        np.ndarray
            Downstream gradient of same shape as forward pass input.

        Notes
        -----
        The gradient at x = 0 is set to 0 (using <= comparison).
        This is a common convention, though some implementations use < 0.
        """
        dL_dx = dL_dy.copy()
        dL_dx[self._x <= 0] = 0
        return dL_dx


class Sigmoid:
    """
    Sigmoid activation function: y = 1 / (1 + exp(-x)).

    The sigmoid function squashes inputs to the range (0, 1), making it useful
    for binary classification outputs and gates in architectures like LSTMs.

    Mathematical Formulation
    ------------------------
    Forward pass:
        y = sigma(x) = 1 / (1 + exp(-x))

    Backward pass:
        dL/dx = dL/dy * y * (1 - y)

    The derivative has the elegant form: sigma'(x) = sigma(x) * (1 - sigma(x))

    Attributes
    ----------
    _x : np.ndarray
        Cached input from forward pass.
    _y : np.ndarray
        Cached output from forward pass, used to compute gradient efficiently.

    Notes
    -----
    - Sigmoid suffers from vanishing gradients: when ``|x|`` is large, the gradient
      approaches zero, slowing down learning.
    - The output is always positive and sums to values > 0, which can cause
      issues with zero-centered gradients.
    - For hidden layers, ReLU is generally preferred; sigmoid is mainly used
      for output layers in binary classification.

    Examples
    --------
    >>> sigmoid = Sigmoid()
    >>> x = np.array([[0, 1, -1], [2, -2, 0]])
    >>> y = sigmoid.forward(x)
    >>> y  # Values between 0 and 1
    array([[0.5       , 0.73105858, 0.26894142],
           [0.88079708, 0.11920292, 0.5       ]])
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass: y = 1 / (1 + exp(-x)).

        Parameters
        ----------
        x : np.ndarray
            Input tensor of any shape.

        Returns
        -------
        np.ndarray
            Output tensor of same shape, with values in (0, 1).

        Notes
        -----
        For very large negative x, exp(-x) can overflow. A numerically stable
        implementation would handle this, but for typical neural network
        inputs with proper initialization, this is usually not an issue.
        """
        self._x = x
        self._y = 1 / (1 + np.exp(-x))
        return self._y

    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        """
        Compute backward pass: dL/dx = dL/dy * y * (1 - y).

        Parameters
        ----------
        dL_dy : np.ndarray
            Upstream gradient of same shape as forward pass output.

        Returns
        -------
        np.ndarray
            Downstream gradient of same shape as forward pass input.

        Notes
        -----
        The gradient y * (1 - y) has maximum value 0.25 (at y = 0.5),
        which contributes to the vanishing gradient problem in deep networks.
        """
        dL_dx = dL_dy * self._y * (1 - self._y)
        return dL_dx


class Tanh:
    """
    Hyperbolic tangent activation function: y = tanh(x).

    Tanh squashes inputs to the range (-1, 1). It is zero-centered, which can
    help with gradient flow compared to sigmoid, but still suffers from
    vanishing gradients for large input magnitudes.

    Mathematical Formulation
    ------------------------
    Forward pass:
        y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Backward pass:
        dL/dx = dL/dy * (1 - y^2)

    The derivative is: tanh'(x) = 1 - tanh(x)^2 = sech^2(x)

    Attributes
    ----------
    _x : np.ndarray
        Cached input from forward pass.
    _y : np.ndarray
        Cached output from forward pass, used to compute gradient efficiently.

    Notes
    -----
    - Tanh is a scaled and shifted version of sigmoid: tanh(x) = 2*sigmoid(2x) - 1
    - Zero-centered output helps with gradient flow in some architectures.
    - Still suffers from vanishing gradients, but less severe than sigmoid.
    - Commonly used in RNNs and LSTMs for hidden state computations.

    Examples
    --------
    >>> tanh = Tanh()
    >>> x = np.array([[0, 1, -1], [2, -2, 0]])
    >>> y = tanh.forward(x)
    >>> y  # Values between -1 and 1
    array([[ 0.        ,  0.76159416, -0.76159416],
           [ 0.96402758, -0.96402758,  0.        ]])
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass: y = tanh(x).

        Parameters
        ----------
        x : np.ndarray
            Input tensor of any shape.

        Returns
        -------
        np.ndarray
            Output tensor of same shape, with values in (-1, 1).
        """
        self._x = x
        self._y = np.tanh(x)
        return self._y

    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        """
        Compute backward pass: dL/dx = dL/dy * (1 - y^2).

        Parameters
        ----------
        dL_dy : np.ndarray
            Upstream gradient of same shape as forward pass output.

        Returns
        -------
        np.ndarray
            Downstream gradient of same shape as forward pass input.

        Notes
        -----
        The gradient (1 - y^2) ranges from 0 (when y = +/-1) to 1 (when y = 0).
        This is better than sigmoid's max gradient of 0.25, but still
        contributes to vanishing gradients in deep networks.
        """
        dL_dx = dL_dy * (1 - self._y ** 2)
        return dL_dx
