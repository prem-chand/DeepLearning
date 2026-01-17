"""
Autograd-based Multi-Layer Perceptron Module
=============================================

This module reimplements the MNIST MLP using the autograd engine instead of
manual backpropagation. It demonstrates how automatic differentiation simplifies
neural network implementation by eliminating the need to manually derive and
implement backward passes.

Architecture
------------
The network architecture mirrors the manual implementation:

    Input (784) -> Linear(784, 128) -> ReLU -> Linear(128, 10) -> Softmax

Key Differences from Manual Implementation
------------------------------------------
1. **No manual backward pass**: Gradients are computed automatically
2. **Simpler layer code**: Layers only need forward computation
3. **Flexible composition**: Easy to add new operations without deriving gradients

Classes
-------
- Layer: Base class for all layers
- LinearAutograd: Fully connected layer using Tensor autograd
- ReLUAutograd: ReLU activation layer
- SoftmaxAutograd: Softmax activation layer
- CrossEntropyLossAutograd: Cross-entropy loss function
- TwoLayerMLPAutograd: Complete 2-layer MLP for MNIST

Example Usage
-------------
>>> from deeplearning.autograd_mlp import TwoLayerMLPAutograd
>>> import numpy as np
>>>
>>> model = TwoLayerMLPAutograd()
>>> x = np.random.randn(64, 784)
>>> y = np.random.randint(0, 10, 64)
>>> loss = model.train_step(x, y, lr=0.01)
>>> print(f"Loss: {loss:.4f}")

Author: Deep Learning Assignment - Autograd Extension
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from deeplearning.autograd import Tensor


class Layer:
    """
    Base class for neural network layers.

    All layers should inherit from this class and implement:
    - __call__: Forward pass computation
    - parameters: Return list of trainable Tensor parameters

    Attributes
    ----------
    training : bool
        Whether the layer is in training mode (affects dropout, batchnorm, etc.)
    """

    def __init__(self):
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        """Return list of trainable parameters."""
        return []

    def train(self):
        """Set layer to training mode."""
        self.training = True

    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False


class LinearAutograd(Layer):
    """
    Fully connected layer using Tensor autograd: y = x @ W + b

    This layer performs a linear transformation of the input. Unlike the manual
    implementation, gradients are computed automatically via the Tensor class.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features

    Attributes
    ----------
    W : Tensor
        Weight matrix of shape (in_features, out_features)
    b : Tensor
        Bias vector of shape (out_features,)

    Examples
    --------
    >>> layer = LinearAutograd(784, 128)
    >>> x = Tensor(np.random.randn(32, 784))
    >>> y = layer(x)
    >>> y.shape
    (32, 128)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Kaiming He initialization for ReLU
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = x @ W + b

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, in_features)

        Returns
        -------
        Tensor
            Output tensor of shape (N, out_features)
        """
        return x @ self.W + self.b

    def parameters(self) -> List[Tensor]:
        """Return weight and bias tensors."""
        return [self.W, self.b]


class ReLUAutograd(Layer):
    """
    ReLU activation layer using Tensor autograd: y = max(0, x)

    Examples
    --------
    >>> relu = ReLUAutograd()
    >>> x = Tensor([[-1, 0, 1], [2, -2, 3]])
    >>> y = relu(x)
    >>> print(y.data)
    [[0 0 1]
     [2 0 3]]
    """

    def __call__(self, x: Tensor) -> Tensor:
        """Apply ReLU activation."""
        return x.relu()


class SigmoidAutograd(Layer):
    """
    Sigmoid activation layer: y = 1 / (1 + exp(-x))
    """

    def __call__(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation."""
        return x.sigmoid()


class TanhAutograd(Layer):
    """
    Tanh activation layer: y = tanh(x)
    """

    def __call__(self, x: Tensor) -> Tensor:
        """Apply tanh activation."""
        return x.tanh()


class SoftmaxAutograd(Layer):
    """
    Softmax activation layer: y_i = exp(x_i) / sum(exp(x_j))

    Parameters
    ----------
    axis : int
        Axis along which to compute softmax (default: -1, last axis)
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        """Apply softmax activation."""
        return x.softmax(axis=self.axis)


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Compute cross-entropy loss from logits and targets.

    This function combines softmax and negative log-likelihood into a single
    numerically stable operation.

    Parameters
    ----------
    logits : Tensor
        Raw (unnormalized) scores of shape (N, C)
    targets : np.ndarray
        Class indices of shape (N,) with values in [0, C-1]

    Returns
    -------
    Tensor
        Scalar loss tensor

    Mathematical Formulation
    ------------------------
    L = -(1/N) * sum_n(log(softmax(logits_n)[targets_n]))

    Examples
    --------
    >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    >>> targets = np.array([0, 1])
    >>> loss = cross_entropy_loss(logits, targets)
    >>> loss.backward()
    """
    N = logits.shape[0]

    # Compute softmax probabilities
    probs = logits.softmax(axis=-1)

    # Get probabilities of correct classes
    # Create one-hot encoding for indexing
    log_probs = probs.log()

    # Select the log probabilities for the correct classes
    # This is equivalent to: -log(probs[arange(N), targets])
    one_hot = np.zeros_like(logits.data)
    one_hot[np.arange(N), targets] = 1.0
    one_hot_tensor = Tensor(one_hot, requires_grad=False)

    # Compute negative log likelihood
    # Sum over classes (only the correct class contributes due to one-hot)
    # Then average over batch
    nll = -(log_probs * one_hot_tensor).sum() / N

    return nll


class TwoLayerMLPAutograd:
    """
    2-layer MLP for MNIST using autograd.

    This class reimplements TwoLayerMLP using the Tensor autograd engine.
    The architecture and training procedure are identical, but gradients
    are computed automatically instead of manually.

    Architecture
    ------------
    Input (784) -> Linear(784, 128) -> ReLU -> Linear(128, 10) -> Output

    The output is raw logits; softmax is applied in the loss function.

    Attributes
    ----------
    layers : List[Layer]
        Ordered list of layers

    Examples
    --------
    >>> model = TwoLayerMLPAutograd()
    >>> x = np.random.randn(64, 784)
    >>> y = np.random.randint(0, 10, 64)
    >>> loss = model.train_step(x, y, lr=0.01)
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 128,
                 output_size: int = 10):
        """
        Initialize the 2-layer MLP.

        Parameters
        ----------
        input_size : int
            Input dimension (default: 784 for MNIST)
        hidden_size : int
            Hidden layer dimension (default: 128)
        output_size : int
            Output dimension (default: 10 for MNIST digits)
        """
        self.layers = [
            LinearAutograd(input_size, hidden_size),
            ReLUAutograd(),
            LinearAutograd(hidden_size, output_size)
        ]

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Forward pass through all layers.

        Parameters
        ----------
        x : Tensor or np.ndarray
            Input of shape (N, input_size)

        Returns
        -------
        Tensor
            Output logits of shape (N, output_size)
        """
        if isinstance(x, np.ndarray):
            x = Tensor(x, requires_grad=False)

        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero."""
        for param in self.parameters():
            param.zero_grad()

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray,
                   lr: float) -> float:
        """
        Perform a single training step.

        Parameters
        ----------
        x_batch : np.ndarray
            Input batch of shape (N, 784)
        y_batch : np.ndarray
            Target labels of shape (N,)
        lr : float
            Learning rate

        Returns
        -------
        float
            Loss value for this batch
        """
        # Zero gradients
        self.zero_grad()

        # Forward pass
        logits = self.forward(x_batch)

        # Compute loss
        loss = cross_entropy_loss(logits, y_batch)

        # Backward pass (automatic!)
        loss.backward()

        # SGD update
        for param in self.parameters():
            param.data -= lr * param.grad

        return loss.item()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Get class predictions.

        Parameters
        ----------
        x : np.ndarray
            Input of shape (N, 784)

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (N,)
        """
        logits = self.forward(x)
        return np.argmax(logits.data, axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        x : np.ndarray
            Input of shape (N, 784)
        y : np.ndarray
            True labels of shape (N,)

        Returns
        -------
        float
            Accuracy in [0, 1]
        """
        predictions = self.predict(x)
        return np.mean(predictions == y)


def compare_gradients_with_manual(seed: int = 42) -> dict:
    """
    Compare gradients from autograd vs manual implementation.

    This function verifies that the autograd implementation produces
    the same gradients as the manual backpropagation implementation.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with gradient comparison results
    """
    from deeplearning.mnist_mlp import TwoLayerMLP
    from deeplearning.neural_net import Linear

    np.random.seed(seed)

    # Create test data
    x = np.random.randn(32, 784)
    y = np.random.randint(0, 10, 32)

    # Initialize both models with same weights
    manual_model = TwoLayerMLP()
    autograd_model = TwoLayerMLPAutograd()

    # Copy weights from manual to autograd model
    # Manual: layers[0] is Linear(784, 128), layers[2] is Linear(128, 10)
    # Autograd: same structure

    # Layer 1 weights (note: manual uses W.T convention, autograd uses W)
    autograd_model.layers[0].W.data = manual_model.layers[0]._w.T.copy()
    autograd_model.layers[0].b.data = manual_model.layers[0]._b.copy()

    # Layer 2 weights
    autograd_model.layers[2].W.data = manual_model.layers[2]._w.T.copy()
    autograd_model.layers[2].b.data = manual_model.layers[2]._b.copy()

    # Run forward and backward on both models
    # Manual model
    manual_loss = manual_model.train_step(x, y, lr=0.0)  # lr=0 to not update

    # Re-copy weights (they weren't updated due to lr=0)
    # But we need to get gradients, so re-run without update
    manual_model.layers[0]._w = autograd_model.layers[0].W.data.T.copy()
    manual_model.layers[0]._b = autograd_model.layers[0].b.data.copy()
    manual_model.layers[2]._w = autograd_model.layers[2].W.data.T.copy()
    manual_model.layers[2]._b = autograd_model.layers[2].b.data.copy()

    # Forward pass
    y_pred_manual = manual_model.forward(x)
    loss_manual = manual_model.loss_fn.forward(y_pred_manual, y)
    dL_dy = manual_model.loss_fn.backward()
    manual_model.backward(dL_dy)

    # Autograd model
    autograd_model.zero_grad()
    logits = autograd_model.forward(x)
    loss_autograd = cross_entropy_loss(logits, y)
    loss_autograd.backward()

    # Compare gradients
    # Manual dL_dW is shape (out, in), autograd grad is shape (in, out)
    results = {
        'loss_manual': loss_manual,
        'loss_autograd': loss_autograd.item(),
        'loss_match': np.allclose(loss_manual, loss_autograd.item(), rtol=1e-5),

        # Layer 1 weight gradients
        'w1_grad_match': np.allclose(
            manual_model.layers[0].dL_dW.T,
            autograd_model.layers[0].W.grad,
            rtol=1e-5
        ),
        'w1_grad_max_diff': np.max(np.abs(
            manual_model.layers[0].dL_dW.T - autograd_model.layers[0].W.grad
        )),

        # Layer 1 bias gradients
        'b1_grad_match': np.allclose(
            manual_model.layers[0].dL_db,
            autograd_model.layers[0].b.grad,
            rtol=1e-5
        ),

        # Layer 2 weight gradients
        'w2_grad_match': np.allclose(
            manual_model.layers[2].dL_dW.T,
            autograd_model.layers[2].W.grad,
            rtol=1e-5
        ),
        'w2_grad_max_diff': np.max(np.abs(
            manual_model.layers[2].dL_dW.T - autograd_model.layers[2].W.grad
        )),

        # Layer 2 bias gradients
        'b2_grad_match': np.allclose(
            manual_model.layers[2].dL_db,
            autograd_model.layers[2].b.grad,
            rtol=1e-5
        ),
    }

    return results


if __name__ == "__main__":
    """Demonstrate the autograd MLP and verify gradients."""
    np.random.seed(42)

    print("=" * 60)
    print("Autograd MLP Demonstration")
    print("=" * 60)

    # Create model
    model = TwoLayerMLPAutograd()

    # Create dummy data
    x = np.random.randn(64, 784)
    y = np.random.randint(0, 10, 64)

    # Training step
    loss = model.train_step(x, y, lr=0.01)
    print(f"\nTraining loss: {loss:.4f}")

    # Predictions
    accuracy = model.evaluate(x, y)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    # Verify gradients match manual implementation
    print("\n" + "=" * 60)
    print("Gradient Verification: Autograd vs Manual")
    print("=" * 60)

    results = compare_gradients_with_manual()

    print(f"\nLoss comparison:")
    print(f"  Manual:   {results['loss_manual']:.6f}")
    print(f"  Autograd: {results['loss_autograd']:.6f}")
    print(f"  Match:    {results['loss_match']}")

    print(f"\nGradient comparisons:")
    print(f"  Layer 1 W grad match: {results['w1_grad_match']} "
          f"(max diff: {results['w1_grad_max_diff']:.2e})")
    print(f"  Layer 1 b grad match: {results['b1_grad_match']}")
    print(f"  Layer 2 W grad match: {results['w2_grad_match']} "
          f"(max diff: {results['w2_grad_max_diff']:.2e})")
    print(f"  Layer 2 b grad match: {results['b2_grad_match']}")

    all_match = (results['loss_match'] and results['w1_grad_match'] and
                 results['b1_grad_match'] and results['w2_grad_match'] and
                 results['b2_grad_match'])

    print(f"\nAll gradients match: {all_match}")
