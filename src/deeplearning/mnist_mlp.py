"""
MNIST Multi-Layer Perceptron Training Module
============================================

This module implements a complete 2-layer Multi-Layer Perceptron (MLP) for
classifying MNIST handwritten digits. It demonstrates how to combine the
neural network components from this project into a working training pipeline.

Architecture
------------
The network architecture follows the classic LeNet-inspired design:

    Input (784) → Linear(784, 128) → ReLU → Linear(128, 10) → Softmax

- **Input Layer**: 784 features (28x28 flattened images)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (one per digit class 0-9)
- **Loss Function**: Cross-Entropy with Softmax

Training Pipeline
-----------------
1. **Forward Pass**: Input flows through all layers sequentially
2. **Loss Computation**: Cross-entropy measures prediction error
3. **Backward Pass**: Gradients flow backward through all layers
4. **Parameter Update**: Simple SGD updates weights and biases

This implements the core training loop that all modern deep learning frameworks
abstract away:

    for each batch:
        predictions = model.forward(batch)
        loss = loss_fn.forward(predictions, targets)
        gradients = loss_fn.backward()
        model.backward(gradients)
        model.update_parameters(learning_rate)

Example Usage
-------------
Training on MNIST (requires mnist data):

>>> from mnist_mlp import TwoLayerMLP
>>> import numpy as np
>>>
>>> model = TwoLayerMLP()
>>>
>>> # Training loop
>>> for epoch in range(5):
...     for x_batch, y_batch in data_loader:
...         loss = model.train_step(x_batch, y_batch, lr=0.01)
...     print(f"Epoch {epoch}: Loss = {loss:.4f}")

Quick test with random data:

>>> model = TwoLayerMLP()
>>> x = np.random.randn(64, 784)  # Batch of 64
>>> y = np.random.randint(0, 10, size=64)  # Random labels
>>> loss = model.train_step(x, y, lr=0.01)
>>> print(f"Loss: {loss:.4f}")

Performance Expectations
------------------------
With proper training on MNIST:
- Train accuracy: >98%
- Test accuracy: >95% (target from assignment)
- Training loss should decrease steadily

Key Hyperparameters
-------------------
- Learning rate: 0.01 (can be tuned)
- Batch size: 64 (used in example)
- Hidden size: 128 neurons
- Epochs: 5+ for convergence

References
----------
- LeCun et al., "Gradient-Based Learning Applied to Document Recognition"
- CS231n: Convolutional Neural Networks for Visual Recognition
- "Neural Networks and Deep Learning" by Michael Nielsen

Author: Prem Chand
"""

import numpy as np

from deeplearning.losses import CrossEntropyLoss
from deeplearning.neural_net import Linear, ReLU


class TwoLayerMLP:
    """
    A 2-layer Multi-Layer Perceptron for MNIST digit classification.

    This class encapsulates a complete neural network with:
    - Two fully connected layers
    - ReLU activation between layers
    - Cross-entropy loss for training

    The architecture is designed for the MNIST dataset (28x28 grayscale images
    flattened to 784-dimensional vectors, 10 output classes).

    Architecture Details
    --------------------
    Layer 1: Linear(784, 128)
        - Input: Flattened 28x28 image (784 pixels)
        - Output: 128 hidden features
        - Parameters: 784 * 128 + 128 = 100,480

    Activation: ReLU
        - Introduces non-linearity
        - No learnable parameters

    Layer 2: Linear(128, 10)
        - Input: 128 hidden features
        - Output: 10 class logits
        - Parameters: 128 * 10 + 10 = 1,290

    Total Parameters: 101,770

    Attributes
    ----------
    layers : list
        Ordered list of layer objects [Linear, ReLU, Linear].
    loss_fn : CrossEntropyLoss
        Loss function for computing training loss and gradients.

    Examples
    --------
    >>> model = TwoLayerMLP()
    >>>
    >>> # Forward pass only (inference)
    >>> x = np.random.randn(32, 784)
    >>> logits = model.forward(x)
    >>> predictions = np.argmax(logits, axis=1)
    >>>
    >>> # Full training step
    >>> y = np.random.randint(0, 10, 32)
    >>> loss = model.train_step(x, y, lr=0.01)

    Notes
    -----
    - The model expects inputs normalized to have zero mean and unit variance
    - For best results, use mini-batch SGD with batch sizes of 32-128
    - Learning rate of 0.01 is a good starting point

    See Also
    --------
    Linear : Fully connected layer implementation
    ReLU : ReLU activation implementation
    CrossEntropyLoss : Loss function implementation
    """

    def __init__(self):
        """
        Initialize the 2-layer MLP with random weights.

        The network architecture is fixed:
        - Input size: 784 (MNIST image size)
        - Hidden size: 128
        - Output size: 10 (number of digit classes)

        Weight initialization uses Kaiming He initialization,
        which is appropriate for ReLU activations.
        """
        self.layers = [
            Linear(784, 128),  # Input → Hidden
            ReLU(),            # Non-linearity
            Linear(128, 10)    # Hidden → Output
        ]
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through all layers.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (N, 784), where N is batch size.
            Each row is a flattened 28x28 MNIST image.

        Returns
        -------
        np.ndarray
            Output logits of shape (N, 10). These are raw scores
            before softmax; use argmax for predictions.

        Data Flow
        ---------
        x (N, 784)
            ↓ Linear(784, 128)
        h1 (N, 128)
            ↓ ReLU
        h1_activated (N, 128)
            ↓ Linear(128, 10)
        logits (N, 10)

        Notes
        -----
        The forward pass caches intermediate values in each layer,
        which are needed for the backward pass.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dL_dy: np.ndarray) -> None:
        """
        Perform backward pass through all layers (in reverse order).

        Parameters
        ----------
        dL_dy : np.ndarray
            Gradient of loss with respect to network output,
            shape (N, 10). Typically obtained from loss_fn.backward().

        Notes
        -----
        This method:
        1. Propagates gradients backward through each layer
        2. Computes dL/dW and dL/db for each Linear layer
        3. Stores parameter gradients in layer.dL_dW and layer.dL_db

        The gradient for each layer is the upstream gradient times
        the local gradient (chain rule).
        """
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)

    def update_parameters(self, lr: float) -> None:
        """
        Update parameters using Stochastic Gradient Descent (SGD).

        Parameters
        ----------
        lr : float
            Learning rate. Controls the step size of parameter updates.
            Typical values: 0.001 to 0.1.

        Algorithm
        ---------
        For each parameter θ with gradient ∇θ:
            θ = θ - lr * ∇θ

        This is vanilla SGD without momentum or adaptive learning rates.
        More sophisticated optimizers (Adam, RMSprop) would track
        additional state and modify the update rule.

        Notes
        -----
        Only Linear layers have parameters to update.
        ReLU has no learnable parameters.
        """
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer._w -= lr * layer.dL_dW
                layer._b -= lr * layer.dL_db

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray, lr: float) -> float:
        """
        Perform a single training iteration (forward, loss, backward, update).

        This is the core training loop condensed into a single method.
        It performs all four steps of neural network training:

        1. Forward pass: compute predictions
        2. Loss computation: measure prediction error
        3. Backward pass: compute gradients
        4. Parameter update: adjust weights

        Parameters
        ----------
        x_batch : np.ndarray
            Batch of input images, shape (N, 784).
        y_batch : np.ndarray
            Batch of target labels, shape (N,) with values in [0, 9].
        lr : float
            Learning rate for SGD update.

        Returns
        -------
        float
            Scalar loss value for this batch (cross-entropy loss).

        Example
        -------
        >>> model = TwoLayerMLP()
        >>> x = np.random.randn(64, 784)
        >>> y = np.random.randint(0, 10, 64)
        >>> loss = model.train_step(x, y, lr=0.01)
        >>> print(f"Batch loss: {loss:.4f}")

        Notes
        -----
        The returned loss is computed BEFORE the parameter update,
        so it reflects the model's performance on this batch before
        learning from it.
        """
        # Step 1: Forward pass
        y_pred = self.forward(x_batch)

        # Step 2: Compute loss
        loss = self.loss_fn.forward(y_pred, y_batch)

        # Step 3: Backward pass (compute gradients)
        dL_dy = self.loss_fn.backward()
        self.backward(dL_dy)

        # Step 4: Update parameters
        self.update_parameters(lr)

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Get class predictions for input samples.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (N, 784).

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (N,), values in [0, 9].

        Example
        -------
        >>> model = TwoLayerMLP()
        >>> x = np.random.randn(10, 784)
        >>> predictions = model.predict(x)
        >>> predictions.shape
        (10,)
        """
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on a dataset.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (N, 784).
        y : np.ndarray
            True labels of shape (N,).

        Returns
        -------
        float
            Accuracy as a fraction in [0, 1].

        Example
        -------
        >>> model = TwoLayerMLP()
        >>> x_test = np.random.randn(1000, 784)
        >>> y_test = np.random.randint(0, 10, 1000)
        >>> accuracy = model.evaluate(x_test, y_test)
        >>> print(f"Accuracy: {accuracy * 100:.2f}%")
        """
        predictions = self.predict(x)
        return np.mean(predictions == y)


if __name__ == "__main__":
    """
    Demonstrate the MLP with synthetic data.

    This example shows the basic usage pattern:
    1. Create the model
    2. Generate random data (substitute with real MNIST data)
    3. Run a training step
    4. Observe the loss

    For real MNIST training, you would:
    1. Load the MNIST dataset
    2. Normalize the images (subtract mean, divide by std)
    3. Split into batches
    4. Train for multiple epochs
    5. Evaluate on test set
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    # Initialize model
    model = TwoLayerMLP()

    # Create dummy data (would be replaced with real MNIST data)
    # Shape: (batch_size, 784) for images, (batch_size,) for labels
    x_dummy = np.random.randn(64, 784)  # Batch of 64 samples
    y_dummy = np.random.randint(0, 10, size=(64,))  # Random class labels

    # Perform one training step
    loss = model.train_step(x_dummy, y_dummy, lr=0.01)
    print(f"Training loss: {loss:.4f}")

    # Check predictions
    predictions = model.predict(x_dummy)
    accuracy = model.evaluate(x_dummy, y_dummy)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    # Note: With random data, accuracy should be ~10% (random chance)
    # With real MNIST data and proper training, expect >95% test accuracy
