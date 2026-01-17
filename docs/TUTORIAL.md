# Deep Learning from Scratch - Tutorial

A step-by-step guide to understanding and using this neural network implementation.

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Neural Network Basics](#understanding-neural-network-basics)
3. [Building Blocks](#building-blocks)
4. [Training a Neural Network](#training-a-neural-network)
5. [MNIST Digit Classification](#mnist-digit-classification)
6. [Gradient Checking](#gradient-checking)
7. [Extending the Library](#extending-the-library)

---

## Introduction

This library implements neural networks from scratch using only NumPy. The goal is to understand what happens "under the hood" of frameworks like PyTorch and TensorFlow.

### Prerequisites

- Python 3.9+
- NumPy basics (arrays, broadcasting, matrix multiplication)
- Basic calculus (derivatives, chain rule)

### Installation

```bash
# Clone and install
git clone https://github.com/premchand/DeepLearning.git
cd DeepLearning
pip install -e .
```

---

## Understanding Neural Network Basics

### What is a Neural Network?

A neural network is a function `f(x; θ)` that maps inputs `x` to outputs, parameterized by weights `θ`. It's composed of layers, each performing a simple computation.

### The Two Phases

1. **Forward Pass**: Compute output from input
   ```
   Input → Layer 1 → Layer 2 → ... → Output
   ```

2. **Backward Pass**: Compute gradients using chain rule
   ```
   Output ← Layer N ← Layer N-1 ← ... ← Input
   ```

### Training Loop

```python
for each batch:
    # 1. Forward: compute predictions
    predictions = network(inputs)

    # 2. Loss: measure error
    loss = loss_function(predictions, targets)

    # 3. Backward: compute gradients
    gradients = compute_gradients(loss)

    # 4. Update: adjust weights
    weights = weights - learning_rate * gradients
```

---

## Building Blocks

### Linear Layer

The Linear (fully connected) layer performs: `y = Wx + b`

```python
import numpy as np
from deeplearning import Linear

# Create a layer: 784 inputs → 128 outputs
layer = Linear(in_features=784, out_features=128)

# Forward pass
x = np.random.randn(32, 784)  # Batch of 32 samples
y = layer.forward(x)          # Shape: (32, 128)

# Backward pass (given upstream gradient)
dL_dy = np.random.randn(32, 128)  # Gradient from next layer
dL_dx = layer.backward(dL_dy)     # Shape: (32, 784)

# After backward, gradients are stored
print(f"Weight gradient: {layer.dL_dW.shape}")  # (128, 784)
print(f"Bias gradient: {layer.dL_db.shape}")    # (128,)
```

**Mathematical Details:**

Forward:
```
y = x @ W.T + b
```

Backward (given dL/dy):
```
dL/dx = dL/dy @ W        # Propagate to previous layer
dL/dW = dL/dy.T @ x      # Gradient for weights
dL/db = sum(dL/dy)       # Gradient for biases
```

### Activation Functions

Activations introduce non-linearity. Without them, multiple linear layers collapse to a single linear transformation.

#### ReLU (Rectified Linear Unit)

```python
from deeplearning import ReLU

relu = ReLU()

# Forward: y = max(0, x)
x = np.array([[-2, -1, 0, 1, 2]])
y = relu.forward(x)  # [[0, 0, 0, 1, 2]]

# Backward: gradient is 1 where x > 0, else 0
dL_dy = np.array([[1, 1, 1, 1, 1]])
dL_dx = relu.backward(dL_dy)  # [[0, 0, 0, 1, 1]]
```

#### Sigmoid

```python
from deeplearning import Sigmoid

sigmoid = Sigmoid()

# Forward: y = 1 / (1 + exp(-x))
# Maps any value to (0, 1)
x = np.array([[-2, 0, 2]])
y = sigmoid.forward(x)  # [[0.12, 0.5, 0.88]]

# Backward: dL/dx = dL/dy * y * (1 - y)
```

#### Tanh

```python
from deeplearning import Tanh

tanh = Tanh()

# Forward: y = tanh(x)
# Maps any value to (-1, 1)
x = np.array([[-2, 0, 2]])
y = tanh.forward(x)  # [[-0.96, 0, 0.96]]

# Backward: dL/dx = dL/dy * (1 - y^2)
```

### Loss Functions

Loss functions measure how wrong our predictions are.

#### Mean Squared Error (MSE)

For regression problems:

```python
from deeplearning import MSELoss

mse = MSELoss()

# Predictions and targets
y_pred = np.array([[2.5, 3.0]])
y_true = np.array([[2.0, 3.5]])

# Forward: L = mean((y_pred - y_true)^2)
loss = mse.forward(y_pred, y_true)  # 0.25

# Backward: dL/dy_pred = 2 * (y_pred - y_true) / N
grad = mse.backward()
```

#### Cross-Entropy (for Classification)

For multi-class classification:

```python
from deeplearning import CrossEntropyLoss

ce = CrossEntropyLoss()

# Raw scores (logits) for 3 classes, batch of 2
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3]])
targets = np.array([0, 1])  # Correct classes

# Forward: softmax + negative log likelihood
loss = ce.forward(logits, targets)

# Backward: elegant gradient = softmax - one_hot
grad = ce.backward()
```

---

## Training a Neural Network

### Building a Simple Network

```python
import numpy as np
from deeplearning import Linear, ReLU, CrossEntropyLoss

# Architecture: 784 → 128 → 10
layer1 = Linear(784, 128)
relu = ReLU()
layer2 = Linear(128, 10)
loss_fn = CrossEntropyLoss()

def forward(x):
    h = layer1.forward(x)
    h = relu.forward(h)
    return layer2.forward(h)

def backward(grad):
    grad = layer2.backward(grad)
    grad = relu.backward(grad)
    grad = layer1.backward(grad)
    return grad

def update(lr):
    layer1._w -= lr * layer1.dL_dW
    layer1._b -= lr * layer1.dL_db
    layer2._w -= lr * layer2.dL_dW
    layer2._b -= lr * layer2.dL_db
```

### Training Loop

```python
# Hyperparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 64

# Training data (placeholder)
X_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, 1000)

for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = len(X_train) // batch_size

    for i in range(num_batches):
        # Get batch
        start = i * batch_size
        end = start + batch_size
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward pass
        logits = forward(x_batch)

        # Compute loss
        loss = loss_fn.forward(logits, y_batch)
        epoch_loss += loss

        # Backward pass
        grad = loss_fn.backward()
        backward(grad)

        # Update parameters
        update(learning_rate)

    print(f"Epoch {epoch + 1}: Loss = {epoch_loss / num_batches:.4f}")
```

### Using TwoLayerMLP

The library provides a convenience class:

```python
from deeplearning import TwoLayerMLP

model = TwoLayerMLP()

# Single training step
loss = model.train_step(x_batch, y_batch, lr=0.01)

# Predictions
predictions = model.predict(x_batch)

# Evaluation
accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

---

## MNIST Digit Classification

### Loading MNIST Data

```python
import numpy as np

def load_mnist():
    """Load MNIST dataset from a local source or download."""
    # Using keras datasets (if available)
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    except ImportError:
        # Alternative: download manually
        raise ImportError("Please install tensorflow or download MNIST manually")

    # Preprocessing
    # 1. Flatten: (N, 28, 28) → (N, 784)
    X_train = X_train.reshape(-1, 784).astype(np.float32)
    X_test = X_test.reshape(-1, 784).astype(np.float32)

    # 2. Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 3. Standardize (zero mean, unit variance)
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return (X_train, y_train), (X_test, y_test)
```

### Training on MNIST

```python
from deeplearning import TwoLayerMLP
import numpy as np

# Load data
(X_train, y_train), (X_test, y_test) = load_mnist()

# Create model
model = TwoLayerMLP()

# Hyperparameters
epochs = 5
batch_size = 64
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Train on batches
    total_loss = 0
    num_batches = len(X_train) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        loss = model.train_step(x_batch, y_batch, lr=learning_rate)
        total_loss += loss

    # Evaluate
    train_acc = model.evaluate(X_train[:1000], y_train[:1000])
    test_acc = model.evaluate(X_test, y_test)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"  Loss: {total_loss / num_batches:.4f}")
    print(f"  Train Acc: {train_acc * 100:.2f}%")
    print(f"  Test Acc: {test_acc * 100:.2f}%")
```

### Expected Results

After ~5 epochs:
- Training accuracy: >98%
- Test accuracy: >95%

---

## Gradient Checking

Gradient checking validates your backward pass implementation.

### How It Works

1. Compute analytical gradient using `backward()`
2. Compute numerical gradient using finite differences
3. Compare them - they should match closely

### Using the Gradient Check Utilities

```python
from deeplearning import Linear, ReLU
from tests.test_gradients import gradient_check, gradient_check_weights
import numpy as np

np.random.seed(42)

# Test Linear layer input gradients
layer = Linear(10, 5)
x = np.random.randn(4, 10)

print("Checking Linear layer input gradient:")
if gradient_check(layer, x):
    print("✓ PASSED")
else:
    print("✗ FAILED")

# Test Linear layer weight gradients
print("\nChecking Linear layer weight gradients:")
if gradient_check_weights(layer, x):
    print("✓ PASSED")
else:
    print("✗ FAILED")
```

### Interpreting Results

```
Max Relative Error: X.XXe-YY
```

- `< 1e-7`: Excellent, implementation is correct
- `1e-7 to 1e-5`: Good, acceptable for float64
- `1e-5 to 1e-2`: Suspicious, review carefully
- `> 1e-2`: Bug detected, fix implementation

---

## Extending the Library

### Adding a New Activation Function

```python
import numpy as np

class LeakyReLU:
    """Leaky ReLU: y = max(alpha * x, x)"""

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self._x = None

    def forward(self, x):
        self._x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dL_dy):
        return np.where(self._x > 0, dL_dy, self.alpha * dL_dy)
```

### Adding a New Loss Function

```python
class HuberLoss:
    """Huber loss: combines MSE and MAE"""

    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_pred, y_true):
        self._y_pred = y_pred
        self._y_true = y_true

        diff = y_pred - y_true
        abs_diff = np.abs(diff)

        # Quadratic for small errors, linear for large
        loss = np.where(
            abs_diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        return np.mean(loss)

    def backward(self):
        diff = self._y_pred - self._y_true
        abs_diff = np.abs(diff)

        grad = np.where(
            abs_diff <= self.delta,
            diff,
            self.delta * np.sign(diff)
        )
        return grad / self._y_true.size
```

### Adding Optimizers

```python
class SGDMomentum:
    """SGD with momentum"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def update(self, params, grads, layer_id):
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'w': np.zeros_like(params['w']),
                'b': np.zeros_like(params['b'])
            }

        v = self.velocities[layer_id]

        # Update velocities
        v['w'] = self.momentum * v['w'] - self.lr * grads['dw']
        v['b'] = self.momentum * v['b'] - self.lr * grads['db']

        # Update parameters
        params['w'] += v['w']
        params['b'] += v['b']
```

---

## Summary

You've learned:

1. **Neural Network Basics**: Forward/backward passes, training loop
2. **Building Blocks**: Linear layers, activations (ReLU, Sigmoid, Tanh)
3. **Loss Functions**: MSE for regression, CrossEntropy for classification
4. **Training**: Mini-batch SGD, gradient descent
5. **Validation**: Gradient checking for debugging
6. **Extension**: Adding new components

### Next Steps

- Implement batch normalization
- Add dropout for regularization
- Try different optimizers (Adam, RMSprop)
- Experiment with deeper architectures
- Add convolutional layers for image tasks

### Resources

- [CS231n](http://cs231n.stanford.edu/) - Stanford's CNN course
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow et al.
