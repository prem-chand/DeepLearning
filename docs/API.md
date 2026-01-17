# Deep Learning from Scratch - API Reference

This document provides detailed API documentation for all classes and functions in the `deeplearning` package.

## Table of Contents

- [Neural Network Layers](#neural-network-layers)
  - [Linear](#linear)
  - [ReLU](#relu)
  - [Sigmoid](#sigmoid)
  - [Tanh](#tanh)
- [Loss Functions](#loss-functions)
  - [MSELoss](#mseloss)
  - [CrossEntropyLoss](#crossentropyloss)
- [Models](#models)
  - [TwoLayerMLP](#twolayermlp)
- [Utilities](#utilities)
  - [numerical_gradient](#numerical_gradient)
  - [gradient_check](#gradient_check)
  - [gradient_check_weights](#gradient_check_weights)

---

## Neural Network Layers

All layers follow the same interface:
- `forward(x)` - Compute output and cache values for backward pass
- `backward(dL_dy)` - Compute and return gradient w.r.t. input

### Linear

```python
from deeplearning import Linear

layer = Linear(in_features: int, out_features: int)
```

Fully connected (dense) layer implementing `y = Wx + b`.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_features` | int | Number of input features |
| `out_features` | int | Number of output features |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_w` | np.ndarray | Weight matrix, shape `(out_features, in_features)` |
| `_b` | np.ndarray | Bias vector, shape `(out_features,)` |
| `dL_dW` | np.ndarray | Gradient w.r.t. weights after backward pass |
| `dL_db` | np.ndarray | Gradient w.r.t. biases after backward pass |

#### Methods

##### `forward(x: np.ndarray) -> np.ndarray`

Compute the forward pass: `y = x @ W^T + b`

**Parameters:**
- `x` (np.ndarray): Input tensor of shape `(N, in_features)`

**Returns:**
- np.ndarray: Output tensor of shape `(N, out_features)`

##### `backward(dL_dy: np.ndarray) -> np.ndarray`

Compute gradients using the chain rule.

**Parameters:**
- `dL_dy` (np.ndarray): Upstream gradient, shape `(N, out_features)`

**Returns:**
- np.ndarray: Downstream gradient w.r.t. input, shape `(N, in_features)`

**Side Effects:**
- Stores `dL_dW` and `dL_db` as instance attributes

#### Example

```python
import numpy as np
from deeplearning import Linear

# Create layer
layer = Linear(784, 128)

# Forward pass
x = np.random.randn(32, 784)  # Batch of 32
y = layer.forward(x)
print(f"Output shape: {y.shape}")  # (32, 128)

# Backward pass
dL_dy = np.random.randn(32, 128)
dL_dx = layer.backward(dL_dy)
print(f"Input gradient shape: {dL_dx.shape}")  # (32, 784)

# Access parameter gradients for SGD update
layer._w -= 0.01 * layer.dL_dW
layer._b -= 0.01 * layer.dL_db
```

---

### ReLU

```python
from deeplearning import ReLU

layer = ReLU()
```

Rectified Linear Unit activation: `y = max(0, x)`

#### Methods

##### `forward(x: np.ndarray) -> np.ndarray`

Apply ReLU elementwise: zeros out negative values.

**Parameters:**
- `x` (np.ndarray): Input tensor of any shape

**Returns:**
- np.ndarray: Output tensor of same shape

##### `backward(dL_dy: np.ndarray) -> np.ndarray`

Gradient is 1 where `x > 0`, else 0.

**Parameters:**
- `dL_dy` (np.ndarray): Upstream gradient

**Returns:**
- np.ndarray: Downstream gradient

#### Example

```python
import numpy as np
from deeplearning import ReLU

relu = ReLU()

x = np.array([[-1, 0, 1], [2, -2, 3]])
y = relu.forward(x)
print(y)
# [[0, 0, 1],
#  [2, 0, 3]]

dL_dy = np.ones_like(y)
dL_dx = relu.backward(dL_dy)
print(dL_dx)
# [[0, 0, 1],
#  [1, 0, 1]]
```

---

### Sigmoid

```python
from deeplearning import Sigmoid

layer = Sigmoid()
```

Sigmoid activation: `y = 1 / (1 + exp(-x))`

Output range: `(0, 1)`

#### Methods

##### `forward(x: np.ndarray) -> np.ndarray`

**Parameters:**
- `x` (np.ndarray): Input tensor of any shape

**Returns:**
- np.ndarray: Output tensor with values in `(0, 1)`

##### `backward(dL_dy: np.ndarray) -> np.ndarray`

Gradient: `dL/dx = dL/dy * y * (1 - y)`

**Parameters:**
- `dL_dy` (np.ndarray): Upstream gradient

**Returns:**
- np.ndarray: Downstream gradient

#### Example

```python
import numpy as np
from deeplearning import Sigmoid

sigmoid = Sigmoid()

x = np.array([[0, 1, -1]])
y = sigmoid.forward(x)
print(y)  # [[0.5, 0.731, 0.269]]
```

---

### Tanh

```python
from deeplearning import Tanh

layer = Tanh()
```

Hyperbolic tangent activation: `y = tanh(x)`

Output range: `(-1, 1)`

#### Methods

##### `forward(x: np.ndarray) -> np.ndarray`

**Parameters:**
- `x` (np.ndarray): Input tensor of any shape

**Returns:**
- np.ndarray: Output tensor with values in `(-1, 1)`

##### `backward(dL_dy: np.ndarray) -> np.ndarray`

Gradient: `dL/dx = dL/dy * (1 - y^2)`

**Parameters:**
- `dL_dy` (np.ndarray): Upstream gradient

**Returns:**
- np.ndarray: Downstream gradient

#### Example

```python
import numpy as np
from deeplearning import Tanh

tanh = Tanh()

x = np.array([[0, 1, -1]])
y = tanh.forward(x)
print(y)  # [[0.0, 0.762, -0.762]]
```

---

## Loss Functions

All loss functions follow the same interface:
- `forward(y_pred, y_true)` - Compute scalar loss
- `backward()` - Compute gradient w.r.t. predictions

### MSELoss

```python
from deeplearning import MSELoss

loss_fn = MSELoss()
```

Mean Squared Error for regression tasks.

**Formula:** `L = (1/N) * sum((y_pred - y_true)^2)`

#### Methods

##### `forward(y_pred: np.ndarray, y_true: np.ndarray) -> float`

Compute MSE loss.

**Parameters:**
- `y_pred` (np.ndarray): Predicted values
- `y_true` (np.ndarray): Ground truth values (same shape as `y_pred`)

**Returns:**
- float: Scalar loss value

##### `backward() -> np.ndarray`

Compute gradient: `dL/dy_pred = (2/N) * (y_pred - y_true)`

**Returns:**
- np.ndarray: Gradient of same shape as `y_pred`

#### Example

```python
import numpy as np
from deeplearning import MSELoss

mse = MSELoss()

y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
y_true = np.array([[1.5, 2.5], [2.5, 3.5]])

loss = mse.forward(y_pred, y_true)
print(f"Loss: {loss}")  # 0.25

grad = mse.backward()
print(f"Gradient shape: {grad.shape}")  # (2, 2)
```

---

### CrossEntropyLoss

```python
from deeplearning import CrossEntropyLoss

loss_fn = CrossEntropyLoss()
```

Cross-Entropy loss combining Softmax and Negative Log-Likelihood. The standard loss for multi-class classification.

**Formula:**
- Softmax: `p_i = exp(x_i) / sum(exp(x_j))`
- Loss: `L = -(1/N) * sum(log(p[target]))`

#### Methods

##### `forward(logits: np.ndarray, targets: np.ndarray) -> float`

Compute cross-entropy loss.

**Parameters:**
- `logits` (np.ndarray): Raw scores, shape `(N, C)` where N=batch size, C=classes
- `targets` (np.ndarray): Either:
  - Class indices of shape `(N,)` with values in `[0, C-1]`
  - One-hot encoded targets of shape `(N, C)`

**Returns:**
- float: Scalar loss value

##### `backward() -> np.ndarray`

Compute gradient: `dL/dlogits = (1/N) * (softmax(logits) - one_hot(targets))`

**Returns:**
- np.ndarray: Gradient of shape `(N, C)`

#### Example

```python
import numpy as np
from deeplearning import CrossEntropyLoss

ce_loss = CrossEntropyLoss()

# Class indices
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
targets = np.array([0, 1])  # Correct classes: 0 and 1

loss = ce_loss.forward(logits, targets)
print(f"Loss: {loss:.4f}")  # ~0.4076

grad = ce_loss.backward()
print(f"Gradient shape: {grad.shape}")  # (2, 3)

# One-hot targets also work
targets_onehot = np.array([[1, 0, 0], [0, 1, 0]])
loss = ce_loss.forward(logits, targets_onehot)
```

---

## Models

### TwoLayerMLP

```python
from deeplearning import TwoLayerMLP

model = TwoLayerMLP()
```

A 2-layer Multi-Layer Perceptron for MNIST digit classification.

**Architecture:**
```
Input(784) → Linear(128) → ReLU → Linear(10) → Softmax(CrossEntropy)
```

**Parameters:** 101,770 total
- Layer 1: 784 × 128 + 128 = 100,480
- Layer 2: 128 × 10 + 10 = 1,290

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `layers` | list | `[Linear(784,128), ReLU(), Linear(128,10)]` |
| `loss_fn` | CrossEntropyLoss | Loss function for training |

#### Methods

##### `forward(x: np.ndarray) -> np.ndarray`

Compute forward pass through all layers.

**Parameters:**
- `x` (np.ndarray): Input images, shape `(N, 784)`

**Returns:**
- np.ndarray: Output logits, shape `(N, 10)`

##### `backward(dL_dy: np.ndarray) -> None`

Backpropagate gradients through all layers.

**Parameters:**
- `dL_dy` (np.ndarray): Gradient from loss, shape `(N, 10)`

##### `update_parameters(lr: float) -> None`

Update weights using SGD: `w = w - lr * gradient`

**Parameters:**
- `lr` (float): Learning rate

##### `train_step(x_batch: np.ndarray, y_batch: np.ndarray, lr: float) -> float`

Perform one complete training iteration.

**Parameters:**
- `x_batch` (np.ndarray): Input images, shape `(N, 784)`
- `y_batch` (np.ndarray): Labels, shape `(N,)` with values in `[0, 9]`
- `lr` (float): Learning rate

**Returns:**
- float: Loss value for this batch

##### `predict(x: np.ndarray) -> np.ndarray`

Get class predictions.

**Parameters:**
- `x` (np.ndarray): Input images, shape `(N, 784)`

**Returns:**
- np.ndarray: Predicted class labels, shape `(N,)`

##### `evaluate(x: np.ndarray, y: np.ndarray) -> float`

Compute classification accuracy.

**Parameters:**
- `x` (np.ndarray): Input images
- `y` (np.ndarray): True labels

**Returns:**
- float: Accuracy in range `[0, 1]`

#### Example

```python
import numpy as np
from deeplearning import TwoLayerMLP

model = TwoLayerMLP()

# Generate sample data
x = np.random.randn(64, 784)
y = np.random.randint(0, 10, 64)

# Training loop
for epoch in range(10):
    loss = model.train_step(x, y, lr=0.01)
    acc = model.evaluate(x, y)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc*100:.1f}%")

# Inference
predictions = model.predict(x)
```

---

## Utilities

### numerical_gradient

```python
from tests.test_gradients import numerical_gradient

grad = numerical_gradient(f, x, epsilon=1e-5)
```

Compute numerical gradient using centered finite differences.

**Formula:** `df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)`

**Parameters:**
- `f` (callable): Function returning a scalar
- `x` (np.ndarray): Point at which to evaluate gradient
- `epsilon` (float): Perturbation size (default: 1e-5)

**Returns:**
- np.ndarray: Numerical gradient of same shape as `x`

---

### gradient_check

```python
from tests.test_gradients import gradient_check

passed = gradient_check(layer, x, tolerance=1e-5)
```

Verify a layer's backward pass against numerical gradients.

**Parameters:**
- `layer`: A layer with `forward()` and `backward()` methods
- `x` (np.ndarray): Input tensor
- `tolerance` (float): Max allowed relative error

**Returns:**
- bool: True if gradient check passes

---

### gradient_check_weights

```python
from tests.test_gradients import gradient_check_weights

passed = gradient_check_weights(layer, x, tolerance=1e-5)
```

Verify gradients w.r.t. layer weights (for Linear layers).

**Parameters:**
- `layer` (Linear): A Linear layer
- `x` (np.ndarray): Input tensor
- `tolerance` (float): Max allowed relative error

**Returns:**
- bool: True if all parameter gradients pass

---

## Quick Reference

### Import Patterns

```python
# Import all public classes
from deeplearning import (
    Linear, ReLU, Sigmoid, Tanh,
    MSELoss, CrossEntropyLoss,
    TwoLayerMLP
)

# Or import from specific modules
from deeplearning.neural_net import Linear, ReLU
from deeplearning.losses import CrossEntropyLoss
from deeplearning.mnist_mlp import TwoLayerMLP
```

### Common Patterns

```python
# Build a custom network
layers = [
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
]

# Forward pass
x = input_data
for layer in layers:
    x = layer.forward(x)
logits = x

# Backward pass
loss_fn = CrossEntropyLoss()
loss = loss_fn.forward(logits, targets)
grad = loss_fn.backward()

for layer in reversed(layers):
    grad = layer.backward(grad)

# Update parameters
lr = 0.01
for layer in layers:
    if hasattr(layer, '_w'):
        layer._w -= lr * layer.dL_dW
        layer._b -= lr * layer.dL_db
```
