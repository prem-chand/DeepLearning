# Deep Learning from Scratch

[![CI](https://github.com/premchand/DeepLearning/actions/workflows/ci.yml/badge.svg)](https://github.com/premchand/DeepLearning/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A pedagogical implementation of neural networks built entirely from scratch using NumPy. This project demonstrates the fundamental concepts of deep learning, including forward propagation, backpropagation, and gradient descent optimization.

## Overview

This repository contains a complete implementation of:

- **Neural Network Layers**: Linear (fully connected), ReLU, Sigmoid, Tanh
- **Loss Functions**: Mean Squared Error (MSE), Cross-Entropy with Softmax
- **Gradient Checking**: Numerical validation of backpropagation
- **Training Pipeline**: Complete 2-layer MLP for MNIST classification

The goal is to understand what happens "under the hood" of frameworks like PyTorch and TensorFlow by implementing everything manually.

## Project Structure

```
DeepLearning/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── src/
│   └── deeplearning/
│       ├── __init__.py         # Package exports
│       ├── neural_net.py       # Core layers (Linear, ReLU, Sigmoid, Tanh)
│       ├── losses.py           # Loss functions (MSE, CrossEntropy)
│       └── mnist_mlp.py        # 2-layer MLP implementation
├── tests/
│   ├── __init__.py
│   ├── test_layers.py          # Layer unit tests
│   ├── test_losses.py          # Loss function tests
│   ├── test_mlp.py             # MLP integration tests
│   └── test_gradients.py       # Gradient checking utilities
├── .gitignore
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── LICENSE                     # MIT License
├── README.md                   # This file
├── phase1.md                   # Detailed assignment specification
└── pyproject.toml              # Project configuration
```

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/premchand/DeepLearning.git
cd DeepLearning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Basic Installation

```bash
pip install -e .
```

## Quick Start

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/deeplearning --cov-report=html

# Run specific test file
pytest tests/test_layers.py -v
```

### Run Gradient Checks

```bash
python tests/test_gradients.py
```

Expected output:

```text
============================================================
GRADIENT CHECKING VALIDATION
============================================================

Testing Linear layer (input gradient):
----------------------------------------
Max Relative Error: 3.09e-08
✓ Gradient check PASSED!

Testing ReLU activation:
----------------------------------------
Max Relative Error: 2.55e-11
✓ Gradient check PASSED!
...
```

### Train the MLP

```bash
python -c "from deeplearning.mnist_mlp import TwoLayerMLP; import numpy as np; np.random.seed(0); model = TwoLayerMLP(); x = np.random.randn(64, 784); y = np.random.randint(0, 10, 64); loss = model.train_step(x, y, lr=0.01); print(f'Loss: {loss:.4f}')"
```

## Usage Examples

### Using Neural Network Layers

```python
from deeplearning import Linear, ReLU
import numpy as np

# Create layers
linear = Linear(in_features=784, out_features=128)
relu = ReLU()

# Forward pass
x = np.random.randn(32, 784)  # Batch of 32
h = linear.forward(x)
h_activated = relu.forward(h)

# Backward pass
dL_dh = np.random.randn(*h_activated.shape)  # Upstream gradient
dL_dh_pre = relu.backward(dL_dh)
dL_dx = linear.backward(dL_dh_pre)

# Access weight gradients for optimization
print(f"Weight gradient shape: {linear.dL_dW.shape}")
print(f"Bias gradient shape: {linear.dL_db.shape}")
```

### Using Loss Functions

```python
from deeplearning import CrossEntropyLoss
import numpy as np

loss_fn = CrossEntropyLoss()

# Forward: compute loss
logits = np.random.randn(32, 10)  # 32 samples, 10 classes
targets = np.random.randint(0, 10, 32)  # Class labels
loss = loss_fn.forward(logits, targets)

# Backward: get gradients
dL_dlogits = loss_fn.backward()
```

### Training Pipeline

```python
from deeplearning import TwoLayerMLP
import numpy as np

# Create model
model = TwoLayerMLP()

# Architecture: Input(784) → Linear(128) → ReLU → Linear(10)
# Total parameters: 101,770

# Training step
x_batch = np.random.randn(64, 784)
y_batch = np.random.randint(0, 10, 64)
loss = model.train_step(x_batch, y_batch, lr=0.01)

# Inference
predictions = model.predict(x_batch)
accuracy = model.evaluate(x_batch, y_batch)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## Development

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:

```bash
# Install hooks (run once)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Hooks include:

- **Ruff**: Linting and formatting
- **mypy**: Type checking
- **pytest**: Run tests before commit
- General checks: trailing whitespace, YAML validation, etc.

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### Type Checking

```bash
mypy src/ --ignore-missing-imports
```

## Architecture Diagram

```
                    FORWARD PASS
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   Input        Linear      ReLU      Linear     │
    │  (N, 784)  →  (N, 128)  →  (N, 128)  →  (N, 10) │
    │                                                  │
    │              ↓ W₁, b₁               ↓ W₂, b₂    │
    └──────────────────────────────────────────────────┘
                         │
                         ↓
                  CrossEntropyLoss
                         │
                         ↓ Loss (scalar)

                    BACKWARD PASS
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │  dL/dx      dL/dh₁     dL/dh₁'    dL/dlogits   │
    │   ←          ←           ←            ←         │
    │                                                  │
    │       ↑ dL/dW₁           ↑ dL/dW₂              │
    │       ↑ dL/db₁           ↑ dL/db₂              │
    └──────────────────────────────────────────────────┘
```

## Mathematical Background

### Backpropagation

The chain rule enables computing gradients through composed functions:

```text
dL/dx = dL/dy * dy/dx
```

For a network with layers f₁, f₂, ..., fₙ:

```text
y = fₙ(fₙ₋₁(...f₁(x)))
dL/dx = (∂fₙ/∂fₙ₋₁) * (∂fₙ₋₁/∂fₙ₋₂) * ... * (∂f₁/∂x)
```

### Gradient Checking

Numerical gradients validate analytical implementations:

```text
∂f/∂x ≈ [f(x + ε) - f(x - ε)] / (2ε)
```

This centered difference has O(ε²) error, making it accurate enough to detect implementation bugs.

## CI/CD

GitHub Actions runs on every push and PR:

- **Tests**: pytest on Python 3.9-3.12, Ubuntu/macOS/Windows
- **Linting**: Ruff checks and formatting
- **Type Checking**: mypy static analysis
- **Pre-commit**: All hooks validation

## Performance Targets

With proper MNIST training:
- Training accuracy: >98%
- Test accuracy: >95%
- Convergence: ~5 epochs

## Future Extensions

The `phase1.md` file outlines additional implementations:

- **Autograd Engine**: Automatic differentiation with computational graphs
- **Optimizers**: SGD with Momentum, RMSprop, Adam
- **Batch Normalization**: Normalize activations for faster training
- **Learning Rate Schedulers**: Step decay, cosine annealing, warmup

## References

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) by Andrej Karpathy
- He et al., ["Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852) (Kaiming initialization)
- Goodfellow et al., "Deep Learning" (MIT Press)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
