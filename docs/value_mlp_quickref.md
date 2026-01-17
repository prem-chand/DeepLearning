# Value-based MLP - Quick Reference Card

## 🚀 Quick Start

```python
from deeplearning.value_mlp import ValueMLP, train_step_value_mlp, evaluate_value_mlp
import numpy as np

# Create model: [input_size, hidden_size, output_size]
model = ValueMLP([784, 32, 10])

# Train
x_train = np.random.randn(100, 784)
y_train = np.random.randint(0, 10, 100)
loss = train_step_value_mlp(model, x_train, y_train, lr=0.01)

# Evaluate
accuracy = evaluate_value_mlp(model, x_test, y_test)
```

## 📋 Command Cheat Sheet

```bash
# Run demos
python src/deeplearning/value_mlp.py              # Basic demo
python scripts/train_value_mlp_mnist.py quick     # MNIST quick (50 samples)
python scripts/train_value_mlp_mnist.py full      # MNIST full (200 samples)

# Run tests
python -m pytest tests/test_value_mlp.py -v       # MLP tests only
python -m pytest tests/test_autograd.py -v        # Autograd tests
python -m pytest tests/ -v                        # All tests (99 total)
```

## 🏗️ Architecture

```
Input → Dense → ReLU → Dense → Output
(784)   (128)          (10)
```

## 🔧 Key Classes

| Class | Purpose | Example |
|-------|---------|---------|
| `Neuron` | Single unit | `Neuron(n_inputs=10, activation='relu')` |
| `Layer` | Collection of neurons | `Layer(n_inputs=10, n_outputs=5)` |
| `ValueMLP` | Complete network | `ValueMLP([784, 128, 10])` |

## 📊 Training Results (Quick Demo)

```
Dataset: 50 samples, 3 epochs
Architecture: [784, 16, 10]
Parameters: 12,730

Epoch 1: Loss=2.27, Train Acc=84%
Epoch 2: Loss=0.92, Train Acc=92%
Epoch 3: Loss=0.35, Train Acc=94%
```

## ✅ Test Coverage

- **26 tests** for Value-based MLP
- **41 tests** for Value class
- **99 tests** total (all passing)

## 📖 Documentation

- `docs/value_mlp_mnist.md` - Complete guide
- `docs/value_mlp_summary.md` - Project summary
- `docs/value_class_reference.md` - Value class reference

## ⚡ Performance

| Metric | Value |
|--------|-------|
| Speed | ~20s per epoch (50 samples) |
| Memory | High (graph overhead) |
| Use case | Educational, small datasets |

## 🎯 Key Features

✅ Scalar-based autograd
✅ Multiple activations (ReLU, tanh, sigmoid)
✅ Softmax cross-entropy loss
✅ SGD optimizer
✅ Complete training pipeline
✅ Comprehensive tests

## 🔄 vs Vectorized Implementation

| Aspect | Value-based | Vectorized |
|--------|-------------|------------|
| Speed | Slow | Fast |
| Clarity | Very clear | Abstracted |
| Use | Education | Production |

## 💡 Tips

1. **Use small datasets** - 50-200 samples for practical training time
2. **Adjust learning rate** - 0.01-0.1 works well
3. **Monitor loss** - Should decrease steadily
4. **Check gradients** - Use `model.parameters()` to inspect

## 🐛 Debugging

```python
# Check parameter count
print(f"Parameters: {model.num_parameters()}")

# Inspect gradients
for i, p in enumerate(model.parameters()):
    print(f"Param {i}: data={p.data:.4f}, grad={p.grad:.4f}")

# Zero gradients between runs
model.zero_grad()
```

## 📝 Common Patterns

### Training Loop
```python
for epoch in range(epochs):
    loss = train_step_value_mlp(model, x_train, y_train, lr=0.01)
    acc = evaluate_value_mlp(model, x_test, y_test)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc*100:.1f}%")
```

### Custom Architecture
```python
# Small network
model = ValueMLP([784, 16, 10])

# Larger network
model = ValueMLP([784, 128, 64, 10])

# Deep network
model = ValueMLP([784, 256, 128, 64, 32, 10])
```

## 🎓 Educational Use

**Perfect for learning:**
- How backpropagation works
- Gradient flow through networks
- Automatic differentiation
- Neural network fundamentals

**Not for:**
- Large-scale training
- Production applications
- Full MNIST (60k samples)
- Performance-critical tasks
