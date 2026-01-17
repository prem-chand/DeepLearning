# Phase 1: Neural Network Foundations - Implementation Assignment

**Estimated Time:** 2-3 weeks  
**Prerequisites:** Linear algebra, calculus, Python proficiency  
**Due Date:** Self-paced

---

## Assignment Overview

This assignment will guide you through implementing the foundational components of deep learning from scratch. You will build neural networks without relying on autograd frameworks, then create your own automatic differentiation engine.

**Learning Objectives:**
- Understand backpropagation through manual gradient derivation
- Implement core neural network layers and activation functions
- Build a computational graph for automatic differentiation
- Validate implementations against numerical methods
- Appreciate the abstractions provided by PyTorch/JAX

---

## Part 1: Backpropagation from Scratch (Week 1)

### 1.1 Mathematical Foundations

**Task:** Derive the gradients for the following operations on paper.

For each operation, given output gradient `dL/dy`, compute input gradients `dL/dx`, `dL/dW`, `dL/db` where applicable.

**Operations to derive:**

1. **Linear Layer:** `y = Wx + b`
   - Shapes: `x: (N, D_in)`, `W: (D_out, D_in)`, `b: (D_out,)`, `y: (N, D_out)`
   - Derive: `dL/dW`, `dL/db`, `dL/dx`

2. **ReLU:** `y = max(0, x)`
   - Derive: `dL/dx`

3. **Sigmoid:** `y = 1/(1 + exp(-x))`
   - Derive: `dL/dx` (express in terms of `y`)

4. **Tanh:** `y = tanh(x)`
   - Derive: `dL/dx` (express in terms of `y`)

5. **Mean Squared Error:** `L = (1/N) * sum((y_pred - y_true)^2)`
   - Derive: `dL/dy_pred`

6. **Cross-Entropy Loss (Softmax + NLL):**
   - Softmax: `p_i = exp(x_i) / sum_j(exp(x_j))`
   - Negative Log-Likelihood: `L = -log(p_target)`
   - Derive: `dL/dx` (combined gradient)

**Deliverable 1.1:** LaTeX or handwritten PDF with complete derivations

---

### 1.2 Implementation: Core Components

Create a file `neural_net.py` with the following classes:

```python
import numpy as np
from typing import Tuple, Optional

class Linear:
    """Fully connected layer: y = Wx + b"""
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize weights and biases.
        Use He initialization: W ~ N(0, 2/in_features)
        """
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor of shape (N, in_features)
        Returns:
            Output tensor of shape (N, out_features)
        """
        pass
    
    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        """
        Args:
            dL_dy: Gradient of loss w.r.t. output, shape (N, out_features)
        Returns:
            dL_dx: Gradient of loss w.r.t. input, shape (N, in_features)
        
        Side effects:
            Stores dL_dW and dL_db for optimizer update
        """
        pass

class ReLU:
    """ReLU activation: y = max(0, x)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        pass

class Sigmoid:
    """Sigmoid activation: y = 1/(1 + exp(-x))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        pass

class Tanh:
    """Tanh activation: y = tanh(x)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        pass
```

**Implementation Requirements:**
- Cache necessary values during forward pass for backward computation
- Use numerically stable implementations (e.g., clip values in sigmoid/tanh)
- All operations should be vectorized (no explicit loops over batch dimension)

**Deliverable 1.2:** Implemented `neural_net.py`

---

### 1.3 Gradient Checking

Implement numerical gradient checking to validate your backward passes.

```python
def numerical_gradient(f, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Compute numerical gradient using centered finite differences.
    
    Args:
        f: Function that takes x and returns scalar loss
        x: Point at which to evaluate gradient
        epsilon: Perturbation size
    
    Returns:
        Numerical gradient of same shape as x
    """
    pass

def gradient_check(layer, x: np.ndarray, tolerance: float = 1e-5) -> bool:
    """
    Verify backward pass against numerical gradient.
    
    Returns:
        True if max relative error < tolerance
    """
    pass
```

**Test Cases:**
Create `test_gradients.py` with tests for:
1. Linear layer (check `dL/dx`, `dL/dW`, `dL/db` separately)
2. ReLU (including points at x=0)
3. Sigmoid
4. Tanh
5. Combinations (e.g., Linear → ReLU → Linear)

**Deliverable 1.3:** Passing gradient checks with relative error < 1e-5

---

### 1.4 Loss Functions

Implement loss functions in `losses.py`:

```python
class MSELoss:
    """Mean Squared Error Loss"""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Returns scalar loss"""
        pass
    
    def backward(self) -> np.ndarray:
        """Returns dL/dy_pred"""
        pass

class CrossEntropyLoss:
    """Softmax + Negative Log-Likelihood (numerically stable)"""
    
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Args:
            logits: Raw scores of shape (N, C)
            targets: Class indices of shape (N,) or one-hot (N, C)
        Returns:
            Scalar loss
        """
        pass
    
    def backward(self) -> np.ndarray:
        """Returns dL/dlogits"""
        pass
```

**Numerical Stability Requirements:**
- CrossEntropy: Use log-sum-exp trick for softmax
- Handle edge cases (log(0), division by zero)

**Deliverable 1.4:** Implemented loss functions with gradient checks

---

### 1.5 Build and Train a 2-Layer MLP

Create `mnist_mlp.py` to train a simple network on MNIST:

**Network Architecture:**
```
Input (784) → Linear(784, 128) → ReLU → Linear(128, 10) → CrossEntropy
```

**Training Requirements:**
- Use mini-batch SGD (batch size 64)
- Learning rate: 0.01
- Train for 5 epochs
- Report train/test accuracy after each epoch

**Implementation:**
```python
class TwoLayerMLP:
    def __init__(self):
        self.layers = [
            Linear(784, 128),
            ReLU(),
            Linear(128, 10)
        ]
        self.loss_fn = CrossEntropyLoss()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        pass
    
    def backward(self, dL_dy: np.ndarray) -> None:
        """Backward pass through all layers"""
        pass
    
    def update_parameters(self, lr: float) -> None:
        """SGD parameter update"""
        pass
    
    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray, lr: float) -> float:
        """Single training iteration, returns loss"""
        pass
```

**Deliverable 1.5:** 
- Training script with progress logging
- Final test accuracy > 95%
- Loss curve plot (train loss vs iterations)

---

## Part 2: Custom Autograd Engine (Week 2)

### 2.1 Understanding Computational Graphs

**Background Reading:**
- Study how PyTorch's autograd works
- Understand dynamic vs static computational graphs
- Review topological sorting algorithms

**Task:** Draw computational graphs by hand for:
1. `z = (x * y) + (y * w)`
2. `L = sum((Wx + b - y_true)^2)`
3. A 2-layer neural network forward pass

Label each node with:
- Operation type
- Local gradient (∂output/∂input)
- Data shape

**Deliverable 2.1:** PDF with hand-drawn graphs and gradient flow

---

### 2.2 Implement Value Class

Create `autograd.py` with a computational graph engine:

```python
from typing import Tuple, Set, List
import numpy as np

class Value:
    """
    Scalar value in computational graph.
    Stores data, gradient, and operation that created it.
    """
    
    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        """Implement: out = self + other"""
        pass
    
    def __mul__(self, other):
        """Implement: out = self * other"""
        pass
    
    def __pow__(self, other):
        """Implement: out = self ** other (other is a constant)"""
        pass
    
    def relu(self):
        """Implement: out = max(0, self)"""
        pass
    
    def exp(self):
        """Implement: out = e^self"""
        pass
    
    def log(self):
        """Implement: out = log(self)"""
        pass
    
    def backward(self):
        """
        Compute gradients for all nodes in graph via backpropagation.
        Use topological sort to process nodes in correct order.
        """
        pass
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

**Implementation Guide:**

For each operation, you need to:
1. Compute the output value
2. Store the backward function that computes gradients

Example for addition:
```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
        self.grad += out.grad  # dL/dself = dL/dout * dout/dself = dL/dout * 1
        other.grad += out.grad  # dL/dother = dL/dout * dout/dother = dL/dout * 1
    
    out._backward = _backward
    return out
```

**Key Requirements:**
- Support both `Value + Value` and `Value + float`
- Implement `__radd__`, `__rmul__` for commutativity
- `backward()` should: initialize out.grad=1.0, topologically sort graph, call _backward() in reverse order
- Accumulate gradients (+=) to handle shared nodes

**Deliverable 2.2:** Implemented `Value` class

---

### 2.3 Tensor-Based Autograd (Advanced)

Extend to support tensors instead of just scalars:

```python
class Tensor:
    """
    Multi-dimensional array with autograd support.
    Similar to PyTorch tensors but simpler.
    """
    
    def __init__(self, data: np.ndarray, requires_grad: bool = False, _children: Tuple['Tensor', ...] = ()):
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __matmul__(self, other):
        """Matrix multiplication with gradient tracking"""
        pass
    
    def sum(self, axis=None, keepdims=False):
        """Sum with gradient broadcasting"""
        pass
    
    def reshape(self, *shape):
        """Reshape with gradient passthrough"""
        pass
    
    def backward(self):
        """Backpropagation through computation graph"""
        pass
```

**Challenges:**
- Handle gradient broadcasting correctly
- Support in-place operations carefully
- Implement matrix multiplication backward pass

**Deliverable 2.3:** `Tensor` class supporting matmul, sum, reshape, elementwise ops

---

### 2.4 Rebuild MLP with Autograd

Reimplement the MNIST MLP using your autograd engine:

```python
class Layer:
    """Base class for layers"""
    def parameters(self):
        return []

class LinearAutograd(Layer):
    def __init__(self, in_dim: int, out_dim: int):
        self.W = Tensor(np.random.randn(in_dim, out_dim) * 0.01, requires_grad=True)
        self.b = Tensor(np.zeros(out_dim), requires_grad=True)
    
    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b
    
    def parameters(self):
        return [self.W, self.b]

class ReLUAutograd(Layer):
    def __call__(self, x: Tensor) -> Tensor:
        pass
```

**Requirements:**
- Build the same 2-layer network as Part 1
- Train on MNIST and achieve similar accuracy
- Verify gradients match your manual implementation from Part 1

**Deliverable 2.4:** Training script using only your autograd engine

---

## Part 3: Modern Training Practices (Week 3)

### 3.1 Optimizers

Implement optimizers in `optimizers.py`:

```python
class Optimizer:
    """Base optimizer class"""
    def __init__(self, parameters, lr: float):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """Update parameters"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out gradients"""
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)

class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    def step(self):
        pass

class SGDMomentum(Optimizer):
    """SGD with momentum: v = beta*v + grad, param -= lr*v"""
    def __init__(self, parameters, lr: float, momentum: float = 0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.v = [np.zeros_like(p.data) for p in parameters]
    
    def step(self):
        pass

class RMSprop(Optimizer):
    """RMSprop: s = beta*s + (1-beta)*grad^2, param -= lr*grad/sqrt(s+eps)"""
    def __init__(self, parameters, lr: float, beta: float = 0.999, epsilon: float = 1e-8):
        pass
    
    def step(self):
        pass

class Adam(Optimizer):
    """Adam: Combines momentum and RMSprop with bias correction"""
    def __init__(self, parameters, lr: float, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        pass
    
    def step(self):
        pass
```

**Validation:**
- Compare optimizer behavior on simple quadratic: `f(x,y) = x^2 + y^2`
- Plot optimization trajectories
- Verify convergence rates match expected behavior

**Deliverable 3.1:** All optimizers with trajectory visualizations

---

### 3.2 Batch Normalization

Implement batch normalization from scratch:

```python
class BatchNorm1d:
    """
    Batch normalization for 1D inputs (fully connected layers).
    
    During training: normalize using batch statistics
    During inference: use running statistics
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Args:
            num_features: Number of features (C in N×C input)
            eps: Small constant for numerical stability
            momentum: Momentum for running mean/var updates
        """
        # Learnable parameters
        self.gamma = np.ones(num_features)   # Scale
        self.beta = np.zeros(num_features)   # Shift
        
        # Running statistics (not trainable)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.eps = eps
        self.momentum = momentum
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input of shape (N, C)
        Returns:
            Normalized output of shape (N, C)
        """
        pass
    
    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        """
        Compute gradients dL/dx, dL/dgamma, dL/dbeta
        
        Note: This is the tricky part! The batch mean and variance
        depend on all samples in the batch, creating dependencies.
        """
        pass
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
```

**Implementation Notes:**
- During training: compute mean/var per batch, update running statistics
- During eval: use running mean/var
- Backward pass is complex - draw the computational graph!
- Store intermediate values needed for backward pass

**Derivation Exercise:**
Derive the gradient `dL/dx` for batch normalization. The full derivation involves:
1. `dL/dx_hat` (gradient through normalized values)
2. `dL/dvar` (gradient through variance)
3. `dL/dmean` (gradient through mean)
4. Combining them to get `dL/dx`

**Deliverable 3.2:** 
- Paper derivation of BN backward pass
- Implemented BatchNorm1d with gradient checks
- Comparison: train MNIST MLP with/without BN

---

### 3.3 Learning Rate Schedulers

Implement common LR schedules:

```python
class StepLR:
    """Decay LR by gamma every step_size epochs"""
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        pass
    
    def step(self, epoch: int):
        """Update learning rate"""
        pass

class CosineAnnealingLR:
    """Cosine annealing schedule"""
    def __init__(self, optimizer, T_max: int, eta_min: float = 0):
        """
        LR varies as: eta_min + (eta_max - eta_min) * (1 + cos(pi*epoch/T_max)) / 2
        """
        pass

class WarmupLR:
    """Linear warmup followed by constant or decay"""
    def __init__(self, optimizer, warmup_epochs: int, base_scheduler=None):
        pass
```

**Deliverable 3.3:** 
- Scheduler implementations
- Plot LR vs epoch for each scheduler
- Train ResNet-18 on CIFAR-10 comparing schedulers

---

### 3.4 Final Project: ResNet-18 on CIFAR-10

Build ResNet-18 from scratch using your implementations:

**Requirements:**
1. Use PyTorch for tensor operations (leverage your understanding, not reinventing CUDA kernels)
2. Implement ResNet blocks manually (no torchvision.models)
3. Use your custom training loop with:
   - Adam optimizer
   - Batch normalization
   - Cosine annealing LR schedule
   - Data augmentation (random crops, flips)

**Architecture:**
```
Input (3×32×32)
↓ Conv(3→64, k=3, s=1, p=1) + BN + ReLU
↓ ResBlock(64→64) ×2
↓ ResBlock(64→128, downsample) ×2
↓ ResBlock(128→256, downsample) ×2
↓ ResBlock(256→512, downsample) ×2
↓ AvgPool
↓ Linear(512→10)
```

**Target Performance:**
- Test accuracy > 90% after 100 epochs
- Training time < 30 min on single GPU

**Code Structure:**
```
phase1_project/
├── models/
│   ├── resnet.py
│   └── layers.py
├── data/
│   └── cifar10_loader.py
├── training/
│   ├── trainer.py
│   └── metrics.py
├── optimizers.py
├── schedulers.py
└── train.py
```

**Deliverable 3.4:**
- Complete codebase with clean structure
- Training logs and tensorboard curves
- Final test accuracy report
- README with instructions to reproduce

---

## Submission Checklist

### Week 1: Backpropagation from Scratch
- [ ] Gradient derivations (PDF)
- [ ] `neural_net.py` with all layers implemented
- [ ] `test_gradients.py` with passing gradient checks
- [ ] `losses.py` with MSE and CrossEntropy
- [ ] `mnist_mlp.py` with >95% test accuracy

### Week 2: Autograd Engine
- [ ] Computational graph drawings (PDF)
- [ ] `autograd.py` with `Value` class
- [ ] `autograd.py` with `Tensor` class (optional but recommended)
- [ ] MNIST training using only custom autograd

### Week 3: Modern Training
- [ ] `optimizers.py` with SGD, Momentum, RMSprop, Adam
- [ ] Optimizer trajectory plots
- [ ] `batch_norm.py` with derivations and implementation
- [ ] `schedulers.py` with all schedulers
- [ ] ResNet-18 CIFAR-10 project (complete repository)

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Gradient Derivations | 15 | Correctness, clarity, mathematical notation |
| Core Implementations | 25 | Correctness, code quality, documentation |
| Gradient Checking | 15 | Comprehensive tests, passing checks |
| Autograd Engine | 20 | Functionality, graph construction, backward pass |
| Optimizers & Schedulers | 10 | Correctness, proper implementation |
| Batch Normalization | 10 | Correct forward/backward, derivation included |
| Final Project | 25 | Architecture correctness, training performance, code quality |
| **Total** | **120** | Extra credit available |

**Extra Credit Opportunities (+20 points):**
- Implement Layer Normalization and compare with Batch Norm
- Add dropout and show regularization effects
- Implement weight initialization schemes (Xavier, He, orthogonal)
- Create visualization tools for activation distributions
- Implement gradient clipping and show its effects

---

## Resources

**Essential Reading:**
- CS231n notes on backpropagation and neural networks
- "Yes you should understand backprop" by Karpathy
- PyTorch autograd internals documentation

**Debugging Tips:**
- Always start with tiny networks (2-3 parameters)
- Use gradient checking religiously
- Print shapes at each layer during first forward/backward
- Compare intermediate values with PyTorch implementation
- Visualize gradients (check for vanishing/exploding)

**Common Pitfalls:**
- Forgetting to accumulate gradients in backward pass
- Incorrect broadcasting in batch normalization
- Not caching values needed for backward pass
- Numerical instability in softmax/log
- Shape mismatches in matrix operations

---

## Getting Started

1. Create project structure:
```bash
mkdir phase1_deep_learning
cd phase1_deep_learning
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib pytest
```

2. Start with Part 1.2 - implement one layer at a time
3. Test thoroughly before moving forward
4. Don't hesitate to check PyTorch source code for reference

Good luck! This assignment will give you deep understanding of what happens under the hood of modern deep learning frameworks.