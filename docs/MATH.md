# Mathematical Foundations

This document provides detailed mathematical derivations for all components in the library.

## Table of Contents

1. [Linear Layer](#linear-layer)
2. [Activation Functions](#activation-functions)
3. [Loss Functions](#loss-functions)
4. [Backpropagation](#backpropagation)
5. [Gradient Checking](#gradient-checking)

---

## Linear Layer

### Forward Pass

The linear layer computes:

$$y = xW^T + b$$

Where:
- $x \in \mathbb{R}^{N \times d_{in}}$ — input batch
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$ — weight matrix
- $b \in \mathbb{R}^{d_{out}}$ — bias vector
- $y \in \mathbb{R}^{N \times d_{out}}$ — output batch

Element-wise:

$$y_{i,j} = \sum_{k=1}^{d_{in}} x_{i,k} W_{j,k} + b_j$$

### Backward Pass

Given upstream gradient $\frac{\partial L}{\partial y} \in \mathbb{R}^{N \times d_{out}}$:

**Gradient w.r.t. input:**

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} W$$

Derivation:
$$\frac{\partial L}{\partial x_{i,k}} = \sum_{j=1}^{d_{out}} \frac{\partial L}{\partial y_{i,j}} \frac{\partial y_{i,j}}{\partial x_{i,k}} = \sum_{j=1}^{d_{out}} \frac{\partial L}{\partial y_{i,j}} W_{j,k}$$

**Gradient w.r.t. weights:**

$$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial y}\right)^T x$$

Derivation:
$$\frac{\partial L}{\partial W_{j,k}} = \sum_{i=1}^{N} \frac{\partial L}{\partial y_{i,j}} \frac{\partial y_{i,j}}{\partial W_{j,k}} = \sum_{i=1}^{N} \frac{\partial L}{\partial y_{i,j}} x_{i,k}$$

**Gradient w.r.t. bias:**

$$\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \frac{\partial L}{\partial y_i}$$

Since $\frac{\partial y_{i,j}}{\partial b_j} = 1$, we sum over the batch dimension.

---

## Activation Functions

### ReLU (Rectified Linear Unit)

**Forward:**
$$y = \text{ReLU}(x) = \max(0, x)$$

**Backward:**
$$\frac{\partial y}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases} = \mathbf{1}_{x > 0}$$

Using chain rule:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \mathbf{1}_{x > 0}$$

**Note:** The derivative at $x = 0$ is technically undefined. We use 0 by convention.

### Sigmoid

**Forward:**
$$y = \sigma(x) = \frac{1}{1 + e^{-x}}$$

**Backward:**

First, derive $\sigma'(x)$:
$$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x) \cdot (1 - \sigma(x))$$

Therefore:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot y \cdot (1 - y)$$

**Properties:**
- Range: $(0, 1)$
- Maximum gradient: $\sigma'(0) = 0.25$
- Suffers from vanishing gradients when $|x|$ is large

### Tanh

**Forward:**
$$y = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Backward:**

$$\tanh'(x) = 1 - \tanh^2(x)$$

Therefore:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (1 - y^2)$$

**Properties:**
- Range: $(-1, 1)$
- Zero-centered output
- Maximum gradient: $\tanh'(0) = 1$
- Relation to sigmoid: $\tanh(x) = 2\sigma(2x) - 1$

---

## Loss Functions

### Mean Squared Error (MSE)

**Forward:**
$$L = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{pred},i} - y_{\text{true},i})^2$$

For vector outputs with $d$ dimensions:
$$L = \frac{1}{Nd} \sum_{i=1}^{N} \sum_{j=1}^{d} (y_{\text{pred},i,j} - y_{\text{true},i,j})^2$$

**Backward:**
$$\frac{\partial L}{\partial y_{\text{pred},i,j}} = \frac{2}{Nd} (y_{\text{pred},i,j} - y_{\text{true},i,j})$$

### Cross-Entropy Loss

Cross-entropy combines softmax normalization with negative log-likelihood.

#### Softmax Function

Converts raw logits to probabilities:
$$p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

**Properties:**
- $\sum_i p_i = 1$ (probabilities sum to 1)
- $p_i > 0$ for all $i$
- Invariant to constant shifts: $\text{softmax}(z) = \text{softmax}(z + c)$

**Numerical Stability:**

To prevent overflow, subtract the maximum:
$$p_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

#### Negative Log-Likelihood

For a single sample with target class $k$:
$$L = -\log(p_k)$$

For a batch of $N$ samples:
$$L = -\frac{1}{N} \sum_{n=1}^{N} \log(p_{n,y_n})$$

where $y_n$ is the correct class for sample $n$.

#### Combined Gradient (Softmax + Cross-Entropy)

The combined gradient has an elegant form:

$$\frac{\partial L}{\partial z_i} = \frac{1}{N}(p_i - \mathbf{1}_{i = y})$$

**Derivation:**

For class $i \neq k$ (incorrect class):
$$\frac{\partial L}{\partial z_i} = -\frac{1}{p_k} \cdot \frac{\partial p_k}{\partial z_i}$$

Since $p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$:
$$\frac{\partial p_k}{\partial z_i} = -\frac{e^{z_k} e^{z_i}}{(\sum_j e^{z_j})^2} = -p_k p_i$$

Therefore:
$$\frac{\partial L}{\partial z_i} = -\frac{1}{p_k} \cdot (-p_k p_i) = p_i$$

For class $i = k$ (correct class):
$$\frac{\partial p_k}{\partial z_k} = \frac{e^{z_k} \sum_j e^{z_j} - e^{z_k} e^{z_k}}{(\sum_j e^{z_j})^2} = p_k(1 - p_k)$$

Therefore:
$$\frac{\partial L}{\partial z_k} = -\frac{1}{p_k} \cdot p_k(1 - p_k) = -(1 - p_k) = p_k - 1$$

**Combined:**
$$\frac{\partial L}{\partial z_i} = p_i - \mathbf{1}_{i = k}$$

This beautiful result is why softmax + cross-entropy is so commonly used.

---

## Backpropagation

### Chain Rule

For composed functions $f = f_n \circ f_{n-1} \circ \cdots \circ f_1$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdot \cdots \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}$$

### Computational Graph Perspective

Each node computes:
1. **Forward:** Output from inputs
2. **Backward:** Local gradient × upstream gradient

```
x → [Layer 1] → h₁ → [Layer 2] → h₂ → [Loss] → L
       ↑               ↑                ↑
    dL/dh₁         dL/dh₂           dL/dL=1
```

Backward pass:
1. $\frac{\partial L}{\partial L} = 1$
2. $\frac{\partial L}{\partial h_2} = \text{loss.backward()}$
3. $\frac{\partial L}{\partial h_1} = \text{layer2.backward}(\frac{\partial L}{\partial h_2})$
4. $\frac{\partial L}{\partial x} = \text{layer1.backward}(\frac{\partial L}{\partial h_1})$

### Example: 2-Layer MLP

**Architecture:**
$$\hat{y} = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

**Forward pass:**
1. $z_1 = x W_1^T + b_1$
2. $h_1 = \text{ReLU}(z_1)$
3. $z_2 = h_1 W_2^T + b_2$ (logits)
4. $L = \text{CrossEntropy}(z_2, y)$

**Backward pass:**
1. $\frac{\partial L}{\partial z_2} = \text{softmax}(z_2) - \text{one\_hot}(y)$
2. $\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial z_2} W_2$
3. $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h_1} \odot \mathbf{1}_{z_1 > 0}$
4. $\frac{\partial L}{\partial W_1} = (\frac{\partial L}{\partial z_1})^T x$
5. $\frac{\partial L}{\partial W_2} = (\frac{\partial L}{\partial z_2})^T h_1$

---

## Gradient Checking

### Numerical Gradient

Using centered finite differences:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$

where $e_i$ is the $i$-th standard basis vector.

### Error Analysis

**Taylor expansion:**
$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{\epsilon^2}{2}f''(x) + \frac{\epsilon^3}{6}f'''(x) + O(\epsilon^4)$$
$$f(x - \epsilon) = f(x) - \epsilon f'(x) + \frac{\epsilon^2}{2}f''(x) - \frac{\epsilon^3}{6}f'''(x) + O(\epsilon^4)$$

**Centered difference:**
$$\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} = f'(x) + \frac{\epsilon^2}{6}f'''(x) + O(\epsilon^4)$$

The error is $O(\epsilon^2)$, giving ~$10^{-10}$ error for $\epsilon = 10^{-5}$.

**One-sided difference (comparison):**
$$\frac{f(x + \epsilon) - f(x)}{\epsilon} = f'(x) + \frac{\epsilon}{2}f''(x) + O(\epsilon^2)$$

Error is $O(\epsilon)$, giving ~$10^{-5}$ error for $\epsilon = 10^{-5}$.

### Relative Error

Compare using relative error to handle different scales:

$$\text{rel\_error} = \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(|g_{\text{analytical}}| + |g_{\text{numerical}}|, \delta)}$$

where $\delta \approx 10^{-7}$ prevents division by zero.

**Interpretation:**
- $< 10^{-7}$: Excellent
- $10^{-7}$ to $10^{-5}$: Acceptable
- $10^{-5}$ to $10^{-3}$: Suspicious
- $> 10^{-3}$: Bug likely

---

## Weight Initialization

### The Problem

Poor initialization can cause:
- **Vanishing gradients:** Activations become very small
- **Exploding gradients:** Activations become very large

### Variance Analysis

For a linear layer $y = Wx$:

$$\text{Var}(y_j) = \sum_{i=1}^{n_{\text{in}}} \text{Var}(W_{ji}) \cdot \text{Var}(x_i)$$

Assuming $E[x_i] = 0$ and $E[W_{ji}] = 0$.

If $\text{Var}(W_{ji}) = \sigma^2$ and $\text{Var}(x_i) = 1$:
$$\text{Var}(y_j) = n_{\text{in}} \cdot \sigma^2$$

### Xavier/Glorot Initialization

For layers followed by tanh or sigmoid:

$$W \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}}}\right) \quad \text{or} \quad W \sim \mathcal{U}\left(-\sqrt{\frac{3}{n_{\text{in}}}}, \sqrt{\frac{3}{n_{\text{in}}}}\right)$$

Maintains variance: $\text{Var}(y) \approx \text{Var}(x)$

### Kaiming He Initialization

For layers followed by ReLU:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

The factor of 2 accounts for ReLU zeroing out half the activations (on average).

This is what we use in our implementation:
```python
self._w = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
```

---

## Summary of Key Formulas

| Component | Forward | Backward (local gradient) |
|-----------|---------|---------------------------|
| Linear | $y = xW^T + b$ | $\frac{\partial y}{\partial x} = W$, $\frac{\partial y}{\partial W} = x^T$ |
| ReLU | $y = \max(0, x)$ | $\frac{\partial y}{\partial x} = \mathbf{1}_{x>0}$ |
| Sigmoid | $y = \sigma(x)$ | $\frac{\partial y}{\partial x} = y(1-y)$ |
| Tanh | $y = \tanh(x)$ | $\frac{\partial y}{\partial x} = 1-y^2$ |
| MSE | $L = \frac{1}{N}\|y-\hat{y}\|^2$ | $\frac{\partial L}{\partial \hat{y}} = \frac{2}{N}(\hat{y}-y)$ |
| CrossEntropy | $L = -\log(p_k)$ | $\frac{\partial L}{\partial z} = p - \text{one\_hot}(k)$ |
