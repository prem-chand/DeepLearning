# Value Class Quick Reference

## Import
```python
from deeplearning.autograd import Value
```

## Creating Values
```python
x = Value(2.0)      # Create a Value
y = Value(3.0)
```

## Arithmetic Operations

| Operation | Syntax | Example | Gradient Formula |
|-----------|--------|---------|------------------|
| Addition | `a + b` | `c = x + y` | `∂c/∂a = 1, ∂c/∂b = 1` |
| Subtraction | `a - b` | `c = x - y` | `∂c/∂a = 1, ∂c/∂b = -1` |
| Multiplication | `a * b` | `c = x * y` | `∂c/∂a = b, ∂c/∂b = a` |
| Division | `a / b` | `c = x / y` | `∂c/∂a = 1/b, ∂c/∂b = -a/b²` |
| Power | `a ** n` | `c = x ** 2` | `∂c/∂a = n·a^(n-1)` |
| Negation | `-a` | `c = -x` | `∂c/∂a = -1` |

## Activation Functions

| Function | Method | Formula | Gradient Formula |
|----------|--------|---------|------------------|
| ReLU | `x.relu()` | `max(0, x)` | `1 if x > 0 else 0` |
| Tanh | `x.tanh()` | `(e^x - e^-x)/(e^x + e^-x)` | `1 - tanh(x)²` |
| Sigmoid | `x.sigmoid()` | `1/(1 + e^-x)` | `σ(x)·(1 - σ(x))` |

## Mathematical Functions

| Function | Method | Gradient Formula |
|----------|--------|------------------|
| Exponential | `x.exp()` | `e^x` |
| Natural Log | `x.log()` | `1/x` |

## Gradient Operations

```python
# Compute gradients
z.backward()        # Compute ∂z/∂x for all x in the graph

# Access gradients
print(x.grad)       # Get gradient ∂z/∂x

# Reset gradients
x.zero_grad()       # Set gradient to 0
```

## Common Patterns

### Simple Expression
```python
x = Value(2.0)
y = Value(3.0)
z = x * y + x ** 2
z.backward()
print(f"∂z/∂x = {x.grad}")  # 7.0
print(f"∂z/∂y = {y.grad}")  # 2.0
```

### Neuron
```python
# y = activation(w·x + b)
x = Value(0.5)
w = Value(2.0)
b = Value(1.0)
y = (x * w + b).tanh()
y.backward()
print(f"∂y/∂w = {w.grad}")
```

### Multi-layer Expression
```python
x = Value(3.0)
y = Value(2.0)
z = (x + y) * (x - y)  # Expands to x² - y²
z.backward()
print(f"∂z/∂x = {x.grad}")  # 2x = 6.0
print(f"∂z/∂y = {y.grad}")  # -2y = -4.0
```

### Gradient Accumulation
```python
x = Value(2.0)

# First backward pass
y1 = x * 3
y1.backward()
print(x.grad)  # 3.0

# Gradients accumulate!
y2 = x * 5
y2.backward()
print(x.grad)  # 8.0 (3.0 + 5.0)

# Reset for new computation
x.zero_grad()
print(x.grad)  # 0.0
```

## Properties

| Property | Description |
|----------|-------------|
| `value.data` | The scalar value (float) |
| `value.grad` | The gradient (float, default 0.0) |
| `value._op` | Operation that created this value (str) |
| `value._prev` | Parent nodes in computation graph (set) |

## String Representation

```python
x = Value(2.5)
x.grad = 1.5

print(repr(x))  # Value(data=2.5000, grad=1.5000)
print(str(x))   # Value(2.5000)

y = x + 3
print(str(y))   # Value(5.5000) from +
```

## Error Handling

```python
# Log of non-positive value raises ValueError
x = Value(-1.0)
try:
    y = x.log()
except ValueError as e:
    print(e)  # "log undefined for non-positive values (got -1.0)"

# Power only supports numeric exponents
x = Value(2.0)
y = Value(3.0)
# x ** y  # AssertionError: Only supports int/float powers
```

## Tips

1. **Always call backward() before accessing gradients**
   ```python
   x = Value(2.0)
   y = x * 3
   # print(x.grad)  # Still 0.0 - backward not called yet!
   y.backward()
   print(x.grad)    # Now 3.0
   ```

2. **Gradients accumulate by default**
   ```python
   # Call zero_grad() between independent computations
   x.zero_grad()
   ```

3. **Values used multiple times contribute multiple gradients**
   ```python
   x = Value(2.0)
   y = x * x + x * 3  # x used 3 times
   y.backward()
   print(x.grad)  # 2x + 3 = 7.0
   ```

4. **Mix Values and Python scalars freely**
   ```python
   x = Value(2.0)
   y = x * 3 + 1.5  # Scalars auto-converted to Values
   y = 3 * x + 1.5  # Works both ways
   ```

## Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/test_autograd.py -v
```

Run the demonstration:
```bash
python scripts/demo_autograd.py
```
