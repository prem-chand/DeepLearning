"""
Demonstration of Enhanced Value Class
======================================

This script demonstrates the enhanced capabilities of the Value class,
including new operations, activations, and gradient computation.
"""

from deeplearning.autograd import Value
import numpy as np


def demo_basic_operations():
    """Demonstrate basic arithmetic operations."""
    print("=" * 60)
    print("BASIC ARITHMETIC OPERATIONS")
    print("=" * 60)
    
    # Addition
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    print(f"\nAddition: {a.data} + {b.data} = {c.data}")
    c.backward()
    print(f"  Gradients: da={a.grad}, db={b.grad}")
    
    # Subtraction
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    print(f"\nSubtraction: {a.data} - {b.data} = {c.data}")
    c.backward()
    print(f"  Gradients: da={a.grad}, db={b.grad}")
    
    # Multiplication
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    print(f"\nMultiplication: {a.data} * {b.data} = {c.data}")
    c.backward()
    print(f"  Gradients: da={a.grad}, db={b.grad}")
    
    # Division
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    print(f"\nDivision: {a.data} / {b.data} = {c.data}")
    c.backward()
    print(f"  Gradients: da={a.grad:.4f}, db={b.grad:.4f}")
    
    # Power
    a = Value(2.0)
    c = a ** 3
    print(f"\nPower: {a.data}^3 = {c.data}")
    c.backward()
    print(f"  Gradient: da={a.grad}")
    
    # Negation
    a = Value(5.0)
    c = -a
    print(f"\nNegation: -{a.data} = {c.data}")
    c.backward()
    print(f"  Gradient: da={a.grad}")


def demo_activations():
    """Demonstrate activation functions."""
    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTIONS")
    print("=" * 60)
    
    # ReLU
    x = Value(2.0)
    y = x.relu()
    print(f"\nReLU({x.data}) = {y.data}")
    y.backward()
    print(f"  Gradient: dx={x.grad}")
    
    x = Value(-2.0)
    y = x.relu()
    print(f"ReLU({x.data}) = {y.data}")
    y.backward()
    print(f"  Gradient: dx={x.grad}")
    
    # Tanh
    x = Value(0.5)
    y = x.tanh()
    print(f"\ntanh({x.data}) = {y.data:.4f}")
    y.backward()
    print(f"  Gradient: dx={x.grad:.4f}")
    print(f"  (NumPy verification: {np.tanh(0.5):.4f})")
    
    # Sigmoid
    x = Value(0.0)
    y = x.sigmoid()
    print(f"\nsigmoid({x.data}) = {y.data:.4f}")
    y.backward()
    print(f"  Gradient: dx={x.grad:.4f}")


def demo_math_functions():
    """Demonstrate mathematical functions."""
    print("\n" + "=" * 60)
    print("MATHEMATICAL FUNCTIONS")
    print("=" * 60)
    
    # Exponential
    x = Value(1.0)
    y = x.exp()
    print(f"\nexp({x.data}) = {y.data:.4f}")
    y.backward()
    print(f"  Gradient: dx={y.data:.4f}")
    print(f"  (NumPy verification: {np.exp(1.0):.4f})")
    
    # Logarithm
    x = Value(2.718)
    y = x.log()
    print(f"\nlog({x.data}) = {y.data:.4f}")
    y.backward()
    print(f"  Gradient: dx={x.grad:.4f}")


def demo_complex_graph():
    """Demonstrate complex computational graph."""
    print("\n" + "=" * 60)
    print("COMPLEX COMPUTATIONAL GRAPH")
    print("=" * 60)
    
    # Example: z = (x^2 + y) * tanh(x - y)
    x = Value(2.0)
    y = Value(1.0)
    
    z = (x ** 2 + y) * (x - y).tanh()
    
    print(f"\nExpression: z = (x^2 + y) * tanh(x - y)")
    print(f"  x = {x.data}, y = {y.data}")
    print(f"  z = {z.data:.4f}")
    
    z.backward()
    print(f"\nGradients:")
    print(f"  dz/dx = {x.grad:.4f}")
    print(f"  dz/dy = {y.grad:.4f}")


def demo_neuron():
    """Demonstrate a simple neuron."""
    print("\n" + "=" * 60)
    print("SIMPLE NEURON: y = tanh(w*x + b)")
    print("=" * 60)
    
    # Input
    x = Value(0.5)
    
    # Parameters
    w = Value(2.0)
    b = Value(1.0)
    
    # Forward pass
    y = (x * w + b).tanh()
    
    print(f"\nInput: x = {x.data}")
    print(f"Weight: w = {w.data}")
    print(f"Bias: b = {b.data}")
    print(f"Output: y = {y.data:.4f}")
    
    # Backward pass
    y.backward()
    
    print(f"\nGradients (for backpropagation):")
    print(f"  dy/dx = {x.grad:.4f}")
    print(f"  dy/dw = {w.grad:.4f}")
    print(f"  dy/db = {b.grad:.4f}")


def demo_gradient_accumulation():
    """Demonstrate gradient accumulation and zero_grad."""
    print("\n" + "=" * 60)
    print("GRADIENT ACCUMULATION")
    print("=" * 60)
    
    x = Value(2.0)
    
    # First computation
    y1 = x * 3
    y1.backward()
    print(f"\nAfter first backward (y1 = x * 3):")
    print(f"  x.grad = {x.grad}")
    
    # Second computation (gradients accumulate)
    y2 = x * 5
    y2.backward()
    print(f"\nAfter second backward (y2 = x * 5):")
    print(f"  x.grad = {x.grad} (accumulated)")
    
    # Reset gradients
    x.zero_grad()
    print(f"\nAfter zero_grad():")
    print(f"  x.grad = {x.grad}")


def demo_multiple_uses():
    """Demonstrate a value used multiple times in the same graph."""
    print("\n" + "=" * 60)
    print("MULTIPLE USES IN GRAPH")
    print("=" * 60)
    
    x = Value(2.0)
    # x is used multiple times
    y = x * x + x * 3
    
    print(f"\nExpression: y = x*x + x*3")
    print(f"  x = {x.data}")
    print(f"  y = {y.data}")
    
    y.backward()
    print(f"\nGradient:")
    print(f"  dy/dx = {x.grad} (should be 2*x + 3 = 7)")


def demo_chain_rule():
    """Demonstrate chain rule in action."""
    print("\n" + "=" * 60)
    print("CHAIN RULE DEMONSTRATION")
    print("=" * 60)
    
    # f(g(h(x))) where:
    # h(x) = x^2
    # g(x) = x + 1
    # f(x) = tanh(x)
    
    x = Value(2.0)
    h = x ** 2          # h = 4
    g = h + 1           # g = 5
    f = g.tanh()        # f = tanh(5)
    
    print(f"\nComposite function: f(g(h(x))) = tanh((x^2) + 1)")
    print(f"  x = {x.data}")
    print(f"  h(x) = x^2 = {h.data}")
    print(f"  g(h) = h + 1 = {g.data}")
    print(f"  f(g) = tanh(g) = {f.data:.4f}")
    
    f.backward()
    
    # Manual calculation:
    # df/dx = df/dg * dg/dh * dh/dx
    # df/dg = 1 - tanh(g)^2
    # dg/dh = 1
    # dh/dx = 2*x
    tanh_g = np.tanh(5.0)
    expected_grad = (1 - tanh_g**2) * 1 * (2 * 2.0)
    
    print(f"\nGradient:")
    print(f"  df/dx = {x.grad:.6f}")
    print(f"  Expected (manual): {expected_grad:.6f}")
    print(f"  Match: {abs(x.grad - expected_grad) < 1e-6}")


if __name__ == "__main__":
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("ENHANCED VALUE CLASS DEMONSTRATION")
    print("=" * 60)
    
    demo_basic_operations()
    demo_activations()
    demo_math_functions()
    demo_complex_graph()
    demo_neuron()
    demo_gradient_accumulation()
    demo_multiple_uses()
    demo_chain_rule()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThe Value class now supports:")
    print("  ✓ All basic arithmetic operations (+, -, *, /, **)")
    print("  ✓ Activation functions (ReLU, tanh, sigmoid)")
    print("  ✓ Mathematical functions (exp, log)")
    print("  ✓ Automatic gradient computation")
    print("  ✓ Complex computational graphs")
    print("  ✓ Gradient accumulation and reset")
    print("  ✓ Comprehensive error handling")
    print()
