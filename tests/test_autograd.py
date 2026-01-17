"""
Test Suite for Autograd Value Class
====================================

This module provides comprehensive tests for the Value class in the autograd module.
It tests:
1. Basic arithmetic operations (+, -, *, /, **)
2. Activation functions (ReLU, tanh, sigmoid)
3. Mathematical functions (exp, log)
4. Gradient computation correctness
5. Complex computational graphs
6. Edge cases and error handling

The tests use numerical gradient checking to verify analytical gradients.
"""

import pytest
import numpy as np
from deeplearning.autograd import Value


class TestBasicOperations:
    """Test basic arithmetic operations and their gradients."""

    def test_addition(self):
        """Test addition operation and gradients."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b

        assert c.data == 5.0, "Addition forward pass failed"

        c.backward()
        assert a.grad == 1.0, "Addition gradient w.r.t. first operand incorrect"
        assert b.grad == 1.0, "Addition gradient w.r.t. second operand incorrect"

    def test_addition_with_scalar(self):
        """Test addition with Python scalar."""
        a = Value(2.0)
        c = a + 3.0

        assert c.data == 5.0, "Addition with scalar failed"

        c.backward()
        assert a.grad == 1.0, "Addition with scalar gradient incorrect"

    def test_right_addition(self):
        """Test right addition (scalar + Value)."""
        a = Value(2.0)
        c = 3.0 + a

        assert c.data == 5.0, "Right addition failed"

        c.backward()
        assert a.grad == 1.0, "Right addition gradient incorrect"

    def test_multiplication(self):
        """Test multiplication operation and gradients."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b

        assert c.data == 6.0, "Multiplication forward pass failed"

        c.backward()
        assert a.grad == 3.0, "Multiplication gradient w.r.t. first operand incorrect"
        assert b.grad == 2.0, "Multiplication gradient w.r.t. second operand incorrect"

    def test_multiplication_with_scalar(self):
        """Test multiplication with Python scalar."""
        a = Value(2.0)
        c = a * 3.0

        assert c.data == 6.0, "Multiplication with scalar failed"

        c.backward()
        assert a.grad == 3.0, "Multiplication with scalar gradient incorrect"

    def test_power(self):
        """Test power operation and gradients."""
        a = Value(2.0)
        c = a ** 3

        assert c.data == 8.0, "Power forward pass failed"

        c.backward()
        # d(x^3)/dx = 3*x^2 = 3*4 = 12
        assert abs(a.grad - 12.0) < 1e-6, "Power gradient incorrect"

    def test_negation(self):
        """Test negation operation."""
        a = Value(2.0)
        c = -a

        assert c.data == -2.0, "Negation forward pass failed"

        c.backward()
        assert a.grad == -1.0, "Negation gradient incorrect"

    def test_subtraction(self):
        """Test subtraction operation and gradients."""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b

        assert c.data == 2.0, "Subtraction forward pass failed"

        c.backward()
        assert a.grad == 1.0, "Subtraction gradient w.r.t. first operand incorrect"
        assert b.grad == -1.0, "Subtraction gradient w.r.t. second operand incorrect"

    def test_right_subtraction(self):
        """Test right subtraction (scalar - Value)."""
        a = Value(3.0)
        c = 5.0 - a

        assert c.data == 2.0, "Right subtraction failed"

        c.backward()
        assert a.grad == -1.0, "Right subtraction gradient incorrect"

    def test_division(self):
        """Test division operation and gradients."""
        a = Value(6.0)
        b = Value(2.0)
        c = a / b

        assert c.data == 3.0, "Division forward pass failed"

        c.backward()
        # d(a/b)/da = 1/b = 0.5
        assert abs(a.grad - 0.5) < 1e-6, "Division gradient w.r.t. numerator incorrect"
        # d(a/b)/db = -a/b^2 = -6/4 = -1.5
        assert abs(b.grad - (-1.5)) < 1e-6, "Division gradient w.r.t. denominator incorrect"

    def test_right_division(self):
        """Test right division (scalar / Value)."""
        a = Value(2.0)
        c = 6.0 / a

        assert c.data == 3.0, "Right division failed"

        c.backward()
        # d(6/a)/da = -6/a^2 = -6/4 = -1.5
        assert abs(a.grad - (-1.5)) < 1e-6, "Right division gradient incorrect"


class TestActivationFunctions:
    """Test activation functions and their gradients."""

    def test_relu_positive(self):
        """Test ReLU with positive input."""
        a = Value(2.0)
        c = a.relu()

        assert c.data == 2.0, "ReLU forward pass failed for positive input"

        c.backward()
        assert a.grad == 1.0, "ReLU gradient incorrect for positive input"

    def test_relu_negative(self):
        """Test ReLU with negative input."""
        a = Value(-2.0)
        c = a.relu()

        assert c.data == 0.0, "ReLU forward pass failed for negative input"

        c.backward()
        assert a.grad == 0.0, "ReLU gradient incorrect for negative input"

    def test_relu_zero(self):
        """Test ReLU at zero."""
        a = Value(0.0)
        c = a.relu()

        assert c.data == 0.0, "ReLU forward pass failed at zero"

        c.backward()
        assert a.grad == 0.0, "ReLU gradient incorrect at zero"

    def test_tanh(self):
        """Test tanh activation and gradient."""
        a = Value(0.5)
        c = a.tanh()

        expected = np.tanh(0.5)
        assert abs(c.data - expected) < 1e-6, "tanh forward pass failed"

        c.backward()
        # d(tanh(x))/dx = 1 - tanh(x)^2
        expected_grad = 1 - np.tanh(0.5) ** 2
        assert abs(a.grad - expected_grad) < 1e-6, "tanh gradient incorrect"

    def test_sigmoid(self):
        """Test sigmoid activation and gradient."""
        a = Value(0.5)
        c = a.sigmoid()

        expected = 1 / (1 + np.exp(-0.5))
        assert abs(c.data - expected) < 1e-6, "sigmoid forward pass failed"

        c.backward()
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        s = 1 / (1 + np.exp(-0.5))
        expected_grad = s * (1 - s)
        assert abs(a.grad - expected_grad) < 1e-6, "sigmoid gradient incorrect"


class TestMathematicalFunctions:
    """Test mathematical functions and their gradients."""

    def test_exp(self):
        """Test exponential function and gradient."""
        a = Value(2.0)
        c = a.exp()

        expected = np.exp(2.0)
        assert abs(c.data - expected) < 1e-6, "exp forward pass failed"

        c.backward()
        # d(e^x)/dx = e^x
        assert abs(a.grad - expected) < 1e-6, "exp gradient incorrect"

    def test_log(self):
        """Test natural logarithm and gradient."""
        a = Value(2.0)
        c = a.log()

        expected = np.log(2.0)
        assert abs(c.data - expected) < 1e-6, "log forward pass failed"

        c.backward()
        # d(log(x))/dx = 1/x
        assert abs(a.grad - 0.5) < 1e-6, "log gradient incorrect"

    def test_log_error_on_negative(self):
        """Test that log raises error for negative values."""
        a = Value(-1.0)
        with pytest.raises(ValueError, match="log undefined for non-positive values"):
            a.log()

    def test_log_error_on_zero(self):
        """Test that log raises error for zero."""
        a = Value(0.0)
        with pytest.raises(ValueError, match="log undefined for non-positive values"):
            a.log()


class TestComplexGraphs:
    """Test complex computational graphs."""

    def test_simple_expression(self):
        """Test: z = x * y + x^2"""
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x ** 2

        assert z.data == 10.0, "Complex expression forward pass failed"

        z.backward()
        # dz/dx = y + 2*x = 3 + 4 = 7
        assert abs(x.grad - 7.0) < 1e-6, "Complex expression gradient w.r.t. x incorrect"
        # dz/dy = x = 2
        assert abs(y.grad - 2.0) < 1e-6, "Complex expression gradient w.r.t. y incorrect"

    def test_neuron(self):
        """Test a simple neuron: y = tanh(w*x + b)"""
        x = Value(0.5)
        w = Value(2.0)
        b = Value(1.0)
        y = (x * w + b).tanh()

        # Forward pass
        expected = np.tanh(0.5 * 2.0 + 1.0)
        assert abs(y.data - expected) < 1e-6, "Neuron forward pass failed"

        # Backward pass
        y.backward()

        # Numerical gradient check for w
        epsilon = 1e-5
        w_plus = (0.5 * (2.0 + epsilon) + 1.0)
        w_minus = (0.5 * (2.0 - epsilon) + 1.0)
        numerical_grad = (np.tanh(w_plus) - np.tanh(w_minus)) / (2 * epsilon)

        assert abs(w.grad - numerical_grad) < 1e-4, "Neuron gradient w.r.t. weight incorrect"

    def test_multi_layer_expression(self):
        """Test: z = (x + y) * (x - y)"""
        x = Value(3.0)
        y = Value(2.0)
        z = (x + y) * (x - y)

        # z = 5 * 1 = 5
        assert z.data == 5.0, "Multi-layer expression forward pass failed"

        z.backward()
        # dz/dx = (x - y) + (x + y) = 2*x = 6
        assert abs(x.grad - 6.0) < 1e-6, "Multi-layer gradient w.r.t. x incorrect"
        # dz/dy = (x + y) - (x - y) = 2*y = 4, but with correct sign: -4
        # Actually: dz/dy = (x + y) * (-1) + (x - y) * 1 = -(x+y) + (x-y) = -2y = -4
        # Let me recalculate: z = x^2 - y^2, so dz/dy = -2y = -4
        assert abs(y.grad - (-4.0)) < 1e-6, "Multi-layer gradient w.r.t. y incorrect"

    def test_division_in_graph(self):
        """Test division in computational graph."""
        x = Value(4.0)
        y = Value(2.0)
        z = (x + y) / (x - y)

        # z = 6 / 2 = 3
        assert z.data == 3.0, "Division in graph forward pass failed"

        z.backward()

        # Numerical gradient check
        epsilon = 1e-5
        
        def f_x(val):
            return (val + 2.0) / (val - 2.0)
        
        numerical_grad_x = (f_x(4.0 + epsilon) - f_x(4.0 - epsilon)) / (2 * epsilon)
        assert abs(x.grad - numerical_grad_x) < 1e-4, "Division gradient w.r.t. x incorrect"

    def test_power_in_graph(self):
        """Test power operation in computational graph."""
        x = Value(2.0)
        y = Value(3.0)
        z = (x + y) ** 2

        # z = 5^2 = 25
        assert z.data == 25.0, "Power in graph forward pass failed"

        z.backward()
        # dz/dx = 2 * (x + y) = 2 * 5 = 10
        assert abs(x.grad - 10.0) < 1e-6, "Power in graph gradient w.r.t. x incorrect"
        assert abs(y.grad - 10.0) < 1e-6, "Power in graph gradient w.r.t. y incorrect"


class TestGradientAccumulation:
    """Test gradient accumulation and zero_grad."""

    def test_gradient_accumulation(self):
        """Test that gradients accumulate across multiple backward passes."""
        x = Value(2.0)
        
        # First computation
        y1 = x * 2
        y1.backward()
        first_grad = x.grad
        
        # Second computation (gradients should accumulate)
        y2 = x * 3
        y2.backward()
        
        assert x.grad == first_grad + 3.0, "Gradients should accumulate"

    def test_zero_grad(self):
        """Test zero_grad method."""
        x = Value(2.0)
        y = x * 3
        y.backward()
        
        assert x.grad != 0.0, "Gradient should be non-zero after backward"
        
        x.zero_grad()
        assert x.grad == 0.0, "Gradient should be zero after zero_grad"

    def test_multiple_uses_in_graph(self):
        """Test a value used multiple times in the same graph."""
        x = Value(2.0)
        y = x * x + x * x  # x is used 4 times total

        assert y.data == 8.0, "Multiple uses forward pass failed"

        y.backward()
        # dy/dx = 2*x + 2*x = 4*x = 8
        assert abs(x.grad - 8.0) < 1e-6, "Multiple uses gradient incorrect"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_multiplication(self):
        """Test multiplication by zero."""
        x = Value(5.0)
        y = Value(0.0)
        z = x * y

        assert z.data == 0.0, "Zero multiplication failed"

        z.backward()
        assert x.grad == 0.0, "Zero multiplication gradient w.r.t. x incorrect"
        assert y.grad == 5.0, "Zero multiplication gradient w.r.t. y incorrect"

    def test_power_zero(self):
        """Test raising to power of zero."""
        x = Value(5.0)
        y = x ** 0

        assert y.data == 1.0, "Power of zero failed"

        y.backward()
        assert x.grad == 0.0, "Power of zero gradient incorrect"

    def test_power_one(self):
        """Test raising to power of one."""
        x = Value(5.0)
        y = x ** 1

        assert y.data == 5.0, "Power of one failed"

        y.backward()
        assert x.grad == 1.0, "Power of one gradient incorrect"

    def test_negative_power(self):
        """Test negative power (reciprocal)."""
        x = Value(2.0)
        y = x ** -1

        assert abs(y.data - 0.5) < 1e-6, "Negative power failed"

        y.backward()
        # d(x^-1)/dx = -1 * x^-2 = -1/4 = -0.25
        assert abs(x.grad - (-0.25)) < 1e-6, "Negative power gradient incorrect"

    def test_fractional_power(self):
        """Test fractional power (square root)."""
        x = Value(4.0)
        y = x ** 0.5

        assert abs(y.data - 2.0) < 1e-6, "Fractional power failed"

        y.backward()
        # d(x^0.5)/dx = 0.5 * x^-0.5 = 0.5 / 2 = 0.25
        assert abs(x.grad - 0.25) < 1e-6, "Fractional power gradient incorrect"

    def test_large_values(self):
        """Test with large values."""
        x = Value(1000.0)
        y = Value(2000.0)
        z = x + y

        assert z.data == 3000.0, "Large values addition failed"

        z.backward()
        assert x.grad == 1.0, "Large values gradient incorrect"

    def test_small_values(self):
        """Test with small values."""
        x = Value(1e-6)
        y = Value(2e-6)
        z = x + y

        assert abs(z.data - 3e-6) < 1e-12, "Small values addition failed"

        z.backward()
        assert abs(x.grad - 1.0) < 1e-6, "Small values gradient incorrect"


class TestStringRepresentation:
    """Test string representation methods."""

    def test_repr(self):
        """Test __repr__ method."""
        x = Value(2.5)
        x.grad = 1.5
        
        repr_str = repr(x)
        assert "2.5000" in repr_str, "__repr__ should show data"
        assert "1.5000" in repr_str, "__repr__ should show grad"

    def test_str_with_op(self):
        """Test __str__ method with operation."""
        x = Value(2.0)
        y = x + 3.0
        
        str_y = str(y)
        assert "5.0000" in str_y, "__str__ should show data"
        assert "+" in str_y, "__str__ should show operation"

    def test_str_without_op(self):
        """Test __str__ method without operation."""
        x = Value(2.0)
        
        str_x = str(x)
        assert "2.0000" in str_x, "__str__ should show data"


class TestNumericalGradientChecking:
    """Test gradients using numerical approximation."""

    def numerical_gradient(self, f, x, epsilon=1e-5):
        """Compute numerical gradient using centered finite differences."""
        return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

    def test_addition_numerical(self):
        """Numerical gradient check for addition."""
        x_val = 2.0
        y_val = 3.0
        
        x = Value(x_val)
        y = Value(y_val)
        z = x + y
        z.backward()
        
        # Numerical gradient w.r.t. x
        def f(x_input):
            return x_input + y_val
        
        numerical_grad = self.numerical_gradient(f, x_val)
        assert abs(x.grad - numerical_grad) < 1e-6, "Numerical gradient check failed for addition"

    def test_multiplication_numerical(self):
        """Numerical gradient check for multiplication."""
        x_val = 2.0
        y_val = 3.0
        
        x = Value(x_val)
        y = Value(y_val)
        z = x * y
        z.backward()
        
        # Numerical gradient w.r.t. x
        def f(x_input):
            return x_input * y_val
        
        numerical_grad = self.numerical_gradient(f, x_val)
        assert abs(x.grad - numerical_grad) < 1e-6, "Numerical gradient check failed for multiplication"

    def test_complex_expression_numerical(self):
        """Numerical gradient check for complex expression."""
        x_val = 2.0
        
        x = Value(x_val)
        z = (x ** 2 + x * 3).tanh()
        z.backward()
        
        # Numerical gradient w.r.t. x
        def f(x_input):
            return np.tanh(x_input ** 2 + x_input * 3)
        
        numerical_grad = self.numerical_gradient(f, x_val)
        assert abs(x.grad - numerical_grad) < 1e-4, "Numerical gradient check failed for complex expression"


if __name__ == "__main__":
    """Run all tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
