"""
Unit tests for neural network layers.

This module contains pytest-compatible tests for validating
the forward and backward passes of all neural network layers.
"""

import numpy as np
import pytest

from deeplearning.neural_net import Linear, ReLU, Sigmoid, Tanh


def numerical_gradient(f, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient using centered finite differences."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        original_value = x[idx]

        x[idx] = original_value + epsilon
        f_plus = f(x)

        x[idx] = original_value - epsilon
        f_minus = f(x)

        grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        x[idx] = original_value
        it.iternext()

    return grad


def gradient_check(layer, x: np.ndarray, tolerance: float = 1e-5) -> bool:
    """Verify backward pass against numerical gradient."""

    def loss_fn(x_input):
        return layer.forward(x_input).sum()

    output = layer.forward(x)
    grad = layer.backward(np.ones_like(output))
    num_grad = numerical_gradient(loss_fn, x)

    rel_error = np.max(
        np.abs(grad - num_grad)
        / (np.maximum(1e-7, np.abs(grad) + np.abs(num_grad)))
    )

    return rel_error < tolerance


class TestLinear:
    """Tests for Linear layer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        layer = Linear(in_features=10, out_features=5)
        x = np.random.randn(32, 10)
        y = layer.forward(x)
        assert y.shape == (32, 5)

    def test_forward_computation(self):
        """Test forward pass computation is correct."""
        layer = Linear(in_features=3, out_features=2)
        layer._w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        layer._b = np.array([0.1, 0.2])
        x = np.array([[1.0, 1.0, 1.0]])
        y = layer.forward(x)
        expected = np.array([[6.1, 15.2]])  # [1+2+3+0.1, 4+5+6+0.2]
        np.testing.assert_allclose(y, expected)

    def test_backward_gradient_check(self):
        """Test backward pass using numerical gradient checking."""
        np.random.seed(42)
        layer = Linear(in_features=5, out_features=3)
        x = np.random.randn(4, 5)
        assert gradient_check(layer, x)

    def test_backward_stores_gradients(self):
        """Test that backward pass stores weight and bias gradients."""
        layer = Linear(in_features=3, out_features=2)
        x = np.random.randn(4, 3)
        y = layer.forward(x)
        layer.backward(np.ones_like(y))
        assert layer.dL_dW is not None
        assert layer.dL_db is not None
        assert layer.dL_dW.shape == (2, 3)
        assert layer.dL_db.shape == (2,)


class TestReLU:
    """Tests for ReLU activation."""

    def test_forward_positive(self):
        """Test ReLU passes positive values unchanged."""
        relu = ReLU()
        x = np.array([[1.0, 2.0, 3.0]])
        y = relu.forward(x)
        np.testing.assert_array_equal(y, x)

    def test_forward_negative(self):
        """Test ReLU zeros out negative values."""
        relu = ReLU()
        x = np.array([[-1.0, -2.0, -3.0]])
        y = relu.forward(x)
        np.testing.assert_array_equal(y, np.zeros_like(x))

    def test_forward_mixed(self):
        """Test ReLU with mixed positive and negative values."""
        relu = ReLU()
        x = np.array([[-1.0, 0.0, 1.0, -2.0, 2.0]])
        y = relu.forward(x)
        expected = np.array([[0.0, 0.0, 1.0, 0.0, 2.0]])
        np.testing.assert_array_equal(y, expected)

    def test_backward_gradient_check(self):
        """Test backward pass using numerical gradient checking."""
        np.random.seed(42)
        relu = ReLU()
        x = np.random.randn(5, 3)
        assert gradient_check(relu, x)


class TestSigmoid:
    """Tests for Sigmoid activation."""

    def test_forward_zero(self):
        """Test sigmoid(0) = 0.5."""
        sigmoid = Sigmoid()
        x = np.array([[0.0]])
        y = sigmoid.forward(x)
        np.testing.assert_allclose(y, [[0.5]])

    def test_forward_range(self):
        """Test sigmoid output is in (0, 1)."""
        sigmoid = Sigmoid()
        x = np.random.randn(100, 10)
        y = sigmoid.forward(x)
        assert np.all(y > 0)
        assert np.all(y < 1)

    def test_backward_gradient_check(self):
        """Test backward pass using numerical gradient checking."""
        np.random.seed(42)
        sigmoid = Sigmoid()
        x = np.random.randn(5, 3)
        assert gradient_check(sigmoid, x)


class TestTanh:
    """Tests for Tanh activation."""

    def test_forward_zero(self):
        """Test tanh(0) = 0."""
        tanh = Tanh()
        x = np.array([[0.0]])
        y = tanh.forward(x)
        np.testing.assert_allclose(y, [[0.0]])

    def test_forward_range(self):
        """Test tanh output is in (-1, 1)."""
        tanh = Tanh()
        x = np.random.randn(100, 10)
        y = tanh.forward(x)
        assert np.all(y > -1)
        assert np.all(y < 1)

    def test_backward_gradient_check(self):
        """Test backward pass using numerical gradient checking."""
        np.random.seed(42)
        tanh = Tanh()
        x = np.random.randn(5, 3)
        assert gradient_check(tanh, x)
