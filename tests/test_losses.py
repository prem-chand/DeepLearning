"""
Unit tests for loss functions.

This module contains pytest-compatible tests for validating
the forward and backward passes of loss functions.
"""

import numpy as np
import pytest

from deeplearning.losses import CrossEntropyLoss, MSELoss


class TestMSELoss:
    """Tests for Mean Squared Error loss."""

    def test_forward_zero_loss(self):
        """Test MSE is zero when predictions match targets."""
        mse = MSELoss()
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        loss = mse.forward(y_pred, y_true)
        assert loss == 0.0

    def test_forward_computation(self):
        """Test MSE computation is correct."""
        mse = MSELoss()
        y_pred = np.array([[1.0, 2.0]])
        y_true = np.array([[2.0, 4.0]])
        loss = mse.forward(y_pred, y_true)
        # MSE = mean((1-2)^2 + (2-4)^2) = mean(1 + 4) = 2.5
        assert loss == 2.5

    def test_backward_shape(self):
        """Test backward returns gradient of correct shape."""
        mse = MSELoss()
        y_pred = np.random.randn(32, 10)
        y_true = np.random.randn(32, 10)
        mse.forward(y_pred, y_true)
        grad = mse.backward()
        assert grad.shape == y_pred.shape

    def test_backward_computation(self):
        """Test backward gradient computation."""
        mse = MSELoss()
        y_pred = np.array([[1.0, 2.0]])
        y_true = np.array([[2.0, 4.0]])
        mse.forward(y_pred, y_true)
        grad = mse.backward()
        # dL/dy_pred = 2 * (y_pred - y_true) / N
        # = 2 * [[-1, -2]] / 2 = [[-1, -2]]
        expected = np.array([[-1.0, -2.0]])
        np.testing.assert_allclose(grad, expected)


class TestCrossEntropyLoss:
    """Tests for Cross-Entropy loss."""

    def test_forward_with_indices(self):
        """Test forward pass with class indices."""
        ce = CrossEntropyLoss()
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = np.array([0, 1])
        loss = ce.forward(logits, targets)
        # With large logits, softmax is nearly one-hot, loss near 0
        assert loss < 0.1

    def test_forward_with_onehot(self):
        """Test forward pass with one-hot encoded targets."""
        ce = CrossEntropyLoss()
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        loss = ce.forward(logits, targets)
        assert loss < 0.1

    def test_forward_uniform_logits(self):
        """Test loss with uniform predictions."""
        ce = CrossEntropyLoss()
        logits = np.zeros((2, 3))  # Uniform predictions
        targets = np.array([0, 1])
        loss = ce.forward(logits, targets)
        # With uniform predictions, loss = -log(1/3) = log(3) ≈ 1.099
        np.testing.assert_allclose(loss, np.log(3), rtol=1e-5)

    def test_backward_shape(self):
        """Test backward returns gradient of correct shape."""
        ce = CrossEntropyLoss()
        logits = np.random.randn(32, 10)
        targets = np.random.randint(0, 10, 32)
        ce.forward(logits, targets)
        grad = ce.backward()
        assert grad.shape == logits.shape

    def test_backward_gradient_sum(self):
        """Test that gradients sum to zero for each sample."""
        ce = CrossEntropyLoss()
        logits = np.random.randn(4, 5)
        targets = np.random.randint(0, 5, 4)
        ce.forward(logits, targets)
        grad = ce.backward()
        # For cross-entropy, each row of gradients should sum to 0
        # (probability - 1 for correct class + probabilities for others = 0)
        row_sums = np.sum(grad, axis=1)
        np.testing.assert_allclose(row_sums, 0, atol=1e-10)

    def test_numerical_stability(self):
        """Test numerical stability with large logits."""
        ce = CrossEntropyLoss()
        # Large logits that would cause overflow without stability trick
        logits = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])
        targets = np.array([0, 1])
        loss = ce.forward(logits, targets)
        # Should not be NaN or Inf
        assert np.isfinite(loss)
        assert loss < 0.1  # Should be near 0 for correct predictions
