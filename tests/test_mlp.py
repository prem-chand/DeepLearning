"""
Unit tests for the TwoLayerMLP model.

This module contains pytest-compatible tests for validating
the complete MLP training pipeline.
"""

import numpy as np
import pytest

from deeplearning.mnist_mlp import TwoLayerMLP


class TestTwoLayerMLP:
    """Tests for TwoLayerMLP model."""

    def test_init(self):
        """Test model initialization."""
        model = TwoLayerMLP()
        assert len(model.layers) == 3
        assert model.loss_fn is not None

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        model = TwoLayerMLP()
        x = np.random.randn(32, 784)
        y = model.forward(x)
        assert y.shape == (32, 10)

    def test_predict_shape(self):
        """Test predict returns class indices."""
        model = TwoLayerMLP()
        x = np.random.randn(32, 784)
        predictions = model.predict(x)
        assert predictions.shape == (32,)
        assert predictions.dtype in [np.int64, np.int32, np.intp]
        assert np.all(predictions >= 0)
        assert np.all(predictions < 10)

    def test_train_step_returns_loss(self):
        """Test train_step returns a valid loss value."""
        np.random.seed(42)
        model = TwoLayerMLP()
        x = np.random.randn(32, 784)
        y = np.random.randint(0, 10, 32)
        loss = model.train_step(x, y, lr=0.01)
        assert isinstance(loss, (float, np.floating))
        assert np.isfinite(loss)
        assert loss > 0  # Cross-entropy is always positive

    def test_train_step_updates_weights(self):
        """Test that train_step actually updates weights."""
        np.random.seed(42)
        model = TwoLayerMLP()
        x = np.random.randn(32, 784)
        y = np.random.randint(0, 10, 32)

        # Save original weights
        w1_before = model.layers[0]._w.copy()
        w2_before = model.layers[2]._w.copy()

        model.train_step(x, y, lr=0.01)

        # Check weights changed
        assert not np.allclose(model.layers[0]._w, w1_before)
        assert not np.allclose(model.layers[2]._w, w2_before)

    def test_training_reduces_loss(self):
        """Test that multiple training steps reduce loss on same batch."""
        np.random.seed(42)
        model = TwoLayerMLP()
        x = np.random.randn(64, 784)
        y = np.random.randint(0, 10, 64)

        # Train for several steps
        losses = []
        for _ in range(10):
            loss = model.train_step(x, y, lr=0.1)
            losses.append(loss)

        # Loss should generally decrease (allow some fluctuation)
        assert losses[-1] < losses[0]

    def test_evaluate_accuracy(self):
        """Test evaluate method returns valid accuracy."""
        model = TwoLayerMLP()
        x = np.random.randn(100, 784)
        y = np.random.randint(0, 10, 100)
        accuracy = model.evaluate(x, y)
        assert 0 <= accuracy <= 1

    def test_evaluate_perfect_accuracy(self):
        """Test evaluate returns 1.0 when all predictions are correct."""
        np.random.seed(42)
        model = TwoLayerMLP()
        x = np.random.randn(10, 784)

        # Get model predictions and use them as targets
        predictions = model.predict(x)
        accuracy = model.evaluate(x, predictions)
        assert accuracy == 1.0
