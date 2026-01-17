"""
Test Suite for Autograd MLP
============================

This module tests the autograd-based MLP implementation, including:
1. Layer implementations (Linear, ReLU, etc.)
2. Forward pass correctness
3. Gradient verification against manual implementation
4. Training convergence

Author: Prem Chand
"""

import pytest
import numpy as np
from deeplearning.autograd import Tensor
from deeplearning.autograd_mlp import (
    Layer,
    LinearAutograd,
    ReLUAutograd,
    SigmoidAutograd,
    TanhAutograd,
    SoftmaxAutograd,
    CrossEntropyLossAutograd,
    TwoLayerMLPAutograd,
    cross_entropy_loss,
    compare_gradients_with_manual,
)


class TestLinearAutograd:
    """Test LinearAutograd layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = LinearAutograd(784, 128)
        assert layer.W.shape == (784, 128), "Weight shape incorrect"
        assert layer.b.shape == (128,), "Bias shape incorrect"
        assert layer.W.requires_grad, "Weights should require grad"
        assert layer.b.requires_grad, "Bias should require grad"

    def test_forward(self):
        """Test forward pass."""
        layer = LinearAutograd(10, 5)
        x = Tensor(np.random.randn(32, 10))
        y = layer(x)
        assert y.shape == (32, 5), "Output shape incorrect"

    def test_parameters(self):
        """Test parameters method."""
        layer = LinearAutograd(10, 5)
        params = layer.parameters()
        assert len(params) == 2, "Should have 2 parameters (W and b)"
        assert params[0] is layer.W
        assert params[1] is layer.b

    def test_gradient_computation(self):
        """Test gradient computation through linear layer."""
        np.random.seed(42)
        layer = LinearAutograd(10, 5)
        x = Tensor(np.random.randn(4, 10))

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.W.grad is not None, "W gradient should exist"
        assert layer.b.grad is not None, "b gradient should exist"
        assert layer.W.grad.shape == (10, 5), "W gradient shape incorrect"
        assert layer.b.grad.shape == (5,), "b gradient shape incorrect"


class TestActivationLayers:
    """Test activation layers."""

    def test_relu_forward(self):
        """Test ReLU forward pass."""
        relu = ReLUAutograd()
        x = Tensor([[-1, 0, 1], [2, -2, 3]])
        y = relu(x)
        expected = np.array([[0, 0, 1], [2, 0, 3]])
        assert np.allclose(y.data, expected), "ReLU forward incorrect"

    def test_relu_backward(self):
        """Test ReLU backward pass."""
        relu = ReLUAutograd()
        x = Tensor([[-1, 0, 1], [2, -2, 3]])
        y = relu(x)
        y.sum().backward()

        expected_grad = np.array([[0, 0, 1], [1, 0, 1]])
        assert np.allclose(x.grad, expected_grad), "ReLU gradient incorrect"

    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        sigmoid = SigmoidAutograd()
        x = Tensor([[0, 1, -1]])
        y = sigmoid(x)

        expected = 1 / (1 + np.exp(-x.data))
        assert np.allclose(y.data, expected), "Sigmoid forward incorrect"

    def test_tanh_forward(self):
        """Test Tanh forward pass."""
        tanh = TanhAutograd()
        x = Tensor([[0, 1, -1]])
        y = tanh(x)

        expected = np.tanh(x.data)
        assert np.allclose(y.data, expected), "Tanh forward incorrect"

    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        softmax = SoftmaxAutograd()
        x = Tensor([[1, 2, 3], [1, 1, 1]])
        y = softmax(x)

        # Check softmax sums to 1
        assert np.allclose(y.data.sum(axis=-1), [1, 1]), "Softmax should sum to 1"


class TestCrossEntropyLoss:
    """Test cross-entropy loss function."""

    def test_basic_loss(self):
        """Test basic loss computation."""
        logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        targets = np.array([0, 1])

        loss = cross_entropy_loss(logits, targets)
        assert loss.data.shape == (), "Loss should be scalar"
        assert loss.data > 0, "Loss should be positive"

    def test_perfect_prediction(self):
        """Test loss with near-perfect predictions."""
        # Very high logit for correct class
        logits = Tensor([[10.0, -10.0, -10.0]])
        targets = np.array([0])

        loss = cross_entropy_loss(logits, targets)
        assert loss.data < 0.01, "Loss should be very small for perfect prediction"

    def test_uniform_prediction(self):
        """Test loss with uniform predictions."""
        logits = Tensor([[0.0, 0.0, 0.0]])  # Uniform softmax
        targets = np.array([0])

        loss = cross_entropy_loss(logits, targets)
        expected_loss = -np.log(1/3)  # Log of 1/3 probability
        assert np.allclose(loss.data, expected_loss, rtol=1e-5), \
            "Uniform prediction loss incorrect"

    def test_gradient_computation(self):
        """Test gradient computation."""
        logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        targets = np.array([0, 1])

        loss = cross_entropy_loss(logits, targets)
        loss.backward()

        assert logits.grad is not None, "Gradient should exist"
        assert logits.grad.shape == (2, 3), "Gradient shape incorrect"

    def test_class_matches_function(self):
        """Test that CrossEntropyLossAutograd class matches convenience function."""
        logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        targets = np.array([0, 1])

        # Using class
        loss_fn = CrossEntropyLossAutograd()
        loss_class = loss_fn(logits, targets)

        # Using function (creates new logits to avoid grad accumulation)
        logits2 = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        loss_func = cross_entropy_loss(logits2, targets)

        assert np.allclose(loss_class.data, loss_func.data), \
            "Class and function should produce same loss"

    def test_matches_manual_loss(self):
        """Test that autograd loss matches manual CrossEntropyLoss."""
        from deeplearning.losses import CrossEntropyLoss

        logits_data = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        targets = np.array([0, 1])

        # Manual
        manual_loss = CrossEntropyLoss()
        manual_value = manual_loss.forward(logits_data, targets)

        # Autograd
        logits = Tensor(logits_data)
        autograd_loss = cross_entropy_loss(logits, targets)

        assert np.allclose(manual_value, autograd_loss.data), \
            "Autograd loss should match manual implementation"

    def test_gradient_matches_manual(self):
        """Test that autograd gradient matches manual backward."""
        from deeplearning.losses import CrossEntropyLoss

        logits_data = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        targets = np.array([0, 1])

        # Manual
        manual_loss = CrossEntropyLoss()
        manual_loss.forward(logits_data, targets)
        manual_grad = manual_loss.backward()

        # Autograd
        logits = Tensor(logits_data.copy())
        loss = cross_entropy_loss(logits, targets)
        loss.backward()

        assert np.allclose(logits.grad, manual_grad), \
            "Autograd gradient should match manual implementation"


class TestTwoLayerMLPAutograd:
    """Test the full MLP model."""

    def test_initialization(self):
        """Test model initialization."""
        model = TwoLayerMLPAutograd()
        assert len(model.layers) == 3, "Should have 3 layers"
        assert isinstance(model.layers[0], LinearAutograd)
        assert isinstance(model.layers[1], ReLUAutograd)
        assert isinstance(model.layers[2], LinearAutograd)

    def test_forward(self):
        """Test forward pass."""
        model = TwoLayerMLPAutograd()
        x = np.random.randn(32, 784)
        y = model.forward(x)
        assert y.shape == (32, 10), "Output shape incorrect"

    def test_parameters(self):
        """Test parameters method."""
        model = TwoLayerMLPAutograd()
        params = model.parameters()
        assert len(params) == 4, "Should have 4 parameters (2 W + 2 b)"

        # Check total parameter count
        total_params = sum(p.data.size for p in params)
        expected = 784 * 128 + 128 + 128 * 10 + 10  # W1, b1, W2, b2
        assert total_params == expected, "Total parameters incorrect"

    def test_zero_grad(self):
        """Test zero_grad method."""
        model = TwoLayerMLPAutograd()
        x = np.random.randn(32, 784)
        y = np.random.randint(0, 10, 32)

        # Do a forward/backward pass
        model.train_step(x, y, lr=0.01)

        # Gradients should be non-zero
        for param in model.parameters():
            assert not np.allclose(param.grad, 0), "Grad should be non-zero after training"

        # Zero gradients
        model.zero_grad()

        # Gradients should be zero
        for param in model.parameters():
            assert np.allclose(param.grad, 0), "Grad should be zero after zero_grad"

    def test_train_step(self):
        """Test training step."""
        np.random.seed(42)
        model = TwoLayerMLPAutograd()
        x = np.random.randn(64, 784)
        y = np.random.randint(0, 10, 64)

        loss = model.train_step(x, y, lr=0.01)
        assert isinstance(loss, float), "Loss should be float"
        assert loss > 0, "Loss should be positive"

    def test_predict(self):
        """Test prediction."""
        model = TwoLayerMLPAutograd()
        x = np.random.randn(10, 784)
        predictions = model.predict(x)

        assert predictions.shape == (10,), "Predictions shape incorrect"
        assert all(0 <= p < 10 for p in predictions), "Predictions should be in [0, 9]"

    def test_evaluate(self):
        """Test evaluation."""
        model = TwoLayerMLPAutograd()
        x = np.random.randn(100, 784)
        y = np.random.randint(0, 10, 100)

        accuracy = model.evaluate(x, y)
        assert 0 <= accuracy <= 1, "Accuracy should be in [0, 1]"

    def test_training_reduces_loss(self):
        """Test that training reduces loss over time."""
        np.random.seed(42)
        model = TwoLayerMLPAutograd()

        # Use a small dataset for quick test
        x = np.random.randn(100, 784)
        y = np.random.randint(0, 10, 100)

        # Initial loss
        model.zero_grad()
        logits = model.forward(x)
        initial_loss = cross_entropy_loss(logits, y).item()

        # Train for several steps
        for _ in range(50):
            model.train_step(x, y, lr=0.1)

        # Final loss
        model.zero_grad()
        logits = model.forward(x)
        final_loss = cross_entropy_loss(logits, y).item()

        assert final_loss < initial_loss, "Training should reduce loss"


class TestGradientVerification:
    """Test gradient verification against manual implementation."""

    def test_gradients_match_manual(self):
        """Test that autograd gradients match manual implementation."""
        results = compare_gradients_with_manual(seed=42)

        assert results['loss_match'], "Loss should match"
        assert results['w1_grad_match'], "W1 gradient should match"
        assert results['b1_grad_match'], "b1 gradient should match"
        assert results['w2_grad_match'], "W2 gradient should match"
        assert results['b2_grad_match'], "b2 gradient should match"

    def test_gradient_differences_small(self):
        """Test that gradient differences are very small."""
        results = compare_gradients_with_manual(seed=42)

        assert results['w1_grad_max_diff'] < 1e-10, "W1 gradient diff should be tiny"
        assert results['w2_grad_max_diff'] < 1e-10, "W2 gradient diff should be tiny"


class TestNumericalGradients:
    """Test gradients using numerical approximation."""

    def numerical_gradient(self, f, x, epsilon=1e-5):
        """Compute numerical gradient using centered differences."""
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = x[idx]

            x[idx] = old_val + epsilon
            fx_plus = f(x.copy())

            x[idx] = old_val - epsilon
            fx_minus = f(x.copy())

            grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
            x[idx] = old_val
            it.iternext()
        return grad

    def test_linear_gradient_numerical(self):
        """Numerical gradient check for linear layer."""
        np.random.seed(42)

        layer = LinearAutograd(5, 3)
        x_data = np.random.randn(2, 5)

        # Analytical gradient
        x = Tensor(x_data.copy())
        y = layer(x)
        y.sum().backward()
        analytical_grad = layer.W.grad.copy()

        # Numerical gradient
        def f(w):
            w_tensor = Tensor(w, requires_grad=False)
            x_tensor = Tensor(x_data, requires_grad=False)
            return (x_tensor.data @ w + layer.b.data).sum()

        numerical_grad = self.numerical_gradient(f, layer.W.data.copy())

        assert np.allclose(analytical_grad, numerical_grad, atol=1e-5), \
            "Numerical gradient check failed for linear layer"


class TestCustomArchitecture:
    """Test custom MLP architectures."""

    def test_different_hidden_size(self):
        """Test MLP with different hidden size."""
        model = TwoLayerMLPAutograd(hidden_size=256)
        x = np.random.randn(32, 784)
        y = model.forward(x)
        assert y.shape == (32, 10)

    def test_different_input_size(self):
        """Test MLP with different input size."""
        model = TwoLayerMLPAutograd(input_size=100, hidden_size=50, output_size=5)
        x = np.random.randn(32, 100)
        y = model.forward(x)
        assert y.shape == (32, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
