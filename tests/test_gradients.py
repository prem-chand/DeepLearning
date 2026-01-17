"""
Gradient Checking Module
========================

This module provides tools for validating neural network implementations through
numerical gradient checking. It compares analytically computed gradients (from
backward passes) against numerically approximated gradients to ensure correctness.

Why Gradient Checking?
----------------------
Implementing backpropagation by hand is error-prone. Common mistakes include:
- Sign errors in gradient formulas
- Incorrect handling of broadcasting
- Missing terms in the chain rule
- Shape mismatches

Numerical gradient checking provides a ground truth to validate against.
If your analytical gradient matches the numerical gradient to high precision,
your implementation is almost certainly correct.

How It Works
------------
The numerical gradient is computed using centered finite differences:

    df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)

This approximation has O(ε²) error, making it more accurate than the one-sided
difference (f(x + ε) - f(x)) / ε which has O(ε) error.

Usage
-----
>>> from neural_net import Linear, ReLU
>>> from test_gradients import gradient_check
>>>
>>> # Test a single layer
>>> layer = Linear(10, 5)
>>> x = np.random.randn(4, 10)
>>> if gradient_check(layer, x):
...     print("Gradient check passed!")
... else:
...     print("Gradient check failed!")

Running Tests
-------------
Execute this file directly to run gradient checks on all implemented layers:

    $ python test_gradients.py

References
----------
- CS231n: Convolutional Neural Networks for Visual Recognition
- "Neural Networks and Deep Learning" by Michael Nielsen
- "Yes you should understand backprop" by Andrej Karpathy

Author: Phase 1 Deep Learning Assignment
"""

import numpy as np


def numerical_gradient(f, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Compute numerical gradient using centered finite differences.

    This function approximates the gradient of a scalar-valued function f
    at point x by perturbing each element of x by a small amount ε and
    observing the change in f.

    Parameters
    ----------
    f : callable
        A function that takes an ndarray and returns a scalar.
        Example: lambda x: np.sum(layer.forward(x))
    x : np.ndarray
        The point at which to evaluate the gradient. Can be any shape.
    epsilon : float, optional
        The perturbation size. Default is 1e-5, which provides a good
        balance between approximation error and numerical precision.

    Returns
    -------
    np.ndarray
        Numerical gradient of same shape as x. Each element grad[i] is
        the partial derivative df/dx[i].

    Algorithm
    ---------
    For each element x[i]:
        1. Add epsilon: x[i] += epsilon
        2. Compute f_plus = f(x)
        3. Subtract 2*epsilon: x[i] -= 2*epsilon
        4. Compute f_minus = f(x)
        5. Restore original: x[i] += epsilon
        6. grad[i] = (f_plus - f_minus) / (2 * epsilon)

    Mathematical Background
    -----------------------
    Taylor expansion of f around x:
        f(x + ε) = f(x) + εf'(x) + (ε²/2)f''(x) + O(ε³)
        f(x - ε) = f(x) - εf'(x) + (ε²/2)f''(x) + O(ε³)

    Subtracting:
        f(x + ε) - f(x - ε) = 2εf'(x) + O(ε³)

    Therefore:
        f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

    The error is O(ε²), which for ε = 1e-5 gives error ≈ 1e-10.

    Notes
    -----
    - The function must return a scalar (single number)
    - For high-dimensional x, this is slow (O(n) function evaluations)
    - Gradient checking should only be used for debugging, not training

    Examples
    --------
    >>> def f(x):
    ...     return np.sum(x ** 2)  # Gradient should be 2*x
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> grad = numerical_gradient(f, x)
    >>> grad  # Should be close to [2, 4, 6]
    array([2., 4., 6.])
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        original_value = x[idx]

        # Compute f(x + epsilon)
        x[idx] = original_value + epsilon
        f_plus = f(x)

        # Compute f(x - epsilon)
        x[idx] = original_value - epsilon
        f_minus = f(x)

        # Centered difference formula
        grad[idx] = (f_plus - f_minus) / (2 * epsilon)

        # Restore original value
        x[idx] = original_value
        it.iternext()

    return grad


def gradient_check(layer, x: np.ndarray, tolerance: float = 1e-5) -> bool:
    """
    Verify a layer's backward pass against numerical gradients.

    This function tests whether the analytical gradient computed by
    layer.backward() matches the numerical gradient computed by
    finite differences.

    Parameters
    ----------
    layer : object
        A layer with forward(x) and backward(dL_dy) methods.
        The forward method should return an ndarray.
        The backward method should return dL/dx (gradient w.r.t. input).
    x : np.ndarray
        Input tensor to test with. Shape should match layer's expected input.
    tolerance : float, optional
        Maximum allowed relative error. Default is 1e-5, which is
        appropriate for float64 computations.

    Returns
    -------
    bool
        True if the maximum relative error is below tolerance.

    Algorithm
    ---------
    1. Define loss function: L(x) = sum(layer.forward(x))
       Using sum as loss means dL/doutput = ones, simplifying the test.

    2. Compute analytical gradient:
       - Run forward pass to get output shape
       - Call backward(ones) to get dL/dx

    3. Compute numerical gradient:
       - Use centered finite differences on the loss function

    4. Compare using relative error:
       rel_error = |grad - num_grad| / max(|grad| + |num_grad|, 1e-7)

    Relative Error Interpretation
    -----------------------------
    - rel_error > 1e-2: Definitely a bug
    - rel_error 1e-2 to 1e-4: Suspicious, check carefully
    - rel_error 1e-4 to 1e-7: Acceptable for float64
    - rel_error < 1e-7: Excellent

    Notes
    -----
    - This tests dL/dx (gradient w.r.t. input), not dL/dW or dL/db
    - For testing parameter gradients, create separate test functions
    - Gradient checking is slow; use small tensors for testing

    Examples
    --------
    >>> from neural_net import ReLU
    >>> layer = ReLU()
    >>> x = np.random.randn(5, 3)
    >>> gradient_check(layer, x)
    Max Relative Error: 2.55e-11
    True

    See Also
    --------
    numerical_gradient : Computes the numerical gradient used for comparison
    """
    # Define a function that returns a scalar loss
    # Using sum of outputs means backward() receives gradient of ones
    def loss_fn(x_input):
        return layer.forward(x_input).sum()

    # Compute analytical gradient
    # First do forward pass to get output shape, then backward with ones
    output = layer.forward(x)
    grad = layer.backward(np.ones_like(output))

    # Compute numerical gradient
    num_grad = numerical_gradient(loss_fn, x)

    # Compute relative error
    # The formula: |a - b| / max(|a| + |b|, small_constant)
    # The small constant prevents division by zero when both gradients are near zero
    rel_error = np.max(np.abs(grad - num_grad) /
                       (np.maximum(1e-7, np.abs(grad) + np.abs(num_grad))))

    print(f"Max Relative Error: {rel_error:.2e}")
    return rel_error < tolerance


def gradient_check_weights(layer, x: np.ndarray, tolerance: float = 1e-5) -> bool:
    """
    Verify gradients with respect to layer weights (for Linear layers).

    This function specifically tests dL/dW and dL/db for layers that have
    learnable parameters.

    Parameters
    ----------
    layer : Linear
        A Linear layer with _w, _b, dL_dW, and dL_db attributes.
    x : np.ndarray
        Input tensor to test with.
    tolerance : float, optional
        Maximum allowed relative error.

    Returns
    -------
    bool
        True if all parameter gradients pass the check.

    Examples
    --------
    >>> from neural_net import Linear
    >>> layer = Linear(10, 5)
    >>> x = np.random.randn(4, 10)
    >>> gradient_check_weights(layer, x)
    Weight gradient - Max Relative Error: 3.09e-08
    Bias gradient - Max Relative Error: 1.23e-10
    True
    """
    # Forward and backward to compute analytical gradients
    output = layer.forward(x)
    layer.backward(np.ones_like(output))

    # Check weight gradient
    def loss_fn_w(w):
        original_w = layer._w.copy()
        layer._w = w
        loss = layer.forward(x).sum()
        layer._w = original_w
        return loss

    num_grad_w = numerical_gradient(loss_fn_w, layer._w.copy())
    rel_error_w = np.max(np.abs(layer.dL_dW - num_grad_w) /
                         (np.maximum(1e-7, np.abs(layer.dL_dW) + np.abs(num_grad_w))))
    print(f"Weight gradient - Max Relative Error: {rel_error_w:.2e}")

    # Check bias gradient
    def loss_fn_b(b):
        original_b = layer._b.copy()
        layer._b = b
        loss = layer.forward(x).sum()
        layer._b = original_b
        return loss

    num_grad_b = numerical_gradient(loss_fn_b, layer._b.copy())
    rel_error_b = np.max(np.abs(layer.dL_db - num_grad_b) /
                         (np.maximum(1e-7, np.abs(layer.dL_db) + np.abs(num_grad_b))))
    print(f"Bias gradient - Max Relative Error: {rel_error_b:.2e}")

    return rel_error_w < tolerance and rel_error_b < tolerance


if __name__ == "__main__":
    """
    Run gradient checks on all implemented layers.

    This script validates the backward pass implementations for:
    1. Linear layer (fully connected)
    2. ReLU activation
    3. Sigmoid activation
    4. Tanh activation

    Each test creates a random input tensor and compares the analytical
    gradient (from backward()) against the numerical gradient.

    Expected output: All tests should pass with relative error < 1e-5.
    """
    from deeplearning.neural_net import Linear, ReLU, Sigmoid, Tanh

    # Set seed for reproducibility
    np.random.seed(0)

    print("=" * 60)
    print("GRADIENT CHECKING VALIDATION")
    print("=" * 60)
    print()

    # Test Linear layer
    print("Testing Linear layer (input gradient):")
    print("-" * 40)
    layer = Linear(in_features=3, out_features=2)
    x = np.random.randn(5, 3)

    if gradient_check(layer, x):
        print("✓ Gradient check PASSED!")
    else:
        print("✗ Gradient check FAILED.")
    print()

    # Test ReLU
    print("Testing ReLU activation:")
    print("-" * 40)
    layer = ReLU()
    x = np.random.randn(5, 3)
    if gradient_check(layer, x):
        print("✓ Gradient check PASSED!")
    else:
        print("✗ Gradient check FAILED.")
    print()

    # Test Sigmoid
    print("Testing Sigmoid activation:")
    print("-" * 40)
    layer = Sigmoid()
    x = np.random.randn(5, 3)
    if gradient_check(layer, x):
        print("✓ Gradient check PASSED!")
    else:
        print("✗ Gradient check FAILED.")
    print()

    # Test Tanh
    print("Testing Tanh activation:")
    print("-" * 40)
    layer = Tanh()
    x = np.random.randn(5, 3)
    if gradient_check(layer, x):
        print("✓ Gradient check PASSED!")
    else:
        print("✗ Gradient check FAILED.")
    print()

    print("=" * 60)
    print("All gradient checks completed!")
    print("=" * 60)

    # Commented out: Sequential layer composition test
    # Uncomment when Sequential class is implemented
    # ---------------------------------------------
    # layer1 = Linear(in_features=4, out_features=4)
    # layer2 = ReLU()
    # x = np.random.randn(6, 4)
    # layer = Sequential([layer1, layer2])
    # if gradient_check(layer, x):
    #     print("Gradient check passed!")
    # else:
    #     print("Gradient check failed.")
