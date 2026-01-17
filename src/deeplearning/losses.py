"""
Loss Functions Module
=====================

This module implements common loss functions for neural network training from scratch
using only NumPy. Loss functions measure the discrepancy between predicted outputs
and ground truth labels, providing the signal that drives learning through backpropagation.

Overview
--------
Loss functions serve two purposes in neural networks:
1. **Forward pass**: Compute a scalar value measuring prediction error
2. **Backward pass**: Compute gradients to propagate back through the network

Both implemented loss functions follow this interface:
- `forward(y_pred, y_true)` -> scalar loss value
- `backward()` -> gradient tensor of same shape as y_pred

Loss Functions Included
-----------------------
- **MSELoss**: Mean Squared Error for regression tasks
- **CrossEntropyLoss**: Softmax + Negative Log-Likelihood for classification

Example Usage
-------------
>>> import numpy as np
>>> from losses import CrossEntropyLoss
>>>
>>> # Classification with 3 classes, batch of 4 samples
>>> logits = np.random.randn(4, 3)  # Raw model outputs
>>> targets = np.array([0, 1, 2, 1])  # Class indices
>>>
>>> loss_fn = CrossEntropyLoss()
>>> loss = loss_fn.forward(logits, targets)
>>> print(f"Loss: {loss:.4f}")
>>>
>>> # Get gradients for backpropagation
>>> dL_dlogits = loss_fn.backward()
>>> print(f"Gradient shape: {dL_dlogits.shape}")  # (4, 3)

References
----------
- CS231n: Convolutional Neural Networks for Visual Recognition
- "Deep Learning" by Goodfellow, Bengio, and Courville

Author: Prem Chand
"""

import numpy as np
from typing import Tuple


class MSELoss:
    """
    Mean Squared Error Loss for regression tasks.

    MSE measures the average squared difference between predictions and targets.
    It is the most common loss function for regression problems and penalizes
    large errors more heavily than small ones due to the squaring operation.

    Mathematical Formulation
    ------------------------
    Forward pass:
        L = (1/N) * sum((y_pred - y_true)^2)

    Backward pass:
        dL/dy_pred = (2/N) * (y_pred - y_true)

    where N is the total number of elements in the prediction tensor.

    Properties
    ----------
    - Always non-negative (L >= 0)
    - Minimum value of 0 when predictions exactly match targets
    - Convex function, making optimization well-behaved
    - Sensitive to outliers due to squaring

    Attributes
    ----------
    _y_pred : np.ndarray
        Cached predictions from forward pass.
    _y_true : np.ndarray
        Cached targets from forward pass.

    Examples
    --------
    >>> mse = MSELoss()
    >>> y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y_true = np.array([[1.5, 2.5], [2.5, 3.5]])
    >>> loss = mse.forward(y_pred, y_true)
    >>> loss  # Average squared error
    0.25
    >>> grad = mse.backward()
    >>> grad.shape
    (2, 2)

    Notes
    -----
    For multi-output regression, MSE averages over all outputs and samples.
    Some implementations average only over the batch dimension; this one
    averages over all elements for consistency with common practice.

    See Also
    --------
    - MAE (Mean Absolute Error): Less sensitive to outliers
    - Huber Loss: Combines MSE and MAE properties
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Mean Squared Error between predictions and targets.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values of any shape.
        y_true : np.ndarray
            Ground truth values, must match shape of y_pred.

        Returns
        -------
        float
            Scalar loss value representing average squared error.

        Notes
        -----
        Values are cached for use in backward pass.
        """
        self._y_pred = y_pred
        self._y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self) -> np.ndarray:
        """
        Compute gradient of MSE loss with respect to predictions.

        Returns
        -------
        np.ndarray
            Gradient dL/dy_pred of same shape as predictions.

        Mathematical Derivation
        -----------------------
        L = (1/N) * sum((y_pred - y_true)^2)

        dL/dy_pred[i] = (2/N) * (y_pred[i] - y_true[i])

        The 2 comes from the power rule (d/dx x^2 = 2x), and
        1/N comes from the mean operation.

        Notes
        -----
        Must call forward() before backward() to cache values.
        """
        return 2 * (self._y_pred - self._y_true) / self._y_true.size


class CrossEntropyLoss:
    """
    Cross-Entropy Loss combining Softmax and Negative Log-Likelihood.

    This is the standard loss function for multi-class classification. It first
    applies the softmax function to convert raw logits into probabilities, then
    computes the negative log-likelihood of the correct class.

    Mathematical Formulation
    ------------------------
    Softmax (converting logits to probabilities):
        p_i = exp(x_i) / sum_j(exp(x_j))

    Negative Log-Likelihood:
        L = -(1/N) * sum_n(log(p_n[target_n]))

    Combined gradient (elegant result):
        dL/dx_i = (1/N) * (p_i - 1{i == target})

    where 1{i == target} is 1 if i is the target class, 0 otherwise.

    Numerical Stability
    -------------------
    The implementation uses the log-sum-exp trick to prevent overflow:
        log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    This is achieved by shifting logits: x_shifted = x - max(x), which
    doesn't change the softmax output but prevents exp() overflow.

    Attributes
    ----------
    _probs : np.ndarray
        Cached softmax probabilities from forward pass.
    _targets : np.ndarray
        Cached targets from forward pass.
    _N : int
        Batch size, used for averaging.

    Examples
    --------
    >>> ce_loss = CrossEntropyLoss()
    >>> logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    >>> targets = np.array([0, 1])  # Class indices
    >>> loss = ce_loss.forward(logits, targets)
    >>> print(f"Loss: {loss:.4f}")
    Loss: 0.4076
    >>> grad = ce_loss.backward()
    >>> grad.shape
    (2, 3)

    With one-hot encoded targets:
    >>> targets_onehot = np.array([[1, 0, 0], [0, 1, 0]])
    >>> loss = ce_loss.forward(logits, targets_onehot)

    Notes
    -----
    - For binary classification, use this with 2 classes or use Binary Cross-Entropy
    - The gradient has a beautiful simple form: probability minus indicator
    - Cross-entropy is equivalent to KL divergence when targets are one-hot

    See Also
    --------
    - Binary Cross-Entropy: For binary classification
    - Focal Loss: Addresses class imbalance
    - Label Smoothing: Regularization technique
    """

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss between logits and targets.

        Parameters
        ----------
        logits : np.ndarray
            Raw (unnormalized) scores of shape (N, C) where N is batch size
            and C is the number of classes.
        targets : np.ndarray
            Either:
            - Class indices of shape (N,) with values in [0, C-1]
            - One-hot encoded targets of shape (N, C)

        Returns
        -------
        float
            Scalar loss value (average negative log-likelihood).

        Algorithm
        ---------
        1. Shift logits for numerical stability: x -= max(x)
        2. Compute softmax: exp(x) / sum(exp(x))
        3. Extract probabilities of correct classes
        4. Compute negative log and average over batch

        Notes
        -----
        The log-sum-exp trick prevents numerical overflow that would occur
        when computing exp() of large positive numbers. Subtracting the max
        ensures all exponents are <= 0, so exp() values are in (0, 1].
        """
        # Shift logits for numerical stability (log-sum-exp trick)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        N = logits.shape[0]
        if targets.ndim == 1:
            # Targets are class indices
            correct_logprobs = -np.log(probs[np.arange(N), targets])
        else:
            # Targets are one-hot encoded
            correct_logprobs = -np.log(np.sum(probs * targets, axis=1))

        loss = np.sum(correct_logprobs) / N

        # Cache for backward pass
        self._probs = probs
        self._targets = targets
        self._N = N

        return loss

    def backward(self) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss with respect to logits.

        Returns
        -------
        np.ndarray
            Gradient dL/dlogits of shape (N, C).

        Mathematical Derivation
        -----------------------
        The combined softmax + cross-entropy gradient has an elegant form:

        For a single sample with target class k:
            dL/dx_i = p_i - 1{i == k}

        This means:
        - For the correct class: gradient = probability - 1 (negative, pushes up)
        - For incorrect classes: gradient = probability (positive, pushes down)

        The derivation involves the quotient rule on softmax and simplifies
        beautifully. See CS231n notes for the full derivation.

        For batch processing, we average over all samples.

        Notes
        -----
        Must call forward() before backward() to cache probabilities and targets.
        """
        dL_dlogits = self._probs.copy()
        if self._targets.ndim == 1:
            # Targets are class indices: subtract 1 from correct class probabilities
            dL_dlogits[np.arange(self._N), self._targets] -= 1
        else:
            # Targets are one-hot: subtract the one-hot targets
            dL_dlogits -= self._targets

        # Average over batch
        dL_dlogits /= self._N
        return dL_dlogits
