#!/usr/bin/env python3
"""
MNIST Training Script
=====================

This script trains the TwoLayerMLP on the MNIST handwritten digits dataset
and evaluates its performance.

Expected Performance:
- Training accuracy: >98%
- Test accuracy: >95%

Usage:
    python scripts/train_mnist.py

Author: Deep Learning Assignment
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from deeplearning import TwoLayerMLP


def load_mnist(data_dir: Path):
    """
    Load preprocessed MNIST data.

    Parameters
    ----------
    data_dir : Path
        Directory containing the .npy files

    Returns
    -------
    tuple
        (X_train, y_train), (X_test, y_test)
    """
    X_train = np.load(data_dir / "train_images.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    X_test = np.load(data_dir / "test_images.npy")
    y_test = np.load(data_dir / "test_labels.npy")

    return (X_train, y_train), (X_test, y_test)


def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    Create mini-batches from data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (N, D)
    y : np.ndarray
        Labels of shape (N,)
    batch_size : int
        Size of each batch
    shuffle : bool
        Whether to shuffle data before batching

    Yields
    ------
    tuple
        (X_batch, y_batch) for each batch
    """
    n_samples = len(X)

    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]


def train_epoch(model: TwoLayerMLP, X: np.ndarray, y: np.ndarray,
                batch_size: int, learning_rate: float) -> float:
    """
    Train for one epoch.

    Returns
    -------
    float
        Average loss for the epoch
    """
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in create_batches(X, y, batch_size):
        loss = model.train_step(X_batch, y_batch, lr=learning_rate)
        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


def main():
    """Train and evaluate the model on MNIST."""
    print("=" * 60)
    print("MNIST Training with TwoLayerMLP")
    print("=" * 60)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Configuration
    config = {
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.1,
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Load data
    print("Loading MNIST data...")
    data_dir = Path(__file__).parent.parent / "data" / "mnist"

    if not (data_dir / "train_images.npy").exists():
        print("Error: MNIST data not found. Please run download_mnist.py first.")
        return

    (X_train, y_train), (X_test, y_test) = load_mnist(data_dir)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()

    # Initialize model
    print("Initializing model...")
    model = TwoLayerMLP()

    # Count parameters
    n_params = 0
    for layer in model.layers:
        if hasattr(layer, '_w'):
            n_params += layer._w.size + layer._b.size
    print(f"  Total parameters: {n_params:,}")
    print()

    # Training loop
    print("Training...")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Time':>8}")
    print("-" * 60)

    best_test_acc = 0

    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()

        # Train
        avg_loss = train_epoch(
            model, X_train, y_train,
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"]
        )

        # Evaluate (on subset for speed during training)
        train_acc = model.evaluate(X_train[:5000], y_train[:5000])
        test_acc = model.evaluate(X_test, y_test)

        elapsed = time.time() - start_time

        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"{epoch:>6} | {avg_loss:>10.4f} | {train_acc*100:>9.2f}% | {test_acc*100:>9.2f}% | {elapsed:>6.1f}s")

    print("-" * 60)
    print()

    # Final evaluation
    print("Final Evaluation...")
    print("-" * 40)
    final_train_acc = model.evaluate(X_train, y_train)
    final_test_acc = model.evaluate(X_test, y_test)

    print(f"Training accuracy: {final_train_acc * 100:.2f}%")
    print(f"Test accuracy:     {final_test_acc * 100:.2f}%")
    print()

    # Performance check
    print("Performance Check:")
    print("-" * 40)

    train_target = 0.98
    test_target = 0.95

    train_pass = final_train_acc >= train_target
    test_pass = final_test_acc >= test_target

    print(f"Training accuracy >= {train_target*100:.0f}%: {'PASS' if train_pass else 'FAIL'}")
    print(f"Test accuracy >= {test_target*100:.0f}%:     {'PASS' if test_pass else 'FAIL'}")
    print()

    if train_pass and test_pass:
        print("All targets met!")
    else:
        print("Some targets not met. Consider:")
        print("  - Training for more epochs")
        print("  - Adjusting learning rate")
        print("  - Using learning rate decay")

    # Sample predictions
    print()
    print("Sample Predictions:")
    print("-" * 40)

    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    predictions = model.predict(X_test[sample_indices])
    actuals = y_test[sample_indices]

    print("Index | Predicted | Actual | Correct")
    print("-" * 40)
    for i, (idx, pred, actual) in enumerate(zip(sample_indices, predictions, actuals)):
        correct = "Yes" if pred == actual else "No"
        print(f"{idx:>5} | {pred:>9} | {actual:>6} | {correct}")


if __name__ == "__main__":
    main()
