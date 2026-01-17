"""
Train Value-based MLP on MNIST Dataset
=======================================

This script trains a scalar-based MLP (using Value class) on MNIST.
Due to the computational cost of scalar operations, we train on a
small subset of the data for demonstration purposes.

For production use, the vectorized numpy implementation is recommended.
This implementation is educational and demonstrates how autograd works
at the scalar level.
"""

import numpy as np
from deeplearning.value_mlp import ValueMLP, train_step_value_mlp, evaluate_value_mlp
import random
import time


def load_mnist_subset(n_train=100, n_test=50):
    """
    Load a small subset of MNIST for testing.
    
    If MNIST data is not available, generates synthetic data.
    
    Parameters
    ----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
        
    Returns
    -------
    tuple
        (x_train, y_train, x_test, y_test)
    """
    try:
        # Try to load from numpy files if available
        import os
        data_dir = "data"
        
        if os.path.exists(f"{data_dir}/mnist_train.npz"):
            print("Loading MNIST from local files...")
            train_data = np.load(f"{data_dir}/mnist_train.npz")
            test_data = np.load(f"{data_dir}/mnist_test.npz")
            
            x_train_full = train_data['images']
            y_train_full = train_data['labels']
            x_test_full = test_data['images']
            y_test_full = test_data['labels']
            
            # Take subset
            x_train = x_train_full[:n_train]
            y_train = y_train_full[:n_train]
            x_test = x_test_full[:n_test]
            y_test = y_test_full[:n_test]
            
            # Flatten images
            x_train = x_train.reshape(n_train, -1)
            x_test = x_test.reshape(n_test, -1)
            
            # Normalize
            x_train = (x_train - 127.5) / 127.5
            x_test = (x_test - 127.5) / 127.5
            
            print(f"Loaded {n_train} training and {n_test} test samples")
            return x_train, y_train, x_test, y_test
            
    except Exception as e:
        print(f"Could not load MNIST: {e}")
    
    # Fallback: Generate synthetic data
    print("Generating synthetic data (MNIST not available)...")
    x_train = np.random.randn(n_train, 784) * 0.5
    y_train = np.random.randint(0, 10, n_train)
    x_test = np.random.randn(n_test, 784) * 0.5
    y_test = np.random.randint(0, 10, n_test)
    
    return x_train, y_train, x_test, y_test


def train_value_mlp_on_mnist(
    n_train=100,
    n_test=50,
    hidden_size=32,
    epochs=5,
    lr=0.01,
    batch_size=10
):
    """
    Train Value-based MLP on MNIST subset.
    
    Parameters
    ----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    hidden_size : int
        Size of hidden layer
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    batch_size : int
        Batch size for training
    """
    print("=" * 70)
    print("TRAINING VALUE-BASED MLP ON MNIST")
    print("=" * 70)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    print("\nLoading data...")
    x_train, y_train, x_test, y_test = load_mnist_subset(n_train, n_test)
    
    # Create model
    print(f"\nCreating MLP: [784, {hidden_size}, 10]")
    model = ValueMLP([784, hidden_size, 10])
    print(f"Total parameters: {model.num_parameters():,}")
    
    # Training
    print(f"\nTraining for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("-" * 70)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Shuffle data
        indices = np.random.permutation(n_train)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Train on batches
        epoch_loss = 0.0
        n_batches = (n_train + batch_size - 1) // batch_size
        
        for i in range(0, n_train, batch_size):
            batch_end = min(i + batch_size, n_train)
            x_batch = x_train_shuffled[i:batch_end]
            y_batch = y_train_shuffled[i:batch_end]
            
            loss = train_step_value_mlp(model, x_batch, y_batch, lr)
            epoch_loss += loss
            
            # Progress indicator
            if (i // batch_size) % max(1, n_batches // 5) == 0:
                print(f"  Batch {i//batch_size + 1}/{n_batches}: Loss = {loss:.4f}")
        
        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        train_acc = evaluate_value_mlp(model, x_train[:min(50, n_train)], 
                                       y_train[:min(50, n_train)])
        test_acc = evaluate_value_mlp(model, x_test, y_test)
        
        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Train Acc: {train_acc * 100:.1f}%")
        print(f"  Test Acc: {test_acc * 100:.1f}%")
        print(f"  Time: {epoch_time:.1f}s")
        print("-" * 70)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_train_acc = evaluate_value_mlp(model, x_train[:min(100, n_train)], 
                                         y_train[:min(100, n_train)])
    final_test_acc = evaluate_value_mlp(model, x_test, y_test)
    
    print(f"  Training Accuracy: {final_train_acc * 100:.2f}%")
    print(f"  Test Accuracy: {final_test_acc * 100:.2f}%")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return model


def quick_demo():
    """
    Quick demonstration with minimal data.
    """
    print("\n" + "=" * 70)
    print("QUICK DEMO - Value-based MLP")
    print("=" * 70)
    
    # Very small dataset for quick demo
    print("\nTraining on 50 samples (quick demo)...")
    model = train_value_mlp_on_mnist(
        n_train=50,
        n_test=20,
        hidden_size=16,
        epochs=3,
        lr=0.05,
        batch_size=5
    )
    
    print("\nDemo complete!")
    print("\nNote: This is a scalar-based implementation for educational purposes.")
    print("For production use on full MNIST, use the vectorized implementation.")


def full_demo():
    """
    More comprehensive demo with more data.
    """
    print("\n" + "=" * 70)
    print("FULL DEMO - Value-based MLP")
    print("=" * 70)
    
    print("\nTraining on 200 samples (this will take a few minutes)...")
    model = train_value_mlp_on_mnist(
        n_train=200,
        n_test=50,
        hidden_size=32,
        epochs=5,
        lr=0.01,
        batch_size=10
    )
    
    print("\nDemo complete!")


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("VALUE-BASED MLP ON MNIST")
    print("=" * 70)
    print("\nThis script demonstrates training a scalar-based MLP using")
    print("the Value class for automatic differentiation.")
    print("\nOptions:")
    print("  python scripts/train_value_mlp_mnist.py quick  - Quick demo (50 samples)")
    print("  python scripts/train_value_mlp_mnist.py full   - Full demo (200 samples)")
    print("=" * 70)
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full_demo()
    else:
        quick_demo()
