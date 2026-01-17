#!/usr/bin/env python3
"""
MNIST Dataset Downloader and Preprocessor
==========================================

This script downloads the MNIST handwritten digits dataset and saves it
in NumPy format for use with the deeplearning library.

MNIST Dataset Details:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 classes (digits 0-9)

The script downloads from the official MNIST source (Yann LeCun's website)
and preprocesses the data for neural network training.

Usage:
    python scripts/download_mnist.py

Output:
    data/mnist/
        ├── train_images.npy    (60000, 784) float32
        ├── train_labels.npy    (60000,) int64
        ├── test_images.npy     (10000, 784) float32
        └── test_labels.npy     (10000,) int64

Author: Deep Learning Assignment
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# MNIST URLs and expected checksums
MNIST_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

# MD5 checksums for verification
CHECKSUMS = {
    "train_images": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train_labels": "d53e105ee54ea40749a09fcbcd1e9432",
    "test_images": "9fb629c4189551a2d022fa330f9573f3",
    "test_labels": "ec29112dd5afa0611ce80d1b7f02629c",
}


def download_file(url: str, filepath: Path, checksum: Optional[str] = None) -> None:
    """
    Download a file from URL with progress indicator.

    Parameters
    ----------
    url : str
        URL to download from
    filepath : Path
        Local path to save the file
    checksum : str, optional
        Expected MD5 checksum for verification
    """
    if filepath.exists():
        if checksum:
            with open(filepath, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash == checksum:
                print(f"  ✓ {filepath.name} already exists (checksum verified)")
                return
            else:
                print(f"  ! {filepath.name} exists but checksum mismatch, re-downloading")
        else:
            print(f"  ✓ {filepath.name} already exists")
            return

    print(f"  Downloading {filepath.name}...")

    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        # Try alternate mirror
        alt_url = url.replace("http://yann.lecun.com/exdb/mnist/",
                              "https://ossci-datasets.s3.amazonaws.com/mnist/")
        print(f"  Primary source failed, trying alternate mirror...")
        try:
            urllib.request.urlretrieve(alt_url, filepath)
        except Exception as e2:
            raise RuntimeError(f"Failed to download from both sources: {e}, {e2}")

    if checksum:
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != checksum:
            raise ValueError(f"Checksum mismatch for {filepath.name}")

    print(f"  ✓ Downloaded {filepath.name}")


def read_idx_images(filepath: Path) -> np.ndarray:
    """
    Read IDX image file format.

    The IDX file format is:
    - 4 bytes: magic number (2051 for images)
    - 4 bytes: number of images
    - 4 bytes: number of rows
    - 4 bytes: number of columns
    - N*rows*cols bytes: pixel data (unsigned bytes)

    Parameters
    ----------
    filepath : Path
        Path to the gzipped IDX file

    Returns
    -------
    np.ndarray
        Image array of shape (N, rows*cols) with float32 dtype, normalized to [0, 1]
    """
    with gzip.open(filepath, "rb") as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}, expected 2051")

        # Read pixel data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)

        # Normalize to [0, 1] and convert to float32
        return data.astype(np.float32) / 255.0


def read_idx_labels(filepath: Path) -> np.ndarray:
    """
    Read IDX label file format.

    The IDX file format is:
    - 4 bytes: magic number (2049 for labels)
    - 4 bytes: number of labels
    - N bytes: label data (unsigned bytes)

    Parameters
    ----------
    filepath : Path
        Path to the gzipped IDX file

    Returns
    -------
    np.ndarray
        Label array of shape (N,) with int64 dtype
    """
    with gzip.open(filepath, "rb") as f:
        # Read header
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}, expected 2049")

        # Read label data
        data = np.frombuffer(f.read(), dtype=np.uint8)

        return data.astype(np.int64)


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize data to zero mean and unit variance.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    X_test : np.ndarray
        Test data

    Returns
    -------
    tuple
        Standardized (X_train, X_test)

    Notes
    -----
    Statistics are computed on training data only to prevent data leakage.
    """
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    return X_train, X_test


def main():
    """Download and preprocess MNIST dataset."""
    print("=" * 60)
    print("MNIST Dataset Downloader")
    print("=" * 60)
    print()

    # Create directories
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "mnist" / "raw"
    processed_dir = project_root / "data" / "mnist"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Download raw files
    print("Step 1: Downloading raw files...")
    for name, url in MNIST_URLS.items():
        filename = url.split("/")[-1]
        filepath = raw_dir / filename
        download_file(url, filepath, CHECKSUMS.get(name))
    print()

    # Parse files
    print("Step 2: Parsing IDX files...")
    train_images = read_idx_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    test_images = read_idx_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")

    print(f"  Training images: {train_images.shape}")
    print(f"  Training labels: {train_labels.shape}")
    print(f"  Test images: {test_images.shape}")
    print(f"  Test labels: {test_labels.shape}")
    print()

    # Standardize
    print("Step 3: Standardizing data...")
    train_images, test_images = standardize(train_images, test_images)
    print(f"  Train mean: {train_images.mean():.6f}, std: {train_images.std():.6f}")
    print(f"  Test mean: {test_images.mean():.6f}, std: {test_images.std():.6f}")
    print()

    # Save processed files
    print("Step 4: Saving processed files...")
    np.save(processed_dir / "train_images.npy", train_images)
    np.save(processed_dir / "train_labels.npy", train_labels)
    np.save(processed_dir / "test_images.npy", test_images)
    np.save(processed_dir / "test_labels.npy", test_labels)

    print(f"  ✓ Saved train_images.npy ({train_images.nbytes / 1e6:.1f} MB)")
    print(f"  ✓ Saved train_labels.npy ({train_labels.nbytes / 1e3:.1f} KB)")
    print(f"  ✓ Saved test_images.npy ({test_images.nbytes / 1e6:.1f} MB)")
    print(f"  ✓ Saved test_labels.npy ({test_labels.nbytes / 1e3:.1f} KB)")
    print()

    # Summary
    print("=" * 60)
    print("Download complete!")
    print("=" * 60)
    print()
    print("Dataset summary:")
    print(f"  Training samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  Image dimensions: {train_images.shape[1]} (28x28 flattened)")
    print(f"  Number of classes: {len(np.unique(train_labels))}")
    print()
    print("Label distribution (training):")
    for digit in range(10):
        count = np.sum(train_labels == digit)
        print(f"  Digit {digit}: {count} samples ({count/len(train_labels)*100:.1f}%)")
    print()
    print(f"Files saved to: {processed_dir}")


if __name__ == "__main__":
    main()
