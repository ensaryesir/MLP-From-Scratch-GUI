"""MNIST dataset loader and helpers for MLP training.

This module is independent from the GUI click-based data handler.
It reads the raw IDX files from the local dataset/MNIST directory and
returns flattened, normalized inputs suitable for the MLP implementation
in this project.
"""

import struct
from array import array
from os.path import dirname, join
from typing import List, Tuple


def _project_root() -> str:
    """Return absolute path to project root (folder containing this repo)."""
    # utils/ -> project root is one level above
    return dirname(dirname(__file__))


def _mnist_base_dir() -> str:
    """Return absolute path to dataset/MNIST directory."""
    return join(_project_root(), "dataset", "MNIST")


def _read_idx_images_labels(
    images_filepath: str,
    labels_filepath: str,
) -> Tuple[List[List[float]], List[int]]:
    """Read MNIST IDX image/label files and return flattened, normalized data.

    Returns
    -------
    images : List[List[float]]
        Each image is a list of 784 floats in [0, 1].
    labels : List[int]
        Integer class labels 0-9.
    """
    # Read labels
    with open(labels_filepath, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic} (expected 2049)")
        labels_raw = array("B", f.read())

    # Read images
    with open(images_filepath, "rb") as f:
        magic, size_img, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic} (expected 2051)")
        if size_img != len(labels_raw):
            # Not fatal, but helpful to know
            print(
                f"Warning: number of images ({size_img}) and labels ({len(labels_raw)}) differ."
            )
        image_data = array("B", f.read())

    images: List[List[float]] = []
    labels: List[int] = []
    pixels_per_image = rows * cols

    for i in range(len(labels_raw)):
        start = i * pixels_per_image
        end = start + pixels_per_image
        # Normalize to [0, 1]
        pixels = [p / 255.0 for p in image_data[start:end]]
        images.append(pixels)
        labels.append(int(labels_raw[i]))

    return images, labels


def load_mnist_dataset(
    limit_train: int | None = None,
    limit_test: int | None = None,
    per_class_train: int | None = None,
    per_class_test: int | None = None,
) -> Tuple[Tuple[List[List[float]], List[int]], Tuple[List[List[float]], List[int]]]:
    """Load MNIST train and test sets from dataset/MNIST.

    Parameters
    ----------
    limit_train : Optional[int]
        If provided, truncate training set to this many samples.
    limit_test : Optional[int]
        If provided, truncate test set to this many samples.
    per_class_train : Optional[int]
        If provided, take at most this many samples per digit (0-9) for training.
    per_class_test : Optional[int]
        If provided, take at most this many samples per digit (0-9) for test.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
        X_* are lists of flattened images (length 784),
        y_* are integer labels 0-9.
    """
    base_dir = _mnist_base_dir()

    train_images = join(base_dir, "train-images-idx3-ubyte", "train-images-idx3-ubyte")
    train_labels = join(base_dir, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte")
    test_images = join(base_dir, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte")
    test_labels = join(base_dir, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte")

    X_train, y_train = _read_idx_images_labels(train_images, train_labels)
    X_test, y_test = _read_idx_images_labels(test_images, test_labels)

    # Class-balanced subsampling if requested
    if per_class_train is not None:
        counts = {d: 0 for d in range(10)}
        X_bal: List[List[float]] = []
        y_bal: List[int] = []
        for x, y in zip(X_train, y_train):
            if counts.get(y, 0) < per_class_train:
                X_bal.append(x)
                y_bal.append(y)
                counts[y] = counts.get(y, 0) + 1
            # Early exit if all classes reached per_class_train
            if all(counts[d] >= per_class_train for d in range(10)):
                break
        X_train, y_train = X_bal, y_bal
    elif limit_train is not None:
        X_train = X_train[:limit_train]
        y_train = y_train[:limit_train]

    if per_class_test is not None:
        counts_t = {d: 0 for d in range(10)}
        X_bal_t: List[List[float]] = []
        y_bal_t: List[int] = []
        for x, y in zip(X_test, y_test):
            if counts_t.get(y, 0) < per_class_test:
                X_bal_t.append(x)
                y_bal_t.append(y)
                counts_t[y] = counts_t.get(y, 0) + 1
            if all(counts_t[d] >= per_class_test for d in range(10)):
                break
        X_test, y_test = X_bal_t, y_bal_t
    elif limit_test is not None:
        X_test = X_test[:limit_test]
        y_test = y_test[:limit_test]

    return (X_train, y_train), (X_test, y_test)
