import struct
import numpy as np
import sys
from os.path import dirname, join, abspath
from typing import Tuple

def _project_root() -> str:
    """Get project root - works for both script and exe"""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        return dirname(sys.executable)
    else:
        # Running as script
        return dirname(dirname(abspath(__file__)))

def _mnist_base_dir() -> str:
    return join(_project_root(), "dataset", "MNIST")

def _read_idx_images_labels(
    images_filepath: str,
    labels_filepath: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with open(labels_filepath, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic} (expected 2049)")
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    with open(images_filepath, "rb") as f:
        magic, size_img, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic} (expected 2051)")
        
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        
    images = image_data.reshape(size_img, rows * cols).astype(np.float32) / 255.0

    return images, labels

def load_mnist_dataset(
    limit_train: int | None = None,
    limit_test: int | None = None,
    per_class_train: int | None = None,
    per_class_test: int | None = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    
    base_dir = _mnist_base_dir()

    train_images = join(base_dir, "train-images-idx3-ubyte", "train-images-idx3-ubyte")
    train_labels = join(base_dir, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte")
    test_images = join(base_dir, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte")
    test_labels = join(base_dir, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte")

    X_train, y_train = _read_idx_images_labels(train_images, train_labels)
    X_test, y_test = _read_idx_images_labels(test_images, test_labels)

    def subsample_balanced(X, y, limit_per_class):
        if limit_per_class is None:
            return X, y
        
        indices = []
        counts = {d: 0 for d in range(10)}
        
        for i, label in enumerate(y):
            if counts[label] < limit_per_class:
                indices.append(i)
                counts[label] += 1
            if all(c >= limit_per_class for c in counts.values()):
                break
                
        return X[indices], y[indices]

    def subsample_limit(X, y, limit):
        if limit is None:
            return X, y
        return X[:limit], y[:limit]

    if per_class_train is not None:
        X_train, y_train = subsample_balanced(X_train, y_train, per_class_train)
    else:
        X_train, y_train = subsample_limit(X_train, y_train, limit_train)
        
    if per_class_test is not None:
        X_test, y_test = subsample_balanced(X_test, y_test, per_class_test)
    else:
        X_test, y_test = subsample_limit(X_test, y_test, limit_test)

    return (X_train, y_train), (X_test, y_test)
