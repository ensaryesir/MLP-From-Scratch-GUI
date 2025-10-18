"""
Data management for the neural network visualizer.
Handles training points, class labels, and train/test splitting.
"""

import numpy as np


class DataHandler:
    """
    Manages training data points and their class labels.
    Stores points as (x, y, class_id) tuples and provides train/test splitting.
    """

    def __init__(self):
        self.data_points = []  # list of (x, y, class_id) tuples
        self.classes = []
        # color palette for visualization
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    def add_class(self, class_name=None):
        """Add a new class, auto-generating name if not provided."""
        if class_name is None:
            class_name = f"Class {len(self.classes)}"
        if class_name not in self.classes:
            self.classes.append(class_name)

    def remove_class(self):
        """Remove the last class and all its data points."""
        if len(self.classes) > 0:
            class_id = len(self.classes) - 1
            self.data_points = [p for p in self.data_points if p[2] != class_id]
            self.classes.pop()

    def add_point(self, x, y, class_id):
        """Add a training point if class_id is valid."""
        if 0 <= class_id < len(self.classes):
            self.data_points.append((x, y, class_id))

    def clear_data(self):
        """Clear all data points (keeps class definitions)."""
        self.data_points = []

    def get_data_by_class(self, class_id):
        """Get all points for a specific class as numpy array."""
        points = [p for p in self.data_points if p[2] == class_id]
        if len(points) == 0:
            return np.array([]).reshape(0, 2)
        return np.array([(p[0], p[1]) for p in points])

    def get_all_data(self):
        """Returns (X, y) as numpy arrays."""
        if len(self.data_points) == 0:
            return np.array([]).reshape(0, 2), np.array([])

        X = np.array([(p[0], p[1]) for p in self.data_points])
        y = np.array([p[2] for p in self.data_points])
        return X, y

    def get_train_test_split(self, test_ratio=0.2, random_state=42):
        """Randomly split data into train and test sets."""
        if len(self.data_points) == 0:
            empty = np.array([]).reshape(0, 2)
            return empty, empty, np.array([]), np.array([])

        X, y = self.get_all_data()

        # shuffle using random permutation
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))

        test_size = int(len(X) * test_ratio)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def get_color(self, class_id):
        """Get the color for a class (cycles if more classes than colors)."""
        return self.colors[class_id % len(self.colors)]

    def get_num_classes(self):
        return len(self.classes)

    def get_num_points(self):
        return len(self.data_points)
