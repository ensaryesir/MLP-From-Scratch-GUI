"""
Data management for the neural network visualizer.
Handles training points, class labels, and train/test splitting.
"""

import random

from config import COLOR_PALETTE


class DataHandler:
    """
    Manages training data points and their class labels.
    Stores points as (x, y, class_id) tuples and provides train/test splitting.
    """

    def __init__(self):
        self.data_points = []  # list of (x, y, class_id) tuples
        self.classes = []
        # color palette for visualization
        self.colors = COLOR_PALETTE

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
        """Get all points for a specific class as list of lists."""
        points = [p for p in self.data_points if p[2] == class_id]
        if len(points) == 0:
            return []
        return [[p[0], p[1]] for p in points]
    
    def get_all_points(self):
        """Get all data points regardless of class (for regression visualization)."""
        if len(self.data_points) == 0:
            return []
        return [[p[0], p[1]] for p in self.data_points]

    def get_all_data(self, task='classification'):
        """
        Returns (X, y) as lists.
        
        Args:
            task: 'classification' or 'regression'
        
        Returns:
            For classification:
                X = [[x, y], ...] (both coordinates)
                y = [class_id, ...] (class labels)
            
            For regression:
                X = [[x], ...] (only x coordinate)
                y = [y_pos, ...] (y coordinate as target value)
        """
        if len(self.data_points) == 0:
            return [], []

        if task == 'regression':
            # Regression: Predict Y coordinate from X coordinate
            X = [[p[0]] for p in self.data_points]  # Only X feature
            y = [p[1] for p in self.data_points]     # Y position as target
        else:
            # Classification: Use both coordinates, predict class
            X = [[p[0], p[1]] for p in self.data_points]
            y = [p[2] for p in self.data_points]  # Class ID
        
        return X, y

    def get_train_test_split(self, test_ratio=0.2, random_state=42, task='classification'):
        """
        Randomly split data into train and test sets.
        
        Args:
            test_ratio: Fraction of data for testing
            random_state: Random seed for reproducibility
            task: 'classification' or 'regression'
        """
        if len(self.data_points) == 0:
            return [], [], [], []

        X, y = self.get_all_data(task=task)  # Pass task parameter

        # Shuffle using random permutation
        random.seed(random_state)
        indices = list(range(len(X)))
        random.shuffle(indices)

        test_size = int(len(X) * test_ratio)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        return X_train, X_test, y_train, y_test

    def get_color(self, class_id):
        """Get the color for a class (cycles if more classes than colors)."""
        return self.colors[class_id % len(self.colors)]

    def get_num_classes(self):
        return len(self.classes)

    def get_num_points(self):
        return len(self.data_points)
