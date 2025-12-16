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

    def _add_random_points(self, center_x, center_y, class_id, count=20, spread=1.0):
        """Helper to add gaussian distributed points around a center."""
        for _ in range(count):
            x = random.gauss(center_x, spread)
            y = random.gauss(center_y, spread)
            # Clip to 0-10 range roughly
            x = max(0.1, min(9.9, x))
            y = max(0.1, min(9.9, y))
            self.add_point(x, y, class_id)

    def generate_xor(self, n_samples=200):
        """Generate XOR dataset (Classification)."""
        self.clear_data()
        self.classes = ["Red", "Green"]
        
        # Red: Top-Left (2, 8) and Bottom-Right (8, 2)
        count = n_samples // 4
        self._add_random_points(2, 8, 0, count, 0.8)
        self._add_random_points(8, 2, 0, count, 0.8)
        
        # Green: Top-Right (8, 8) and Bottom-Left (2, 2)
        self._add_random_points(8, 8, 1, count, 0.8)
        self._add_random_points(2, 2, 1, count, 0.8)

    def generate_circles(self, n_samples=200):
        """Generate Concentric Circles dataset (Classification)."""
        self.clear_data()
        self.classes = ["Inner", "Outer"]
        import math
        
        center_x, center_y = 5.0, 5.0
        
        # Inner Circle (Radius ~2)
        for _ in range(n_samples // 2):
            angle = random.uniform(0, 2 * math.pi)
            r = random.gauss(2.0, 0.3)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            self.add_point(x, y, 0)
            
        # Outer Ring (Radius ~4)
        for _ in range(n_samples // 2):
            angle = random.uniform(0, 2 * math.pi)
            r = random.gauss(4.0, 0.3)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            self.add_point(x, y, 1)

    def generate_moons(self, n_samples=200):
        """Generate Two Moons dataset (Classification)."""
        self.clear_data()
        self.classes = ["Upper", "Lower"]
        import math
        
        # Upper Moon
        for _ in range(n_samples // 2):
            angle = random.uniform(0, math.pi)
            r = random.gauss(2.5, 0.1)
            x = 3.5 + r * math.cos(angle)
            y = 4.0 + r * math.sin(angle)
            self.add_point(x, y, 0)
            
        # Lower Moon
        for _ in range(n_samples // 2):
            angle = random.uniform(math.pi, 2*math.pi)
            r = random.gauss(2.5, 0.1)
            x = 6.5 + r * math.cos(angle)
            y = 6.0 + r * math.sin(angle)
            self.add_point(x, y, 1)

    def generate_blobs(self, n_samples=300):
        """Generate Multi-class Blobs (Classification)."""
        self.clear_data()
        self.classes = [f"C{i}" for i in range(9)]
        centers = [
            (2, 8), (5, 8), (8, 8),
            (2, 5), (5, 5), (8, 5),
            (2, 2), (5, 2), (8, 2)
        ]
        points_per_blob = n_samples // 9
        for i, (cx, cy) in enumerate(centers):
            self._add_random_points(cx, cy, i, points_per_blob, 0.4)

    # --- Regression Presets ---

    def generate_sine(self, n_samples=200):
        """Generate Sine Wave dataset (Regression)."""
        self.clear_data()
        self.classes = ["Output"] 
        import math
        for _ in range(n_samples):
            x = random.uniform(0, 10)
            # Sine wave: y = 5 + 3*sin(x) + noise
            noise = random.gauss(0, 0.5)
            y = 5 + 3 * math.sin(x) + noise
            y = max(0, min(10, y))
            self.add_point(x, y, 0)

    def generate_parabola(self, n_samples=200):
        """Generate Parabola dataset (Regression)."""
        self.clear_data()
        self.classes = ["Output"]
        for _ in range(n_samples):
            x = random.uniform(0, 10)
            # Parabola: y = 0.2*(x-5)^2 + 2 + noise
            noise = random.gauss(0, 0.5)
            y = 0.2 * (x - 5)**2 + 2 + noise
            y = max(0, min(10, y))
            self.add_point(x, y, 0)

    def generate_linear(self, n_samples=200):
        """Generate Linear dataset (Regression)."""
        self.clear_data()
        self.classes = ["Output"]
        for _ in range(n_samples):
            x = random.uniform(0, 10)
            # Linear: y = x + noise
            noise = random.gauss(0, 0.8)
            y = x + noise
            y = max(0, min(10, y))
            self.add_point(x, y, 0)

    def generate_abs(self, n_samples=200):
        """Generate Absolute Value dataset (Regression)."""
        self.clear_data()
        self.classes = ["Output"]
        for _ in range(n_samples):
            x = random.uniform(0, 10)
            # V-shape: y = |x-5| + 2
            noise = random.gauss(0, 0.5)
            y = abs(x - 5) + 2 + noise
            y = max(0, min(10, y))
            self.add_point(x, y, 0)
