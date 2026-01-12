import random
import numpy as np

from config import COLOR_PALETTE


class DataHandler:
    def __init__(self):
        self.data_points = []
        self.classes = []
        self.colors = COLOR_PALETTE

    def add_class(self, class_name=None):
        if class_name is None:
            class_name = f"Class {len(self.classes)}"
        if class_name not in self.classes:
            self.classes.append(class_name)

    def remove_class(self):
        if len(self.classes) > 0:
            class_id = len(self.classes) - 1
            self.data_points = [p for p in self.data_points if p[2] != class_id]
            self.classes.pop()

    def add_point(self, x, y, class_id):
        if 0 <= class_id < len(self.classes):
            self.data_points.append((x, y, class_id))

    def clear_data(self):
        self.data_points = []

    def get_data_by_class(self, class_id):
        points = [p for p in self.data_points if p[2] == class_id]
        if len(points) == 0:
            return []
        return [[p[0], p[1]] for p in points]
    
    def get_all_points(self):
        if len(self.data_points) == 0:
            return []
        return [[p[0], p[1]] for p in self.data_points]

    def get_all_data(self, task='classification'):
        if len(self.data_points) == 0:
            return np.array([]), np.array([])

        if task == 'regression':
            X = np.array([[p[0]] for p in self.data_points], dtype=np.float32)
            y = np.array([[p[1]] for p in self.data_points], dtype=np.float32)
        else:
            X = np.array([[p[0], p[1]] for p in self.data_points], dtype=np.float32)
            y = np.array([p[2] for p in self.data_points], dtype=np.int32)
        
        return X, y

    def get_train_test_split(self, test_ratio=0.2, random_state=42, task='classification'):
        if len(self.data_points) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        X, y = self.get_all_data(task=task)

        indices = np.arange(len(X))
        
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

        test_size = int(len(X) * test_ratio)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        return X_train, X_test, y_train, y_test

    def get_color(self, class_id):
        return self.colors[class_id % len(self.colors)]

    def get_num_classes(self):
        return len(self.classes)

    def get_num_points(self):
        return len(self.data_points)

    def _add_random_points(self, center_x, center_y, class_id, count=20, spread=1.0):
        for _ in range(count):
            x = random.gauss(center_x, spread)
            y = random.gauss(center_y, spread)
            x = max(0.1, min(9.9, x))
            y = max(0.1, min(9.9, y))
            self.add_point(x, y, class_id)

    def generate_xor(self, n_samples=200):
        self.clear_data()
        self.classes = ["Red", "Green"]
        
        count = n_samples // 4
        self._add_random_points(2, 8, 0, count, 0.8)
        self._add_random_points(8, 2, 0, count, 0.8)
        
        self._add_random_points(8, 8, 1, count, 0.8)
        self._add_random_points(2, 2, 1, count, 0.8)

    def generate_circles(self, n_samples=200):
        self.clear_data()
        self.classes = ["Inner", "Outer"]
        import math
        
        center_x, center_y = 5.0, 5.0
        
        for _ in range(n_samples // 2):
            angle = random.uniform(0, 2 * math.pi)
            r = random.gauss(2.0, 0.3)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            self.add_point(x, y, 0)
            
        for _ in range(n_samples // 2):
            angle = random.uniform(0, 2 * math.pi)
            r = random.gauss(4.0, 0.3)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            self.add_point(x, y, 1)

    def generate_moons(self, n_samples=200):
        self.clear_data()
        self.classes = ["Upper", "Lower"]
        import math
        
        for _ in range(n_samples // 2):
            angle = random.uniform(0, math.pi)
            r = random.gauss(2.5, 0.1)
            x = 3.5 + r * math.cos(angle)
            y = 4.0 + r * math.sin(angle)
            self.add_point(x, y, 0)
            
        for _ in range(n_samples // 2):
            angle = random.uniform(math.pi, 2*math.pi)
            r = random.gauss(2.5, 0.1)
            x = 6.5 + r * math.cos(angle)
            y = 6.0 + r * math.sin(angle)
            self.add_point(x, y, 1)

    def generate_blobs(self, n_samples=300):
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

    def generate_sine(self, n_samples=200):
        self.clear_data()
        self.classes = ["Output"] 
        import math
        for _ in range(n_samples):
            x = random.uniform(0, 10)
            noise = random.gauss(0, 0.5)
            y = 5 + 3 * math.sin(x) + noise
            y = max(0, min(10, y))
            self.add_point(x, y, 0)

    def generate_parabola(self, n_samples=200):
        self.clear_data()
        self.classes = ["Output"]
        for _ in range(n_samples):
            x = random.uniform(0, 10)
            noise = random.gauss(0, 0.5)
            y = 0.2 * (x - 5)**2 + 2 + noise
            y = max(0, min(10, y))
            self.add_point(x, y, 0)

    def generate_linear(self, n_samples=200):
        self.clear_data()
        self.classes = ["Output"]
        for _ in range(n_samples):
            x = random.uniform(0, 10)
            noise = random.gauss(0, 0.8)
            y = x + noise
            y = max(0, min(10, y))
            self.add_point(x, y, 0)

    def generate_abs(self, n_samples=200):
        self.clear_data()
        self.classes = ["Output"]
        for _ in range(n_samples):
            x = random.uniform(0, 10)
            noise = random.gauss(0, 0.5)
            y = abs(x - 5) + 2 + noise
            y = max(0, min(10, y))
            self.add_point(x, y, 0)
