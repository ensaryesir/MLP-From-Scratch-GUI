import random
import math
import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.01, n_classes=2, task='classification'):
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.task = task
        self.weights = None
        self.bias = None

    def _initialize_parameters(self, n_features):
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))

    def predict(self, X):
        if self.weights is None:
            return np.zeros(len(X))
        
        X = np.array(X)
        if hasattr(self, 'weights') and X.shape[1] != self.weights.shape[0]:
             pass

        z = np.dot(X, self.weights) + self.bias
        
        if self.task == 'regression':
            if self.n_classes == 1:
                return z.flatten()
            return z
        else:
            return np.argmax(z, axis=1)

    def fit(self, X, y, epochs=100, stop_callback=None):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        if self.task == 'regression':
            if y.ndim == 1:
                 targets = y.reshape(-1, 1)
            else:
                 targets = y
        else:
            targets = np.zeros((n_samples, self.n_classes))

            y_indices = y.astype(int)

            targets[np.arange(n_samples), y_indices.flatten()] = 1.0
        
        for epoch in range(epochs):
            if stop_callback and stop_callback():
                return

            total_error = 0.0
            
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                if stop_callback and stop_callback():
                    return
                    
                xi = X[i:i+1]
                target_i = targets[i:i+1]
                
                z = np.dot(xi, self.weights) + self.bias
                
                if self.task == 'regression':
                    output = z
                else:
                    output = np.zeros_like(z)
                    output[0, np.argmax(z)] = 1.0
                
                error = target_i - output
                
                abs_error = np.sum(np.abs(error))
                total_error += abs_error
                
                should_update = True
                if self.task == 'classification' and abs_error < 1e-9:
                    should_update = False
                
                if should_update:
                    self.weights += self.learning_rate * np.dot(xi.T, error)
                    self.bias += self.learning_rate * error
            
            avg_error = total_error / (n_samples * self.n_classes)
            yield epoch + 1, avg_error, self

class DeltaRule:
    
    def __init__(self, learning_rate=0.01, n_classes=2, task='classification'):
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.task = task
        self.weights = None
        self.bias = None
        
    def _initialize_parameters(self, n_features):
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))

    def predict(self, X):
        if self.weights is None:
            return np.zeros(len(X))
        
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        
        if self.task == 'regression':
            if self.n_classes == 1:
                return z.flatten()
            return z
        else:
            return np.argmax(z, axis=1)
    
    def fit(self, X, y, epochs=100, stop_callback=None):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        if self.task == 'regression':
             if y.ndim == 1:
                 targets = y.reshape(-1, 1)
             else:
                 targets = y
        else:
            targets = np.zeros((n_samples, self.n_classes))
            y_indices = y.astype(int)
            targets[np.arange(n_samples), y_indices.flatten()] = 1.0
        
        for epoch in range(epochs):
            if stop_callback and stop_callback():
                return
                
            z = np.dot(X, self.weights) + self.bias
            
            error = targets - z
            
            loss = np.mean(error ** 2)
            
            grad_W = np.dot(X.T, error) / n_samples
            grad_b = np.mean(error, axis=0, keepdims=True)
            
            self.weights += self.learning_rate * grad_W
            self.bias += self.learning_rate * grad_b
            
            yield epoch + 1, loss, self
