"""
Single-layer neural network algorithms.
Contains Perceptron (1958) and Delta Rule/ADALINE (1960).

Author: Ensar Yesir
"""

import numpy as np


class Perceptron:
    """
    Classic Perceptron algorithm (Rosenblatt, 1958).
    Single neuron with step activation - only works for linearly separable data.
    Learning rule: Δw = η * (y_true - y_pred) * x
    """

    def __init__(self, learning_rate=0.01, n_classes=2):
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.weights = None
        self.bias = None

    def _initialize_parameters(self, n_features):
        # one weight vector per class
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))

    def _step_function(self, x):
        """Step activation: 1 if x >= 0 else 0"""
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        if self.weights is None:
            return np.zeros(X.shape[0])

        z = np.dot(X, self.weights) + self.bias
        # multi-class: pick highest activation
        predictions = np.argmax(z, axis=1)
        return predictions

    def fit(self, X, y, epochs=100):
        """Train using Perceptron learning algorithm (online learning)."""
        n_samples, n_features = X.shape
        
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        # one-hot encoding
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        
        for epoch in range(epochs):
            total_error = 0
            
            # online learning: update after each sample
            for i in range(n_samples):
                xi = X[i:i+1]
                yi = y_onehot[i:i+1]
                
                # forward pass
                z = np.dot(xi, self.weights) + self.bias
                prediction_onehot = np.zeros_like(z)
                prediction_onehot[0, np.argmax(z)] = 1
                
                # compute error
                error = yi - prediction_onehot
                total_error += np.sum(np.abs(error))
                
                # update weights: Δw = η * error * x
                self.weights += self.learning_rate * np.dot(xi.T, error)
                self.bias += self.learning_rate * error
            
            avg_error = total_error / (n_samples * self.n_classes)
            yield epoch + 1, avg_error, self


class DeltaRule:
    """
    Delta Rule/ADALINE (Widrow-Hoff, 1960).
    Uses linear activation and minimizes MSE with batch gradient descent.
    More stable than Perceptron but still limited to linear boundaries.
    """
    
    def __init__(self, learning_rate=0.01, n_classes=2):
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
        
    def _initialize_parameters(self, n_features):
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))
        
    def _activation(self, x):
        """Linear activation: f(x) = x"""
        return x
    
    def predict(self, X):
        if self.weights is None:
            return np.zeros(X.shape[0])
        
        z = np.dot(X, self.weights) + self.bias
        a = self._activation(z)
        # pick class with highest output
        predictions = np.argmax(a, axis=1)
        return predictions
    
    def fit(self, X, y, epochs=100):
        """Train using batch gradient descent to minimize MSE."""
        n_samples, n_features = X.shape
        
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        # one-hot encoding
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        
        for epoch in range(epochs):
            # forward pass (batch)
            z = np.dot(X, self.weights) + self.bias
            a = self._activation(z)
            
            # compute MSE loss
            error = y_onehot - a
            loss = np.mean(error ** 2)
            
            # compute gradients
            dW = -2 * np.dot(X.T, error) / n_samples
            db = -2 * np.mean(error, axis=0, keepdims=True)
            
            # update parameters
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            yield epoch + 1, loss, self
