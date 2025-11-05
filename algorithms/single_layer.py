"""
Single-layer neural network algorithms.
Contains Perceptron (1958) and Delta Rule/ADALINE (1960).
No external dependencies - all operations implemented with nested loops.

Author: Ensar Yesir
"""

import random
import math


class MatrixOperations:
    """
    Utility class for manual matrix operations without NumPy.
    Provides basic linear algebra operations using nested loops.
    """
    
    @staticmethod
    def multiply(A, B):
        """Matrix multiplication: C = A @ B"""
        rows_A = len(A)
        cols_A = len(A[0]) if rows_A > 0 else 0
        rows_B = len(B)
        cols_B = len(B[0]) if rows_B > 0 else 0
        
        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        
        return result
    
    @staticmethod
    def add(A, B):
        """Matrix addition with broadcasting support"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        if len(B) == 1:  # Broadcasting: B is 1xN (bias vector)
            for i in range(rows):
                for j in range(cols):
                    result[i][j] = A[i][j] + B[0][j]
        else:  # Normal element-wise addition
            for i in range(rows):
                for j in range(cols):
                    result[i][j] = A[i][j] + B[i][j]
        
        return result
    
    @staticmethod
    def subtract(A, B):
        """Matrix subtraction: C = A - B"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] - B[i][j]
        
        return result
    
    @staticmethod
    def transpose(A):
        """Matrix transpose: A^T"""
        if not A or not A[0]:
            return [[]]
        rows = len(A)
        cols = len(A[0])
        result = [[0.0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                result[j][i] = A[i][j]
        return result
    
    @staticmethod
    def scalar_multiply(A, scalar):
        """Multiply matrix by scalar: C = scalar * A"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] * scalar
        
        return result


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
        self.matrix_ops = MatrixOperations()  # Matrix operations helper

    def _initialize_parameters(self, n_features):
        # One weight vector per class - random small values
        self.weights = []
        for i in range(n_features):
            row = []
            for j in range(self.n_classes):
                # Box-Muller transform for normal distribution
                u1 = random.random()
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                row.append(z * 0.01)
            self.weights.append(row)
        
        self.bias = [[0.0 for _ in range(self.n_classes)]]

    def predict(self, X):
        if self.weights is None:
            return [0 for _ in range(len(X))]

        # z = X × W + b (linear combination)
        z = self.matrix_ops.add(self.matrix_ops.multiply(X, self.weights), self.bias)
        
        # Multi-class: pick highest activation
        predictions = []
        for row in z:
            max_idx = 0
            max_val = row[0]
            for j in range(1, len(row)):
                if row[j] > max_val:
                    max_val = row[j]
                    max_idx = j
            predictions.append(max_idx)
        
        return predictions

    def fit(self, X, y, epochs=100):
        """Train using Perceptron learning algorithm (online learning)."""
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        # One-hot encoding
        y_onehot = [[0.0 for _ in range(self.n_classes)] for _ in range(n_samples)]
        for i in range(n_samples):
            y_onehot[i][int(y[i])] = 1.0
        
        for epoch in range(epochs):
            total_error = 0.0
            
            # Online learning: update after each sample
            for i in range(n_samples):
                xi = [X[i]]
                yi = [y_onehot[i]]
                
                # Forward pass
                z = self.matrix_ops.add(self.matrix_ops.multiply(xi, self.weights), self.bias)
                
                # Get prediction (one-hot)
                prediction_onehot = [[0.0 for _ in range(self.n_classes)]]
                max_idx = 0
                max_val = z[0][0]
                for j in range(1, len(z[0])):
                    if z[0][j] > max_val:
                        max_val = z[0][j]
                        max_idx = j
                prediction_onehot[0][max_idx] = 1.0
                
                # Compute error
                error = [[0.0 for _ in range(self.n_classes)]]
                for j in range(self.n_classes):
                    error[0][j] = yi[0][j] - prediction_onehot[0][j]
                    total_error += abs(error[0][j])
                
                # Update weights: Δw = η * error * x
                xi_T = self.matrix_ops.transpose(xi)
                weight_update = self.matrix_ops.multiply(xi_T, error)
                
                for ii in range(len(self.weights)):
                    for jj in range(len(self.weights[0])):
                        self.weights[ii][jj] += self.learning_rate * weight_update[ii][jj]
                
                for jj in range(len(self.bias[0])):
                    self.bias[0][jj] += self.learning_rate * error[0][jj]
            
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
        self.matrix_ops = MatrixOperations()  # Matrix operations helper
        
    def _initialize_parameters(self, n_features):
        self.weights = []
        for i in range(n_features):
            row = []
            for j in range(self.n_classes):
                # Box-Muller transform for normal distribution
                u1 = random.random()
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                row.append(z * 0.01)
            self.weights.append(row)
        
        self.bias = [[0.0 for _ in range(self.n_classes)]]
        
    def _activation(self, x):
        """Linear activation: f(x) = x"""
        return [row[:] for row in x]
    
    def predict(self, X):
        if self.weights is None:
            return [0 for _ in range(len(X))]
        
        z = self.matrix_ops.add(self.matrix_ops.multiply(X, self.weights), self.bias)
        a = self._activation(z)
        
        # Pick class with highest output
        predictions = []
        for row in a:
            max_idx = 0
            max_val = row[0]
            for j in range(1, len(row)):
                if row[j] > max_val:
                    max_val = row[j]
                    max_idx = j
            predictions.append(max_idx)
        
        return predictions
    
    def fit(self, X, y, epochs=100):
        """Train using batch gradient descent to minimize MSE."""
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        # One-hot encoding
        y_onehot = [[0.0 for _ in range(self.n_classes)] for _ in range(n_samples)]
        for i in range(n_samples):
            y_onehot[i][int(y[i])] = 1.0
        
        for epoch in range(epochs):
            # Forward pass (batch)
            z = self.matrix_ops.add(self.matrix_ops.multiply(X, self.weights), self.bias)
            a = self._activation(z)
            
            # Compute error: e = y - a
            error = self.matrix_ops.subtract(y_onehot, a)
            
            # Compute mean squared error for monitoring
            loss = 0.0
            for i in range(len(error)):
                for j in range(len(error[0])):
                    loss += error[i][j] ** 2
            loss /= (len(error) * len(error[0]))
            
            # Compute weight updates
            # Delta Rule: Δw = η * (y - a) * x
            # For MSE loss L = (1/2n)Σ(y-a)², gradient is: ∂L/∂w = -(1/n)X^T@(y-a)
            # Gradient descent: w = w - η*∂L/∂w = w + η*(1/n)X^T@(y-a)
            X_T = self.matrix_ops.transpose(X)
            weight_gradient = self.matrix_ops.scalar_multiply(
                self.matrix_ops.multiply(X_T, error),
                1.0 / n_samples  # Normalized by batch size
            )
            
            # Compute bias updates: Δb = (1/n) * sum(y - a)
            bias_gradient = [[0.0 for _ in range(self.n_classes)]]
            for i in range(len(error)):
                for j in range(len(error[0])):
                    bias_gradient[0][j] += error[i][j]
            for j in range(len(bias_gradient[0])):
                bias_gradient[0][j] /= n_samples
            
            # Update parameters: w_new = w_old + η * gradient
            # This implements: w = w - η*∂L/∂w = w + η*(1/n)*X^T@error
            for i in range(len(self.weights)):
                for j in range(len(self.weights[0])):
                    self.weights[i][j] += self.learning_rate * weight_gradient[i][j]
            
            for j in range(len(self.bias[0])):
                self.bias[0][j] += self.learning_rate * bias_gradient[0][j]
            
            yield epoch + 1, loss, self
