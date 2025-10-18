"""
Multi-Layer Perceptron implementation from scratch.
Supports multiple layers, various activations, and backpropagation.

Author: Ensar Yesir
"""

import numpy as np


class MLP:
    """MLP with backpropagation. Supports multiple layers and various activations."""

    def __init__(self, layer_dims, activation_funcs, learning_rate=0.01, l2_lambda=0.0):
        self.layer_dims = layer_dims
        self.activation_funcs = activation_funcs
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.parameters = {}
        self.L = len(layer_dims) - 1
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize weights using He init for ReLU, Xavier for others."""
        for l in range(1, self.L + 1):
            if self.activation_funcs[l-1] == 'relu':
                # He init for ReLU
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l-1], self.layer_dims[l]
                ) * np.sqrt(2.0 / self.layer_dims[l-1])
            else:
                # Xavier init for tanh/sigmoid
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l-1], self.layer_dims[l]
                ) * np.sqrt(1.0 / self.layer_dims[l-1])
            
            self.parameters[f'b{l}'] = np.zeros((1, self.layer_dims[l]))
    
    # Activation functions
    
    def _relu(self, Z):
        """ReLU: max(0, z)"""
        return np.maximum(0, Z)
    
    def _relu_backward(self, dA, Z):
        """ReLU derivative: 1 if z > 0 else 0"""
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ
    
    def _tanh(self, Z):
        """Tanh activation: range (-1, 1)"""
        return np.tanh(Z)
    
    def _tanh_backward(self, dA, Z):
        """Tanh derivative: 1 - tanh^2(z)"""
        A = np.tanh(Z)
        dZ = dA * (1 - A ** 2)
        return dZ
    
    def _sigmoid(self, Z):
        """Sigmoid activation: range (0, 1)"""
        # clip for stability
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def _sigmoid_backward(self, dA, Z):
        """Sigmoid derivative: σ(z) * (1 - σ(z))"""
        A = self._sigmoid(Z)
        dZ = dA * A * (1 - A)
        return dZ
    
    def _softmax(self, Z):
        """Softmax: converts scores to probabilities"""
        # subtract max for numerical stability
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def _softmax_backward(self, dA, Z):
        """Softmax+CrossEntropy derivative: y_pred - y_true"""
        return dA
    
    def _linear(self, Z):
        """Linear activation: f(z) = z"""
        return Z
    
    def _linear_backward(self, dA, Z):
        """Linear derivative: 1"""
        return dA
    
    def _activate(self, Z, activation):
        if activation == 'relu':
            return self._relu(Z)
        elif activation == 'tanh':
            return self._tanh(Z)
        elif activation == 'sigmoid':
            return self._sigmoid(Z)
        elif activation == 'softmax':
            return self._softmax(Z)
        elif activation == 'linear':
            return self._linear(Z)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _activate_backward(self, dA, Z, activation):
        if activation == 'relu':
            return self._relu_backward(dA, Z)
        elif activation == 'tanh':
            return self._tanh_backward(dA, Z)
        elif activation == 'sigmoid':
            return self._sigmoid_backward(dA, Z)
        elif activation == 'softmax':
            return self._softmax_backward(dA, Z)
        elif activation == 'linear':
            return self._linear_backward(dA, Z)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    # Forward and Backward Propagation
    
    def _forward_propagation(self, X):
        """Forward pass through all layers. Returns output and caches."""
        caches = []
        A = X
        
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(A_prev, W) + b
            activation = self.activation_funcs[l-1]
            A = self._activate(Z, activation)
            
            cache = {
                'A_prev': A_prev,
                'W': W,
                'b': b,
                'Z': Z,
                'activation': activation
            }
            caches.append(cache)
        
        return A, caches
    
    def _compute_loss(self, A_final, Y):
        """Compute cross-entropy loss with optional L2 regularization."""
        m = Y.shape[0]
        
        # cross-entropy with epsilon for numerical stability
        epsilon = 1e-8
        A_final = np.clip(A_final, epsilon, 1 - epsilon)
        loss = -np.sum(Y * np.log(A_final)) / m
        
        # add L2 regularization if needed
        if self.l2_lambda > 0:
            l2_loss = 0
            for l in range(1, self.L + 1):
                W = self.parameters[f'W{l}']
                l2_loss += np.sum(W ** 2)
            loss += (self.l2_lambda / (2 * m)) * l2_loss
        
        return loss
    
    def _backward_propagation(self, Y, caches):
        """Backpropagation: compute gradients using chain rule."""
        gradients = {}
        m = Y.shape[0]
        L = len(caches)
        
        # output layer gradient: dZ = A - Y
        cache_L = caches[L-1]
        A_L = self._activate(cache_L['Z'], cache_L['activation'])
        dZ = A_L - Y
        
        # iterate backwards through all layers
        for l in reversed(range(1, L + 1)):
            cache = caches[l-1]
            A_prev = cache['A_prev']
            W = cache['W']
            
            # compute weight gradient: dW = (1/m) * A_prev^T @ dZ
            dW = np.dot(A_prev.T, dZ) / m
            
            # compute bias gradient: db = (1/m) * sum(dZ)
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # add L2 regularization gradient if enabled
            if self.l2_lambda > 0:
                dW += (self.l2_lambda / m) * W
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            # compute gradient for previous layer
            if l > 1:
                dA_prev = np.dot(dZ, W.T)
                cache_prev = caches[l-2]
                dZ = self._activate_backward(dA_prev, cache_prev['Z'], 
                                            cache_prev['activation'])
        
        return gradients
    
    def _update_parameters(self, gradients):
        """Update parameters using gradient descent: θ = θ - α * dθ"""
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * gradients[f'db{l}']
    
    def predict(self, X):
        """Make predictions. Returns class labels (not probabilities)."""
        A_final, _ = self._forward_propagation(X)
        predictions = np.argmax(A_final, axis=1)
        return predictions
    
    def fit(self, X, y, epochs=100, batch_size=32):
        """Train using mini-batch gradient descent. Yields (epoch, loss, self) for each epoch."""
        n_samples = X.shape[0]
        
        # convert labels to one-hot encoding
        n_classes = self.layer_dims[-1]
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y.astype(int)] = 1
        
        # training loop
        for epoch in range(epochs):
            # shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # forward propagation
                A_final, caches = self._forward_propagation(X_batch)
                
                # compute loss
                batch_loss = self._compute_loss(A_final, Y_batch)
                epoch_loss += batch_loss
                n_batches += 1
                
                # backward propagation
                gradients = self._backward_propagation(Y_batch, caches)
                
                # update parameters
                self._update_parameters(gradients)
            
            avg_loss = epoch_loss / n_batches
            yield epoch + 1, avg_loss, self
