"""
Multi-Layer Perceptron implementation from scratch.
Supports multiple layers, various activations, and backpropagation.
No external dependencies - all operations implemented with nested loops.

Author: Ensar Yesir
"""

import random
import math


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
    
    # Helper functions for matrix operations
    
    def _matrix_multiply(self, A, B):
        """Matrix multiplication: C = A @ B"""
        rows_A = len(A)
        cols_A = len(A[0]) if rows_A > 0 else 0
        rows_B = len(B)
        cols_B = len(B[0]) if rows_B > 0 else 0
        
        if cols_A != rows_B:
            raise ValueError(f"Matrix dimensions don't match: ({rows_A}, {cols_A}) and ({rows_B}, {cols_B})")
        
        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        
        return result
    
    def _matrix_add(self, A, B):
        """Matrix addition: C = A + B (with broadcasting for bias)"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        # Handle broadcasting if B is 1xN (bias)
        if len(B) == 1:
            for i in range(rows):
                for j in range(cols):
                    result[i][j] = A[i][j] + B[0][j]
        else:
            for i in range(rows):
                for j in range(cols):
                    result[i][j] = A[i][j] + B[i][j]
        
        return result
    
    def _matrix_subtract(self, A, B):
        """Matrix subtraction: C = A - B"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] - B[i][j]
        
        return result
    
    def _matrix_transpose(self, A):
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
    
    def _matrix_scalar_multiply(self, A, scalar):
        """Multiply matrix by scalar: C = scalar * A"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] * scalar
        
        return result
    
    def _matrix_element_multiply(self, A, B):
        """Element-wise multiplication: C = A * B"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] * B[i][j]
        
        return result
    
    def _matrix_sum(self, A):
        """Sum all elements in matrix"""
        total = 0.0
        for row in A:
            for val in row:
                total += val
        return total
    
    def _matrix_sum_axis0(self, A):
        """Sum along axis 0 (column sums) - returns 1xN"""
        if not A or not A[0]:
            return [[]]
        
        cols = len(A[0])
        result = [[0.0 for _ in range(cols)]]
        
        for row in A:
            for j in range(cols):
                result[0][j] += row[j]
        
        return result
    
    def _matrix_argmax_axis1(self, A):
        """Find argmax along axis 1 (for predictions)"""
        result = []
        for row in A:
            if row:
                max_idx = 0
                max_val = row[0]
                for j in range(1, len(row)):
                    if row[j] > max_val:
                        max_val = row[j]
                        max_idx = j
                result.append(max_idx)
            else:
                result.append(0)
        return result
        
    def _initialize_parameters(self):
        """Initialize weights using He init for ReLU, Xavier for others."""
        for l in range(1, self.L + 1):
            rows = self.layer_dims[l-1]
            cols = self.layer_dims[l]
            
            if self.activation_funcs[l-1] == 'relu':
                # He init for ReLU
                scale = math.sqrt(2.0 / rows)
            else:
                # Xavier init for tanh/sigmoid
                scale = math.sqrt(1.0 / rows)
            
            # Initialize weights with random normal distribution
            weights = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    # Box-Muller transform for normal distribution
                    u1 = random.random()
                    u2 = random.random()
                    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                    row.append(z * scale)
                weights.append(row)
            
            self.parameters[f'W{l}'] = weights
            
            # Initialize biases to zeros
            self.parameters[f'b{l}'] = [[0.0 for _ in range(cols)]]
    
    # Activation functions
    
    def _relu(self, Z):
        """ReLU: max(0, z)"""
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                result[i][j] = max(0.0, Z[i][j])
        return result
    
    def _relu_backward(self, dA, Z):
        """ReLU derivative: 1 if z > 0 else 0"""
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                if Z[i][j] <= 0:
                    result[i][j] = 0.0
                else:
                    result[i][j] = dA[i][j]
        return result
    
    def _tanh(self, Z):
        """Tanh activation: range (-1, 1)"""
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                result[i][j] = math.tanh(Z[i][j])
        return result
    
    def _tanh_backward(self, dA, Z):
        """Tanh derivative: 1 - tanh^2(z)"""
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                tanh_val = math.tanh(Z[i][j])
                result[i][j] = dA[i][j] * (1.0 - tanh_val ** 2)
        return result
    
    def _sigmoid(self, Z):
        """Sigmoid activation: range (0, 1)"""
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                # Clip for stability
                z_val = max(-500, min(500, Z[i][j]))
                result[i][j] = 1.0 / (1.0 + math.exp(-z_val))
        return result
    
    def _sigmoid_backward(self, dA, Z):
        """Sigmoid derivative: σ(z) * (1 - σ(z))"""
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                z_val = max(-500, min(500, Z[i][j]))
                sig = 1.0 / (1.0 + math.exp(-z_val))
                result[i][j] = dA[i][j] * sig * (1.0 - sig)
        return result
    
    def _softmax(self, Z):
        """Softmax: converts scores to probabilities"""
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        
        for i in range(len(Z)):
            # Find max for numerical stability
            max_val = Z[i][0]
            for j in range(1, len(Z[0])):
                if Z[i][j] > max_val:
                    max_val = Z[i][j]
            
            # Compute exp(z - max)
            exp_vals = []
            exp_sum = 0.0
            for j in range(len(Z[0])):
                exp_val = math.exp(Z[i][j] - max_val)
                exp_vals.append(exp_val)
                exp_sum += exp_val
            
            # Normalize
            for j in range(len(Z[0])):
                result[i][j] = exp_vals[j] / exp_sum
        
        return result
    
    def _softmax_backward(self, dA, Z):
        """Softmax+CrossEntropy derivative: y_pred - y_true"""
        return dA
    
    def _linear(self, Z):
        """Linear activation: f(z) = z"""
        return [row[:] for row in Z]
    
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
            
            Z = self._matrix_add(self._matrix_multiply(A_prev, W), b)
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
        m = len(Y)
        
        # Cross-entropy with epsilon for numerical stability
        epsilon = 1e-8
        loss = 0.0
        
        for i in range(len(Y)):
            for j in range(len(Y[0])):
                # Clip values
                a_val = max(epsilon, min(1.0 - epsilon, A_final[i][j]))
                if Y[i][j] > 0:
                    loss -= Y[i][j] * math.log(a_val)
        
        loss /= m
        
        # Add L2 regularization if needed
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for l in range(1, self.L + 1):
                W = self.parameters[f'W{l}']
                for row in W:
                    for val in row:
                        l2_loss += val ** 2
            loss += (self.l2_lambda / (2 * m)) * l2_loss
        
        return loss
    
    def _backward_propagation(self, Y, caches):
        """Backpropagation: compute gradients using chain rule."""
        gradients = {}
        m = len(Y)
        L = len(caches)
        
        # Output layer gradient: dZ = A - Y
        cache_L = caches[L-1]
        A_L = self._activate(cache_L['Z'], cache_L['activation'])
        dZ = self._matrix_subtract(A_L, Y)
        
        # Iterate backwards through all layers
        for l in reversed(range(1, L + 1)):
            cache = caches[l-1]
            A_prev = cache['A_prev']
            W = cache['W']
            
            # Compute weight gradient: dW = (1/m) * A_prev^T @ dZ
            A_prev_T = self._matrix_transpose(A_prev)
            dW = self._matrix_scalar_multiply(
                self._matrix_multiply(A_prev_T, dZ), 
                1.0 / m
            )
            
            # Compute bias gradient: db = (1/m) * sum(dZ)
            db = self._matrix_scalar_multiply(
                self._matrix_sum_axis0(dZ),
                1.0 / m
            )
            
            # Add L2 regularization gradient if enabled
            if self.l2_lambda > 0:
                l2_grad = self._matrix_scalar_multiply(W, self.l2_lambda / m)
                # Add l2_grad to dW
                for i in range(len(dW)):
                    for j in range(len(dW[0])):
                        dW[i][j] += l2_grad[i][j]
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            # Compute gradient for previous layer
            if l > 1:
                W_T = self._matrix_transpose(W)
                dA_prev = self._matrix_multiply(dZ, W_T)
                cache_prev = caches[l-2]
                dZ = self._activate_backward(dA_prev, cache_prev['Z'], 
                                            cache_prev['activation'])
        
        return gradients
    
    def _update_parameters(self, gradients):
        """Update parameters using gradient descent: θ = θ - α * dθ"""
        for l in range(1, self.L + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            dW = gradients[f'dW{l}']
            db = gradients[f'db{l}']
            
            # Update weights
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j] -= self.learning_rate * dW[i][j]
            
            # Update biases
            for j in range(len(b[0])):
                b[0][j] -= self.learning_rate * db[0][j]
    
    def predict(self, X):
        """Make predictions. Returns class labels (not probabilities)."""
        A_final, _ = self._forward_propagation(X)
        predictions = self._matrix_argmax_axis1(A_final)
        return predictions
    
    def fit(self, X, y, epochs=100, batch_size=32):
        """Train using mini-batch gradient descent. Yields (epoch, loss, self) for each epoch."""
        n_samples = len(X)
        
        # Convert labels to one-hot encoding
        n_classes = self.layer_dims[-1]
        Y = [[0.0 for _ in range(n_classes)] for _ in range(n_samples)]
        for i in range(n_samples):
            Y[i][int(y[i])] = 1.0
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            X_shuffled = [X[i] for i in indices]
            Y_shuffled = [Y[i] for i in indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # Forward propagation
                A_final, caches = self._forward_propagation(X_batch)
                
                # Compute loss
                batch_loss = self._compute_loss(A_final, Y_batch)
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward propagation
                gradients = self._backward_propagation(Y_batch, caches)
                
                # Update parameters
                self._update_parameters(gradients)
            
            avg_loss = epoch_loss / n_batches
            yield epoch + 1, avg_loss, self
