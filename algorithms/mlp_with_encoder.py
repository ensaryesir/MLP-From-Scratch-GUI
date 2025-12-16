"""
Hybrid MLP model with pre-trained encoder for feature extraction.
Combines frozen/trainable encoder with MLP classifier.
"""

import random
import math

from utils import ActivationFunctions, MatrixOperations


class MLPWithEncoder:
    """
    Hybrid model: Pre-trained Encoder + MLP Classifier
    
    Architecture:
        Input (784) → Encoder (frozen/trainable) → Latent (32) → MLP → Output (10)
    
    Example:
        encoder_params = trained_autoencoder.get_encoder_weights()
        model = MLPWithEncoder(
            encoder_params=encoder_params,
            encoder_dims=[784, 256, 128, 32],
            mlp_layer_dims=[32, 64, 10],
            freeze_encoder=True
        )
    """
    
    def __init__(self, encoder_params, encoder_dims, encoder_activations,
                 mlp_layer_dims, mlp_activations, learning_rate=0.01,
                 l2_lambda=0.0, freeze_encoder=True, use_momentum=False, 
                 momentum_factor=0.9):
        """
        Initialize hybrid model.
        
        Args:
            encoder_params: Dictionary of pre-trained encoder weights (W1, b1, ...)
            encoder_dims: Layer sizes for encoder (e.g., [784, 256, 128, 32])
            encoder_activations: Activation functions for encoder layers
            mlp_layer_dims: Layer sizes for MLP (e.g., [32, 64, 10])
            mlp_activations: Activation functions for MLP layers
            learning_rate: Learning rate
            l2_lambda: L2 regularization
            freeze_encoder: If True, encoder weights are not updated
            use_momentum: Whether to use momentum
            momentum_factor: Momentum factor
        """
        self.encoder_dims = encoder_dims
        self.encoder_activations = encoder_activations
        self.mlp_layer_dims = mlp_layer_dims
        self.mlp_activations = mlp_activations
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.freeze_encoder = freeze_encoder
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        
        self.encoder_layers = len(encoder_dims) - 1
        self.mlp_layers = len(mlp_layer_dims) - 1
        self.total_layers = self.encoder_layers + self.mlp_layers
        
        self.parameters = {}
        self.velocities = {}
        
        self.matrix_ops = MatrixOperations()
        self.activations = ActivationFunctions()
        
        # Load pre-trained encoder weights
        for l in range(1, self.encoder_layers + 1):
            self.parameters[f'W{l}'] = encoder_params[f'W{l}']
            self.parameters[f'b{l}'] = encoder_params[f'b{l}']
            
            if self.use_momentum and not self.freeze_encoder:
                rows = len(self.parameters[f'W{l}'])
                cols = len(self.parameters[f'W{l}'][0])
                self.velocities[f'dW{l}'] = [[0.0 for _ in range(cols)] for _ in range(rows)]
                self.velocities[f'db{l}'] = [[0.0 for _ in range(cols)]]
        
        # Initialize MLP layers
        self._initialize_mlp_layers()
    
    def _initialize_mlp_layers(self):
        """Initialize MLP layers (after encoder)."""
        for l in range(1, self.mlp_layers + 1):
            layer_idx = self.encoder_layers + l
            rows = self.mlp_layer_dims[l-1]
            cols = self.mlp_layer_dims[l]
            
            # Determine initialization scale
            if self.mlp_activations[l-1] == 'relu':
                scale = math.sqrt(2.0 / rows)  # He init
            else:
                scale = math.sqrt(1.0 / rows)  # Xavier init
            
            # Initialize weights
            weights = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    u1 = random.random()
                    u2 = random.random()
                    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                    row.append(z * scale)
                weights.append(row)
            
            self.parameters[f'W{layer_idx}'] = weights
            self.parameters[f'b{layer_idx}'] = [[0.0 for _ in range(cols)]]
            
            # Initialize velocities
            if self.use_momentum:
                self.velocities[f'dW{layer_idx}'] = [[0.0 for _ in range(cols)] for _ in range(rows)]
                self.velocities[f'db{layer_idx}'] = [[0.0 for _ in range(cols)]]
    
    def _activate(self, Z, activation):
        """Apply activation function."""
        return self.activations.apply(Z, activation)
    
    def _activate_backward(self, dA, Z, activation):
        """Apply activation derivative."""
        return self.activations.apply_backward(dA, Z, activation)
    
    def _forward_propagation(self, X):
        """Forward pass through encoder + MLP."""
        caches = []
        A = X
        
        # Encoder forward pass
        for l in range(1, self.encoder_layers + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = self.matrix_ops.add(self.matrix_ops.multiply(A_prev, W), b)
            activation = self.encoder_activations[l-1]
            A = self._activate(Z, activation)
            
            cache = {
                'A_prev': A_prev,
                'W': W,
                'b': b,
                'Z': Z,
                'activation': activation,
                'is_encoder': True,
            }
            caches.append(cache)
        
        # MLP forward pass
        for l in range(1, self.mlp_layers + 1):
            layer_idx = self.encoder_layers + l
            A_prev = A
            W = self.parameters[f'W{layer_idx}']
            b = self.parameters[f'b{layer_idx}']
            
            Z = self.matrix_ops.add(self.matrix_ops.multiply(A_prev, W), b)
            activation = self.mlp_activations[l-1]
            A = self._activate(Z, activation)
            
            cache = {
                'A_prev': A_prev,
                'W': W,
                'b': b,
                'Z': Z,
                'activation': activation,
                'is_encoder': False,
            }
            caches.append(cache)
        
        return A, caches
    
    def _compute_loss(self, A_final, Y):
        """Compute cross-entropy loss with optional L2 regularization."""
        m = len(Y)
        epsilon = 1e-8
        loss = 0.0
        
        for i in range(len(Y)):
            for j in range(len(Y[0])):
                a_val = max(epsilon, min(1.0 - epsilon, A_final[i][j]))
                if Y[i][j] > 0:
                    loss -= Y[i][j] * math.log(a_val)
        
        loss /= m
        
        # L2 regularization (only on MLP layers, not encoder)
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for l in range(self.encoder_layers + 1, self.total_layers + 1):
                W = self.parameters[f'W{l}']
                for row in W:
                    for val in row:
                        l2_loss += val ** 2
            loss += (self.l2_lambda / (2 * m)) * l2_loss
        
        return loss
    
    def _backward_propagation(self, Y, caches):
        """Backpropagation with optional encoder freezing."""
        gradients = {}
        m = len(Y)
        L = len(caches)
        
        # Output layer gradient
        cache_L = caches[L - 1]
        A_L = self._activate(cache_L['Z'], cache_L['activation'])
        
        # dZ = A - Y (for softmax + cross-entropy)
        dZ = self.matrix_ops.subtract(A_L, Y)
        
        # Backprop through all layers
        for l in reversed(range(1, L + 1)):
            cache = caches[l - 1]
            A_prev = cache['A_prev']
            W = cache['W']
            is_encoder = cache['is_encoder']
            
            # Skip gradient computation for frozen encoder
            if is_encoder and self.freeze_encoder:
                # Still need to propagate gradients backward, but don't store dW/db
                if l > 1:
                    W_T = self.matrix_ops.transpose(W)
                    dA_prev = self.matrix_ops.multiply(dZ, W_T)
                    cache_prev = caches[l - 2]
                    dZ = self._activate_backward(
                        dA_prev,
                        cache_prev['Z'],
                        cache_prev['activation'],
                    )
                continue
            
            # Compute gradients
            A_prev_T = self.matrix_ops.transpose(A_prev)
            dW = self.matrix_ops.scalar_multiply(
                self.matrix_ops.multiply(A_prev_T, dZ),
                1.0 / m,
            )
            
            db = self.matrix_ops.scalar_multiply(
                self.matrix_ops.sum_axis0(dZ),
                1.0 / m,
            )
            
            # L2 regularization (only on MLP layers)
            if self.l2_lambda > 0 and not is_encoder:
                l2_grad = self.matrix_ops.scalar_multiply(W, self.l2_lambda / m)
                for i in range(len(dW)):
                    for j in range(len(dW[0])):
                        dW[i][j] += l2_grad[i][j]
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            # Propagate gradient backward
            if l > 1:
                W_T = self.matrix_ops.transpose(W)
                dA_prev = self.matrix_ops.multiply(dZ, W_T)
                cache_prev = caches[l - 2]
                dZ = self._activate_backward(
                    dA_prev,
                    cache_prev['Z'],
                    cache_prev['activation'],
                )
        
        return gradients
    
    def _update_parameters(self, gradients):
        """Update parameters (skip frozen encoder)."""
        for l in range(1, self.total_layers + 1):
            # Skip encoder updates if frozen
            if l <= self.encoder_layers and self.freeze_encoder:
                continue
            
            if f'dW{l}' not in gradients:
                continue
            
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            dW = gradients[f'dW{l}']
            db = gradients[f'db{l}']
            
            if self.use_momentum:
                v_dW = self.velocities[f'dW{l}']
                v_db = self.velocities[f'db{l}']
                
                for i in range(len(W)):
                    for j in range(len(W[0])):
                        v_dW[i][j] = self.momentum_factor * v_dW[i][j] + dW[i][j]
                        W[i][j] -= self.learning_rate * v_dW[i][j]
                
                for j in range(len(b[0])):
                    v_db[0][j] = self.momentum_factor * v_db[0][j] + db[0][j]
                    b[0][j] -= self.learning_rate * v_db[0][j]
            else:
                for i in range(len(W)):
                    for j in range(len(W[0])):
                        W[i][j] -= self.learning_rate * dW[i][j]
                
                for j in range(len(b[0])):
                    b[0][j] -= self.learning_rate * db[0][j]
    
    def get_latent_features(self, X):
        """
        Extract latent features (encoder output).
        
        Args:
            X: Input data
        
        Returns:
            Latent representation
        """
        A = X
        for l in range(1, self.encoder_layers + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = self.matrix_ops.add(self.matrix_ops.multiply(A, W), b)
            activation = self.encoder_activations[l-1]
            A = self._activate(Z, activation)
        return A
    
    def predict(self, X):
        """Make predictions."""
        A_final, _ = self._forward_propagation(X)
        return self.matrix_ops.argmax_axis1(A_final)
    
    def fit(self, X, y, epochs=100, batch_size=32):
        """
        Train the classifier (and optionally fine-tune encoder).
        
        Args:
            X: Training data
            y: Labels (class indices)
            epochs: Number of epochs
            batch_size: Mini-batch size
        
        Yields:
            (epoch, loss, self) after each epoch
        """
        n_samples = len(X)
        n_classes = self.mlp_layer_dims[-1]
        
        # Convert labels to one-hot
        Y = [[0.0 for _ in range(n_classes)] for _ in range(n_samples)]
        for i in range(n_samples):
            Y[i][int(y[i])] = 1.0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            X_shuffled = [X[i] for i in indices]
            Y_shuffled = [Y[i] for i in indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]
                
                # Forward pass
                A_final, caches = self._forward_propagation(X_batch)
                batch_loss = self._compute_loss(A_final, Y_batch)
                
                epoch_loss += batch_loss
                num_batches += 1
                
                # Backward pass
                gradients = self._backward_propagation(Y_batch, caches)
                
                # Update parameters
                self._update_parameters(gradients)
            
            # Average loss
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            yield epoch + 1, avg_loss, self
