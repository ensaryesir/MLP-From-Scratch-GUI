"""
Autoencoder implementation for MNIST feature extraction.
Encoder-Decoder architecture with configurable layers.
"""

import random
import math

from utils import ActivationFunctions, MatrixOperations


class Autoencoder:
    """
    Autoencoder for unsupervised feature learning.
    
    Architecture:
        Encoder: Input → [hidden layers] → Latent
        Decoder: Latent → [hidden layers] → Reconstruction
    
    Example:
        encoder_dims = [784, 256, 128, 32]
        → Encoder: 784 → 256 → 128 → 32
        → Decoder: 32 → 128 → 256 → 784 (symmetric)
    """
    
    def __init__(self, encoder_dims, activation_funcs, learning_rate=0.01, 
                 use_momentum=False, momentum_factor=0.9):
        """
        Initialize autoencoder.
        
        Args:
            encoder_dims: List of layer sizes for encoder (e.g., [784, 256, 128, 32])
            activation_funcs: List of activation functions for ALL layers
                             (encoder + decoder, e.g., ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'])
            learning_rate: Learning rate for gradient descent
            use_momentum: Whether to use momentum
            momentum_factor: Momentum factor (typically 0.9)
        """
        self.encoder_dims = encoder_dims
        
        # Create decoder dims (reverse of encoder, excluding input layer)
        self.decoder_dims = encoder_dims[::-1]
        
        # Full architecture: encoder + decoder
        self.layer_dims = encoder_dims + self.decoder_dims[1:]
        
        self.activation_funcs = activation_funcs
        self.learning_rate = learning_rate
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        
        self.parameters = {}
        self.velocities = {}  # For momentum
        
        self.L = len(self.layer_dims) - 1  # Number of weight matrices
        self.encoder_layers = len(encoder_dims) - 1  # Number of encoder layers
        
        self.matrix_ops = MatrixOperations()
        self.activations = ActivationFunctions()
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights using He init for ReLU, Xavier for others."""
        for l in range(1, self.L + 1):
            rows = self.layer_dims[l-1]
            cols = self.layer_dims[l]
            
            # Determine initialization scale
            if l <= len(self.activation_funcs) and self.activation_funcs[l-1] == 'relu':
                scale = math.sqrt(2.0 / rows)  # He init
            else:
                scale = math.sqrt(1.0 / rows)  # Xavier init
            
            # Initialize weights
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
            self.parameters[f'b{l}'] = [[0.0 for _ in range(cols)]]
            
            # Initialize velocities for momentum
            if self.use_momentum:
                self.velocities[f'dW{l}'] = [[0.0 for _ in range(cols)] for _ in range(rows)]
                self.velocities[f'db{l}'] = [[0.0 for _ in range(cols)]]
    
    def _activate(self, Z, activation):
        """Apply activation function."""
        return self.activations.apply(Z, activation)
    
    def _activate_backward(self, dA, Z, activation):
        """Apply activation derivative."""
        return self.activations.apply_backward(dA, Z, activation)
    
    def _forward_propagation(self, X):
        """Forward pass through encoder and decoder."""
        caches = []
        A = X
        
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = self.matrix_ops.add(self.matrix_ops.multiply(A_prev, W), b)
            activation = self.activation_funcs[l-1]
            A = self._activate(Z, activation)
            
            cache = {
                'A_prev': A_prev,
                'W': W,
                'b': b,
                'Z': Z,
                'activation': activation,
            }
            caches.append(cache)
        
        return A, caches
    
    def _compute_reconstruction_loss(self, X_reconstructed, X_original):
        """Compute Mean Squared Error between reconstructed and original."""
        m = len(X_original)
        n_features = len(X_original[0]) if m > 0 else 0
        
        loss = 0.0
        for i in range(m):
            for j in range(n_features):
                diff = X_reconstructed[i][j] - X_original[i][j]
                loss += diff * diff
        
        loss /= (m * n_features)  # Normalize by total elements
        return loss
    
    def _backward_propagation(self, X_original, caches):
        """Backpropagation for reconstruction loss."""
        gradients = {}
        m = len(X_original)
        L = len(caches)
        
        # Output layer gradient: dL/dA = 2 * (A - X) / (m * n_features)
        cache_L = caches[L - 1]
        A_L = self._activate(cache_L['Z'], cache_L['activation'])
        
        n_features = len(X_original[0]) if m > 0 else 1
        
        # Gradient of MSE loss
        dA = [[0.0 for _ in range(len(A_L[0]))] for _ in range(len(A_L))]
        for i in range(len(A_L)):
            for j in range(len(A_L[0])):
                dA[i][j] = 2.0 * (A_L[i][j] - X_original[i][j]) / (m * n_features)
        
        # Backprop through activation
        dZ = self._activate_backward(dA, cache_L['Z'], cache_L['activation'])
        
        # Iterate backwards through all layers
        for l in reversed(range(1, L + 1)):
            cache = caches[l - 1]
            A_prev = cache['A_prev']
            W = cache['W']
            
            # dW = (1/m) * A_prev^T @ dZ (already normalized in dZ)
            A_prev_T = self.matrix_ops.transpose(A_prev)
            dW = self.matrix_ops.multiply(A_prev_T, dZ)
            
            # db = sum(dZ) along axis 0
            db = self.matrix_ops.sum_axis0(dZ)
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
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
        """Update parameters using gradient descent with optional momentum."""
        for l in range(1, self.L + 1):
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
    
    def encode(self, X):
        """
        Encode input through encoder layers only.
        
        Args:
            X: Input data (n_samples, input_dim)
        
        Returns:
            Latent representation (n_samples, latent_dim)
        """
        A = X
        for l in range(1, self.encoder_layers + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = self.matrix_ops.add(self.matrix_ops.multiply(A, W), b)
            activation = self.activation_funcs[l-1]
            A = self._activate(Z, activation)
        return A
    
    def decode(self, Z):
        """
        Decode latent representation through decoder layers.
        
        Args:
            Z: Latent representation (n_samples, latent_dim)
        
        Returns:
            Reconstructed data (n_samples, output_dim)
        """
        A = Z
        for l in range(self.encoder_layers + 1, self.L + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z_layer = self.matrix_ops.add(self.matrix_ops.multiply(A, W), b)
            activation = self.activation_funcs[l-1]
            A = self._activate(Z_layer, activation)
        return A
    
    def reconstruct(self, X):
        """
        Full encode-decode pass.
        
        Args:
            X: Input data
        
        Returns:
            Reconstructed data
        """
        latent = self.encode(X)
        return self.decode(latent)
    
    def fit(self, X, epochs=100, batch_size=32):
        """
        Train autoencoder on unlabeled data.
        
        Args:
            X: Training data (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Mini-batch size
        
        Yields:
            (epoch, reconstruction_loss, self) after each epoch
        """
        n_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            X_shuffled = [X[i] for i in indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                
                # Forward pass
                X_reconstructed, caches = self._forward_propagation(X_batch)
                
                # Compute loss
                batch_loss = self._compute_reconstruction_loss(X_reconstructed, X_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # Backward pass
                gradients = self._backward_propagation(X_batch, caches)
                
                # Update parameters
                self._update_parameters(gradients)
            
            # Average loss over all batches
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            yield epoch + 1, avg_loss, self
    
    def compute_loss_on(self, X):
        """
        Compute reconstruction loss on arbitrary dataset.
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            Reconstruction loss (MSE)
        """
        if len(X) == 0:
            return 0.0
        
        # Forward pass to reconstruct
        X_reconstructed, _ = self._forward_propagation(X)
        
        # Compute reconstruction loss
        return self._compute_reconstruction_loss(X_reconstructed, X)
    
    def get_latent_dim(self):
        """Return the size of the latent representation."""
        return self.encoder_dims[-1]
    
    def get_encoder_weights(self):
        """
        Extract encoder weights for transfer to hybrid model.
        
        Returns:
            Dictionary of encoder parameters (W1, b1, ..., W_encoder, b_encoder)
        """
        encoder_params = {}
        for l in range(1, self.encoder_layers + 1):
            encoder_params[f'W{l}'] = self.parameters[f'W{l}']
            encoder_params[f'b{l}'] = self.parameters[f'b{l}']
        return encoder_params
