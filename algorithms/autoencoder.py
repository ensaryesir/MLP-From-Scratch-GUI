import random
import math
import numpy as np
from utils import ActivationFunctions


class Autoencoder:
    
    def __init__(self, encoder_dims, activation_funcs, learning_rate=0.01, 
                 use_momentum=False, momentum_factor=0.9):
        self.encoder_dims = encoder_dims
        self.decoder_dims = encoder_dims[::-1]
        self.layer_dims = encoder_dims + self.decoder_dims[1:]
        
        self.activation_funcs = activation_funcs
        self.learning_rate = learning_rate
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        
        self.parameters = {}
        self.velocities = {}
        
        self.L = len(self.layer_dims) - 1
        self.encoder_layers = len(encoder_dims) - 1
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        for l in range(1, self.L + 1):
            rows = self.layer_dims[l-1]
            cols = self.layer_dims[l]
            
            if l <= len(self.activation_funcs) and self.activation_funcs[l-1] == 'relu':
                scale = math.sqrt(2.0 / rows)
            else:
                scale = math.sqrt(1.0 / rows)
            
            self.parameters[f'W{l}'] = np.random.randn(rows, cols) * scale
            self.parameters[f'b{l}'] = np.zeros((1, cols))
            
            if self.use_momentum:
                self.velocities[f'dW{l}'] = np.zeros((rows, cols))
                self.velocities[f'db{l}'] = np.zeros((1, cols))
    
    def _activate(self, Z, activation):
        return ActivationFunctions.apply(Z, activation)
    
    def _activate_backward(self, dA, Z, activation):
        return ActivationFunctions.apply_backward(dA, Z, activation)
    
    def _forward_propagation(self, X):
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
                'activation': activation,
            }
            caches.append(cache)
        
        return A, caches
    
    def _compute_reconstruction_loss(self, X_reconstructed, X_original):
        return np.mean((X_reconstructed - X_original) ** 2)
    
    def _backward_propagation(self, X_original, caches):
        gradients = {}
        m = len(X_original)
        L = len(caches)
        n_features = X_original.shape[1] if m > 0 else 1
        
        cache_L = caches[L - 1]
        A_L = self._activate(cache_L['Z'], cache_L['activation'])
        
        dA = 2.0 * (A_L - X_original) / (m * n_features)
        
        dZ = self._activate_backward(dA, cache_L['Z'], cache_L['activation'])
        
        for l in reversed(range(1, L + 1)):
            cache = caches[l - 1]
            A_prev = cache['A_prev']
            W = cache['W']
            
            dW = np.dot(A_prev.T, dZ)
            
            db = np.sum(dZ, axis=0, keepdims=True)
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            if l > 1:
                W_T = W.T
                dA_prev = np.dot(dZ, W_T)
                cache_prev = caches[l - 2]
                dZ = self._activate_backward(dA_prev, cache_prev['Z'], cache_prev['activation'])
        
        return gradients
    
    def _update_parameters(self, gradients):
        for l in range(1, self.L + 1):
            W_key = f'W{l}'
            b_key = f'b{l}'
            dW = gradients[f'dW{l}']
            db = gradients[f'db{l}']
            
            if self.use_momentum:
                v_dW = self.velocities[f'dW{l}']
                v_db = self.velocities[f'db{l}']
                
                v_dW[:] = self.momentum_factor * v_dW + (1.0 - self.momentum_factor) * dW
                v_db[:] = self.momentum_factor * v_db + (1.0 - self.momentum_factor) * db
                
                self.parameters[W_key] -= self.learning_rate * v_dW
                self.parameters[b_key] -= self.learning_rate * v_db
            else:
                self.parameters[W_key] -= self.learning_rate * dW
                self.parameters[b_key] -= self.learning_rate * db
    
    def encode(self, X):
        X = np.array(X)
        A = X
        for l in range(1, self.encoder_layers + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(A, W) + b
            activation = self.activation_funcs[l-1]
            A = self._activate(Z, activation)
        return A
    
    def decode(self, Z):
        Z = np.array(Z)
        A = Z
        for l in range(self.encoder_layers + 1, self.L + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z_layer = np.dot(A, W) + b
            activation = self.activation_funcs[l-1]
            A = self._activate(Z_layer, activation)
        return A
    
    def reconstruct(self, X):
        latent = self.encode(X)
        return self.decode(latent)
    
    def fit(self, X, epochs=100, batch_size=32, stop_callback=None):
        X = np.array(X)
        n_samples = len(X)
        
        for epoch in range(epochs):
            if stop_callback and stop_callback():
                return
                
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, n_samples, batch_size):
                if stop_callback and stop_callback():
                    return
                    
                X_batch = X_shuffled[i : i + batch_size]
                
                X_reconstructed, caches = self._forward_propagation(X_batch)
                
                batch_loss = self._compute_reconstruction_loss(X_reconstructed, X_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                gradients = self._backward_propagation(X_batch, caches)
                self._update_parameters(gradients)
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            yield epoch + 1, avg_loss, self
    
    def get_latent_dim(self):
        return self.encoder_dims[-1]
    
    def get_encoder_weights(self):
        encoder_params = {}
        for l in range(1, self.encoder_layers + 1):
            encoder_params[f'W{l}'] = self.parameters[f'W{l}']
            encoder_params[f'b{l}'] = self.parameters[f'b{l}']
        return encoder_params
