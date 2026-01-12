import random
import math
import numpy as np
from utils import ActivationFunctions


class MLPWithEncoder:
    
    def __init__(self, encoder_params, encoder_dims, encoder_activations,
                 mlp_layer_dims, mlp_activations, learning_rate=0.01,
                 l2_lambda=0.0, freeze_encoder=True, use_momentum=False, 
                 momentum_factor=0.9):
        
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
        
        for l in range(1, self.encoder_layers + 1):
            self.parameters[f'W{l}'] = np.array(encoder_params[f'W{l}'], copy=True)
            self.parameters[f'b{l}'] = np.array(encoder_params[f'b{l}'], copy=True)
            
            if self.use_momentum and not self.freeze_encoder:
                shape_w = self.parameters[f'W{l}'].shape
                shape_b = self.parameters[f'b{l}'].shape
                self.velocities[f'dW{l}'] = np.zeros(shape_w)
                self.velocities[f'db{l}'] = np.zeros(shape_b)
        
        self._initialize_mlp_layers()
    
    def _initialize_mlp_layers(self):
        for l in range(1, self.mlp_layers + 1):
            layer_idx = self.encoder_layers + l
            rows = self.mlp_layer_dims[l-1]
            cols = self.mlp_layer_dims[l]
            
            if self.mlp_activations[l-1] == 'relu':
                scale = math.sqrt(2.0 / rows)
            else:
                scale = math.sqrt(1.0 / rows)
            
            self.parameters[f'W{layer_idx}'] = np.random.randn(rows, cols) * scale
            self.parameters[f'b{layer_idx}'] = np.zeros((1, cols))
            
            if self.use_momentum:
                self.velocities[f'dW{layer_idx}'] = np.zeros((rows, cols))
                self.velocities[f'db{layer_idx}'] = np.zeros((1, cols))
    
    def _activate(self, Z, activation):
        return ActivationFunctions.apply(Z, activation)
    
    def _activate_backward(self, dA, Z, activation):
        return ActivationFunctions.apply_backward(dA, Z, activation)
    
    def _forward_propagation(self, X):
        caches = []
        A = X
        
        for l in range(1, self.encoder_layers + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(A_prev, W) + b
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
        
        for l in range(1, self.mlp_layers + 1):
            layer_idx = self.encoder_layers + l
            A_prev = A
            W = self.parameters[f'W{layer_idx}']
            b = self.parameters[f'b{layer_idx}']
            
            Z = np.dot(A_prev, W) + b
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
        m = len(Y)
        epsilon = 1e-8
        
        A_val = np.clip(A_final, epsilon, 1.0 - epsilon)
        
        loss = -np.sum(Y * np.log(A_val)) / m
        
        if self.l2_lambda > 0:
            l2_sum = 0.0
            for l in range(self.encoder_layers + 1, self.total_layers + 1):
                W = self.parameters[f'W{l}']
                l2_sum += np.sum(W ** 2)
            loss += (self.l2_lambda / (2 * m)) * l2_sum
        
        return loss
    
    def _backward_propagation(self, Y, caches):
        gradients = {}
        m = len(Y)
        L = len(caches)
        
        cache_L = caches[L - 1]
        A_L = self._activate(cache_L['Z'], cache_L['activation'])
        
        dZ = A_L - Y
        
        for l in reversed(range(1, L + 1)):
            cache = caches[l - 1]
            A_prev = cache['A_prev']
            W = cache['W']
            is_encoder = cache['is_encoder']
            
            if is_encoder and self.freeze_encoder:
                if l > 1:
                    W_T = W.T
                    dA_prev = np.dot(dZ, W_T)
                    cache_prev = caches[l - 2]
                    dZ = self._activate_backward(dA_prev, cache_prev['Z'], cache_prev['activation'])
                continue
            
            dW = np.dot(A_prev.T, dZ) / m
            
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            if self.l2_lambda > 0 and not is_encoder:
                dW += (self.l2_lambda / m) * W
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            if l > 1:
                W_T = W.T
                dA_prev = np.dot(dZ, W_T)
                
                cache_prev = caches[l - 2]
                dZ = self._activate_backward(dA_prev, cache_prev['Z'], cache_prev['activation'])
                
        return gradients
    
    def _update_parameters(self, gradients):
        for l in range(1, self.total_layers + 1):
            if l <= self.encoder_layers and self.freeze_encoder:
                continue
            
            if f'dW{l}' not in gradients:
                continue
            
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
    
    def get_latent_features(self, X):
        X = np.array(X)
        A = X
        for l in range(1, self.encoder_layers + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(A, W) + b
            activation = self.encoder_activations[l-1]
            A = self._activate(Z, activation)
        return A
    
    def predict(self, X):
        X = np.array(X)
        A_final, _ = self._forward_propagation(X)
        return np.argmax(A_final, axis=1)
    
    def fit(self, X, y, epochs=100, batch_size=32, stop_callback=None):
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)
        n_classes = self.mlp_layer_dims[-1]
        
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y.astype(int)] = 1.0
        
        for epoch in range(epochs):
            if stop_callback and stop_callback():
                return
                
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, n_samples, batch_size):
                if stop_callback and stop_callback():
                    return
                    
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]
                
                A_final, caches = self._forward_propagation(X_batch)
                batch_loss = self._compute_loss(A_final, Y_batch)
                
                epoch_loss += batch_loss
                num_batches += 1
                
                gradients = self._backward_propagation(Y_batch, caches)
                self._update_parameters(gradients)
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            yield epoch + 1, avg_loss, self
