import random
import math
import numpy as np
from utils import ActivationFunctions


class MLP:

    def __init__(self, layer_dims, activation_funcs, learning_rate=0.01, l2_lambda=0.0, 
                 task='classification', use_momentum=False, momentum_factor=0.9):
        self.layer_dims = layer_dims
        self.activation_funcs = activation_funcs
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.task = task
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        
        if self.task == 'regression':
            if self.activation_funcs[-1] != 'linear':
                print(f"Warning: Output activation changed from '{self.activation_funcs[-1]}' to 'linear' for regression")
            self.activation_funcs[-1] = 'linear'
            
        self.parameters = {}
        self.velocities = {}  

        self.L = len(layer_dims) - 1
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        for l in range(1, self.L + 1):
            rows = self.layer_dims[l-1]
            cols = self.layer_dims[l]
            
            if self.activation_funcs[l-1] == 'relu':
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
            
            cache = {'A_prev': A_prev, 'W': W, 'b': b, 'Z': Z, 'activation': activation}
            caches.append(cache)
        
        return A, caches
    
    def _compute_loss(self, A_final, Y):
        m = len(Y)
        loss = 0.0
        
        if self.task == 'regression':
            loss = np.sum((A_final - Y) ** 2) / (2 * m)
        else:
            epsilon = 1e-8
            A_val = np.clip(A_final, epsilon, 1.0 - epsilon)
            
            if Y.shape[1] == 1:
                ce_loss = -(Y * np.log(A_val) + (1 - Y) * np.log(1 - A_val))
                loss = np.sum(ce_loss) / m 
            else:
                loss = -np.sum(Y * np.log(A_val)) / m

        if self.l2_lambda > 0:
            l2_sum = 0.0
            for l in range(1, self.L + 1):
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
            
            dW = np.dot(A_prev.T, dZ) / m
            
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            if self.l2_lambda > 0:
                dW += (self.l2_lambda / m) * W
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            if l > 1:
                dA_prev = np.dot(dZ, W.T)
                
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

    def predict(self, X):
        X = np.array(X)
        A_final, _ = self._forward_propagation(X)
        
        if self.task == 'regression':
            if A_final.shape[1] == 1:
                return A_final.flatten()
            return A_final
        else:
            return np.argmax(A_final, axis=1)
    
    def fit(self, X, y, epochs=100, batch_size=32, stop_callback=None):
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)
        
        if self.task == 'regression':
            if y.ndim == 1:
                Y = y.reshape(-1, 1)
            else:
                Y = y
        else:
            n_classes = self.layer_dims[-1]
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
