import random
import math

from utils import ActivationFunctions, MatrixOperations


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
        
        # Force linear output activation for regression (with warning)
        if self.task == 'regression':
            if self.activation_funcs[-1] != 'linear':
                print(f"Warning: Output activation changed from '{self.activation_funcs[-1]}' to 'linear' for regression")
            self.activation_funcs[-1] = 'linear'
            
        self.parameters = {}
        self.velocities = {}  # For momentum

        self.L = len(layer_dims) - 1
        self.matrix_ops = MatrixOperations()  # Matrix operations helper
        self.activations = ActivationFunctions()  # Activation functions helper
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize weights using He init for ReLU, Xavier for others."""
        for l in range(1, self.L + 1):
            rows = self.layer_dims[l-1]
            cols = self.layer_dims[l]
            
            if self.activation_funcs[l-1] == 'relu':
                # He init for ReLU
                scale = math.sqrt(2.0 / rows)
            else:
                # Xavier init for tanh/sigmoid/linear
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
            
            # Initialize velocities to zeros if momentum is used
            if self.use_momentum:
                self.velocities[f'dW{l}'] = [[0.0 for _ in range(cols)] for _ in range(rows)]
                self.velocities[f'db{l}'] = [[0.0 for _ in range(cols)]]

    def _activate(self, Z, activation):
        """Apply activation function (delegates to ActivationFunctions class)."""
        return self.activations.apply(Z, activation)
    
    def _activate_backward(self, dA, Z, activation):
        """Apply activation derivative (delegates to ActivationFunctions class)."""
        return self.activations.apply_backward(dA, Z, activation)
    
    # Forward and Backward Propagation
    
    def _forward_propagation(self, X):
        """Forward pass through all layers. Returns output and caches."""
        caches = []
        A = X
        
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = self.matrix_ops.add(self.matrix_ops.multiply(A_prev, W), b)
            activation = self.activation_funcs[l-1]
            A = self._activate(Z, activation)
            
            cache = {'A_prev': A_prev,'W': W,'b': b,'Z': Z,'activation': activation,}
            caches.append(cache)
        
        return A, caches
    
    def _compute_loss(self, A_final, Y):
        """Compute cross-entropy loss with optional L2 regularization."""
        m = len(Y)
        
        if self.task == 'regression':
            # MSE Loss: (1/2m) * sum((y_pred - y_true)^2)
            loss = 0.0
            for i in range(len(Y)):
                for j in range(len(Y[0])):
                    diff = A_final[i][j] - Y[i][j]
                    loss += diff * diff
            loss /= (2 * m)
        else:
            # Cross-entropy with epsilon for numerical stability
            epsilon = 1e-8
            loss = 0.0
            
            for i in range(len(Y)):
                for j in range(len(Y[0])):
                    # Clip values for numerical stability
                    a_val = max(epsilon, min(1.0 - epsilon, A_final[i][j]))
                    
                    # Multi-class CE: -sum(y * log(a))
                    # Binary CE (one-hot, 2 outputs): Same as above
                    # Binary CE (single output): -[y*log(a) + (1-y)*log(1-a)]
                    if Y[i][j] > 0:
                        loss -= Y[i][j] * math.log(a_val)
                    elif len(Y[0]) == 1:  # Single output binary classification
                        # Add (1-y)*log(1-a) term for binary CE
                        loss -= (1 - Y[i][j]) * math.log(1 - a_val)
            
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
        
        # Output layer gradient
        cache_L = caches[L - 1]
        A_L = self._activate(cache_L['Z'], cache_L['activation'])
        
        # Validate activation-task combination
        output_activation = cache_L['activation']
        valid_combinations = [
            (output_activation == 'softmax' and self.task == 'classification'),
            (output_activation == 'sigmoid' and self.task == 'classification'),
            (output_activation == 'linear' and self.task == 'regression'),
        ]
        
        if not any(valid_combinations):
            print(
                f"Warning: Output activation '{output_activation}' with task '{self.task}' "
                f"may require custom gradient computation. Using dZ = A - Y."
            )
        
        # Simplified gradient dZ = A - Y
        dZ = self.matrix_ops.subtract(A_L, Y)
        
        # Iterate backwards through all layers
        for l in reversed(range(1, L + 1)):
            cache = caches[l - 1]
            A_prev = cache['A_prev']
            W = cache['W']
            
            # dW = (1/m) * A_prev^T @ dZ
            A_prev_T = self.matrix_ops.transpose(A_prev)
            dW = self.matrix_ops.scalar_multiply(self.matrix_ops.multiply(A_prev_T, dZ),1.0 / m)
            
            # db = (1/m) * sum(dZ)
            db = self.matrix_ops.scalar_multiply(self.matrix_ops.sum_axis0(dZ),1.0 / m)
            
            # L2 gradient
            if self.l2_lambda > 0:
                l2_grad = self.matrix_ops.scalar_multiply(W, self.l2_lambda / m)
                for i in range(len(dW)):
                    for j in range(len(dW[0])):
                        dW[i][j] += l2_grad[i][j]
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            if l > 1:
                W_T = self.matrix_ops.transpose(W)
                dA_prev = self.matrix_ops.multiply(dZ, W_T)
                cache_prev = caches[l - 2]
                dZ = self._activate_backward(dA_prev,cache_prev['Z'],cache_prev['activation'])
        
        return gradients
    
    def _update_parameters(self, gradients):
        """Update parameters using gradient descent: θ = θ - α * dθ"""
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
    
    def predict(self, X):
        """Make predictions."""
        A_final, _ = self._forward_propagation(X)
        
        if self.task == 'regression':
            if len(A_final[0]) == 1:
                return [row[0] for row in A_final]
            return A_final
        else:
            return self.matrix_ops.argmax_axis1(A_final)
    
    def fit(self, X, y, epochs=100, batch_size=32):
        """Train using mini-batch gradient descent. Yields (epoch, loss, self)."""
        n_samples = len(X)
        
        if self.task == 'regression':
            Y = []
            for val in y:
                if isinstance(val, (list, tuple)):
                    Y.append(val)
                else:
                    Y.append([float(val)])
        else:
            n_classes = self.layer_dims[-1]
            Y = [[0.0 for _ in range(n_classes)] for _ in range(n_samples)]
            for i in range(n_samples):
                Y[i][int(y[i])] = 1.0
        
        for epoch in range(epochs):
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            X_shuffled = [X[i] for i in indices]
            Y_shuffled = [Y[i] for i in indices]
            
            epoch_loss = 0.0
            num_batches = 0  # Renamed: actually counts batches, not samples
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]
                
                A_final, caches = self._forward_propagation(X_batch)
                batch_loss = self._compute_loss(A_final, Y_batch)
                
                # batch_loss is already averaged over samples in the batch
                # We accumulate and will average over number of batches
                epoch_loss += batch_loss
                num_batches += 1
                
                gradients = self._backward_propagation(Y_batch, caches)
                self._update_parameters(gradients)
            
            # Average loss over all batches
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            yield epoch + 1, avg_loss, self
