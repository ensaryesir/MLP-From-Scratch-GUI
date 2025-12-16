import math


class ActivationFunctions:
    
    @staticmethod
    def relu(Z):
        """
        ReLU (Rectified Linear Unit): f(z) = max(0, z)
        Range: [0, ∞)
        Used in hidden layers for deep networks.
        """
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                result[i][j] = max(0.0, Z[i][j])
        return result
    
    @staticmethod
    def relu_backward(dA, Z):
        """
        ReLU derivative: f'(z) = 1 if z > 0, else 0
        Gradient flows only through positive activations.
        """
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                if Z[i][j] <= 0:
                    result[i][j] = 0.0
                else:
                    result[i][j] = dA[i][j]
        return result
    
    @staticmethod
    def tanh(Z):
        """
        Tanh (Hyperbolic Tangent): f(z) = tanh(z)
        Range: (-1, 1)
        Zero-centered, good for hidden layers.
        """
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                result[i][j] = math.tanh(Z[i][j])
        return result
    
    @staticmethod
    def tanh_backward(dA, Z):
        """
        Tanh derivative: f'(z) = 1 - tanh²(z)
        """
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                tanh_val = math.tanh(Z[i][j])
                result[i][j] = dA[i][j] * (1.0 - tanh_val ** 2)
        return result
    
    @staticmethod
    def sigmoid(Z):
        """
        Sigmoid (Logistic): f(z) = 1 / (1 + e^(-z))
        Range: (0, 1)
        Used for binary classification output layer.
        """
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                # Clip for numerical stability
                z_val = max(-500, min(500, Z[i][j]))
                result[i][j] = 1.0 / (1.0 + math.exp(-z_val))
        return result
    
    @staticmethod
    def sigmoid_backward(dA, Z):
        """
        Sigmoid derivative: f'(z) = σ(z) * (1 - σ(z))
        """
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                z_val = max(-500, min(500, Z[i][j]))
                sig = 1.0 / (1.0 + math.exp(-z_val))
                result[i][j] = dA[i][j] * sig * (1.0 - sig)
        return result
    
    @staticmethod
    def softmax(Z):
        """
        Softmax: f(z_i) = e^(z_i) / Σ(e^(z_j))
        Converts logits to probability distribution.
        Used for multi-class classification output layer.
        """
        result = [[0.0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
        
        for i in range(len(Z)):
            # Find max for numerical stability (prevents overflow)
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
            
            # Normalize to get probabilities
            for j in range(len(Z[0])):
                result[i][j] = exp_vals[j] / exp_sum
        
        return result
    
    @staticmethod
    def softmax_backward(dA, Z):
        """
        Softmax derivative (simplified for cross-entropy loss).
        
        IMPORTANT: This simplified form is ONLY valid when:
        1. Softmax is used in the OUTPUT layer
        2. Combined with cross-entropy loss
        3. The gradient dA is already computed as (y_pred - y_true)
        
        Mathematical Background:
        - Softmax: a_i = e^(z_i) / Σ(e^(z_j))
        - Cross-Entropy: L = -Σ(y_i * log(a_i))
        - Combined gradient: ∂L/∂z_i = a_i - y_i (simplifies!)
        
        In MLP implementation:
        - Output layer: dZ = A_final - Y (computed in _backward_propagation)
        - This function just passes dA through (no computation needed)
        
        For softmax in hidden layers (NOT RECOMMENDED):
        - Full Jacobian matrix would be needed:
          ∂a_i/∂z_j = a_i * (δ_ij - a_j)
        - Where δ_ij is Kronecker delta (1 if i==j, else 0)
        - But softmax is rarely used in hidden layers (use ReLU/tanh instead)
        
        Args:
            dA: Gradient from next layer (should be y_pred - y_true for output)
            Z: Pre-activation values (not used in simplified form)
        
        Returns:
            dA: Gradient passed through unchanged
        """
        return dA
    
    @staticmethod
    def linear(Z):
        """
        Linear (Identity): f(z) = z
        No transformation, passes input as-is.
        Used in regression output layers.
        """
        return [row[:] for row in Z]
    
    @staticmethod
    def linear_backward(dA, Z):
        """
        Linear derivative: f'(z) = 1
        Gradient passes through unchanged.
        """
        return dA
    
    @staticmethod
    def apply(Z, activation_name):
        """
        Apply activation function by name.
        
        Args:
            Z: Input matrix (pre-activation values)
            activation_name: String name of activation ('relu', 'tanh', etc.)
        
        Returns:
            Activated output matrix
        """
        if activation_name == 'relu':
            return ActivationFunctions.relu(Z)
        elif activation_name == 'tanh':
            return ActivationFunctions.tanh(Z)
        elif activation_name == 'sigmoid':
            return ActivationFunctions.sigmoid(Z)
        elif activation_name == 'softmax':
            return ActivationFunctions.softmax(Z)
        elif activation_name == 'linear':
            return ActivationFunctions.linear(Z)
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")
    
    @staticmethod
    def apply_backward(dA, Z, activation_name):
        """
        Apply activation derivative by name.
        
        Args:
            dA: Gradient from next layer
            Z: Pre-activation values from forward pass
            activation_name: String name of activation
        
        Returns:
            Gradient with respect to pre-activation (dZ)
        """
        if activation_name == 'relu':
            return ActivationFunctions.relu_backward(dA, Z)
        elif activation_name == 'tanh':
            return ActivationFunctions.tanh_backward(dA, Z)
        elif activation_name == 'sigmoid':
            return ActivationFunctions.sigmoid_backward(dA, Z)
        elif activation_name == 'softmax':
            return ActivationFunctions.softmax_backward(dA, Z)
        elif activation_name == 'linear':
            return ActivationFunctions.linear_backward(dA, Z)
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")
