import numpy as np

class ActivationFunctions:
    
    @staticmethod
    def relu(Z):
        """ReLU: f(z) = max(0, z)"""
        return np.maximum(0, Z)
    
    @staticmethod
    def relu_backward(dA, Z):
        """ReLU derivative: 1 if z > 0 else 0"""
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    @staticmethod
    def tanh(Z):
        """Tanh: f(z) = tanh(z)"""
        return np.tanh(Z)
    
    @staticmethod
    def tanh_backward(dA, Z):
        """Tanh derivative: 1 - tanh^2(z)"""
        return dA * (1 - np.tanh(Z)**2)
    
    @staticmethod
    def sigmoid(Z):
        """Sigmoid: f(z) = 1 / (1 + e^-z)"""
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))
    
    @staticmethod
    def sigmoid_backward(dA, Z):
        """Sigmoid derivative: s * (1 - s)"""
        Z = np.clip(Z, -500, 500)
        s = 1.0 / (1.0 + np.exp(-Z))
        return dA * s * (1.0 - s)
    
    @staticmethod
    def softmax(Z):
        """Softmax (stable): e^z / sum(e^z)"""
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_backward(dA, Z):
        """Softmax derivative (identity for Cross-Entropy + Softmax output)."""
        return dA
    
    @staticmethod
    def linear(Z):
        """Linear: f(z) = z"""
        return Z
    
    @staticmethod
    def linear_backward(dA, Z):
        """Linear derivative: 1"""
        return dA
    
    @staticmethod
    def apply(Z, activation_name):
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
