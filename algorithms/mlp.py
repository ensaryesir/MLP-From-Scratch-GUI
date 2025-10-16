"""
Çok Katmanlı Perceptron (Multi-Layer Perceptron - MLP)
Backpropagation algoritması ile sıfırdan yazılmış tam bağlantılı sinir ağı.
"""

import numpy as np


class MLP:
    """
    Multi-Layer Perceptron sınıfı.
    Backpropagation ile eğitilen derin sinir ağı implementasyonu.
    """
    
    def __init__(self, layer_dims, activation_funcs, learning_rate=0.01, l2_lambda=0.0):
        """
        MLP'yi başlatır.
        
        Args:
            layer_dims: Katman boyutları listesi, örn: [2, 5, 3] 
                       (2 girdi, 5 gizli nöron, 3 çıktı)
            activation_funcs: Her katman için aktivasyon fonksiyonu isimleri,
                             örn: ['relu', 'softmax']
            learning_rate: Öğrenme oranı
            l2_lambda: L2 regularization katsayısı
        """
        self.layer_dims = layer_dims
        self.activation_funcs = activation_funcs
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.parameters = {}
        self.L = len(layer_dims) - 1  # Katman sayısı (girdi katmanı hariç)
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Ağırlıkları ve bias'ları Xavier/He initialization ile başlatır."""
        for l in range(1, self.L + 1):
            # He initialization for ReLU, Xavier for others
            if self.activation_funcs[l-1] == 'relu':
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l-1], self.layer_dims[l]
                ) * np.sqrt(2.0 / self.layer_dims[l-1])
            else:
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l-1], self.layer_dims[l]
                ) * np.sqrt(1.0 / self.layer_dims[l-1])
            
            self.parameters[f'b{l}'] = np.zeros((1, self.layer_dims[l]))
    
    # ========== Aktivasyon Fonksiyonları ==========
    
    def _relu(self, Z):
        """ReLU aktivasyon fonksiyonu."""
        return np.maximum(0, Z)
    
    def _relu_backward(self, dA, Z):
        """ReLU'nun türevi."""
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ
    
    def _tanh(self, Z):
        """Tanh aktivasyon fonksiyonu."""
        return np.tanh(Z)
    
    def _tanh_backward(self, dA, Z):
        """Tanh'ın türevi: 1 - tanh²(Z)."""
        A = np.tanh(Z)
        dZ = dA * (1 - A ** 2)
        return dZ
    
    def _sigmoid(self, Z):
        """Sigmoid (Logistic) aktivasyon fonksiyonu."""
        # Numerical stability için clip
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def _sigmoid_backward(self, dA, Z):
        """Sigmoid'in türevi: σ(Z) * (1 - σ(Z))."""
        A = self._sigmoid(Z)
        dZ = dA * A * (1 - A)
        return dZ
    
    def _softmax(self, Z):
        """Softmax aktivasyon fonksiyonu."""
        # Numerical stability için max çıkar
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def _softmax_backward(self, dA, Z):
        """
        Softmax + Cross-Entropy için türev.
        Cross-entropy loss ile birlikte kullanıldığında,
        türev basitleşir: y_pred - y_true
        """
        return dA
    
    def _linear(self, Z):
        """Linear (Identity) aktivasyon fonksiyonu."""
        return Z
    
    def _linear_backward(self, dA, Z):
        """Linear aktivasyonun türevi."""
        return dA
    
    def _activate(self, Z, activation):
        """Belirtilen aktivasyon fonksiyonunu uygular."""
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
            raise ValueError(f"Bilinmeyen aktivasyon fonksiyonu: {activation}")
    
    def _activate_backward(self, dA, Z, activation):
        """Belirtilen aktivasyon fonksiyonunun türevini hesaplar."""
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
            raise ValueError(f"Bilinmeyen aktivasyon fonksiyonu: {activation}")
    
    # ========== Forward Propagation ==========
    
    def _forward_propagation(self, X):
        """
        İleri yayılım yapar.
        
        Args:
            X: Girdi matrisi (n_samples, n_features)
            
        Returns:
            A_final: Son katmanın çıktısı
            caches: Geri yayılım için gerekli değerler
        """
        caches = []
        A = X
        
        # Her katman için
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            # Linear transformation: Z = A_prev @ W + b
            Z = np.dot(A_prev, W) + b
            
            # Activation
            activation = self.activation_funcs[l-1]
            A = self._activate(Z, activation)
            
            # Cache for backward pass
            cache = {
                'A_prev': A_prev,
                'W': W,
                'b': b,
                'Z': Z,
                'activation': activation
            }
            caches.append(cache)
        
        return A, caches
    
    # ========== Loss Hesaplama ==========
    
    def _compute_loss(self, A_final, Y):
        """
        Cross-Entropy Loss hesaplar.
        
        Args:
            A_final: Modelin çıktısı (n_samples, n_classes)
            Y: Gerçek etiketler (one-hot encoded) (n_samples, n_classes)
            
        Returns:
            loss: Cross-entropy loss değeri
        """
        m = Y.shape[0]
        
        # Cross-entropy loss
        # Numerical stability için küçük epsilon ekle
        epsilon = 1e-8
        A_final = np.clip(A_final, epsilon, 1 - epsilon)
        
        loss = -np.sum(Y * np.log(A_final)) / m
        
        # L2 Regularization ekle
        if self.l2_lambda > 0:
            l2_loss = 0
            for l in range(1, self.L + 1):
                W = self.parameters[f'W{l}']
                l2_loss += np.sum(W ** 2)
            loss += (self.l2_lambda / (2 * m)) * l2_loss
        
        return loss
    
    # ========== Backward Propagation ==========
    
    def _backward_propagation(self, Y, caches):
        """
        Geri yayılım (Backpropagation) algoritması.
        
        Args:
            Y: Gerçek etiketler (one-hot encoded) (n_samples, n_classes)
            caches: İleri yayılımdan gelen cache'ler
            
        Returns:
            gradients: Her katman için hesaplanan gradyanlar
        """
        gradients = {}
        m = Y.shape[0]
        L = len(caches)
        
        # Son katmanın çıktısı
        cache_L = caches[L-1]
        A_L = self._activate(cache_L['Z'], cache_L['activation'])
        
        # Son katman için gradyan (Softmax + Cross-Entropy)
        # dL/dZ_L = A_L - Y
        dZ = A_L - Y
        
        # Geriye doğru git
        for l in reversed(range(1, L + 1)):
            cache = caches[l-1]
            A_prev = cache['A_prev']
            W = cache['W']
            
            # Gradyanları hesapla
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # L2 regularization gradient ekle
            if self.l2_lambda > 0:
                dW += (self.l2_lambda / m) * W
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            # Bir önceki katman için dZ hesapla (son katman değilse)
            if l > 1:
                dA_prev = np.dot(dZ, W.T)
                cache_prev = caches[l-2]
                dZ = self._activate_backward(dA_prev, cache_prev['Z'], 
                                            cache_prev['activation'])
        
        return gradients
    
    # ========== Parametre Güncelleme ==========
    
    def _update_parameters(self, gradients):
        """
        Gradient descent ile parametreleri günceller.
        
        Args:
            gradients: Hesaplanan gradyanlar
        """
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * gradients[f'db{l}']
    
    # ========== Tahmin ==========
    
    def predict(self, X):
        """
        Tahmin yapar.
        
        Args:
            X: Girdi matrisi (n_samples, n_features)
            
        Returns:
            Tahmin edilen sınıflar (n_samples,)
        """
        A_final, _ = self._forward_propagation(X)
        predictions = np.argmax(A_final, axis=1)
        return predictions
    
    # ========== Eğitim ==========
    
    def fit(self, X, y, epochs=100, batch_size=32):
        """
        Modeli eğitir ve her epoch'ta yield ile durumunu döndürür.
        
        Args:
            X: Eğitim verisi (n_samples, n_features)
            y: Etiketler (n_samples,)
            epochs: Epoch sayısı
            batch_size: Batch boyutu (mini-batch gradient descent için)
            
        Yields:
            (epoch, loss, model) her epoch için
        """
        n_samples = X.shape[0]
        
        # One-hot encoding for labels
        n_classes = self.layer_dims[-1]
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y.astype(int)] = 1
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            epoch_loss = 0
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
            
            # Ortalama loss
            avg_loss = epoch_loss / n_batches
            
            # Yield current state
            yield epoch + 1, avg_loss, self
