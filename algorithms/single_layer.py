"""
Tek Katmanlı Sinir Ağı Algoritmaları
Perceptron ve Delta Rule (Adaline) implementasyonları.
"""

import numpy as np


class Perceptron:
    """
    Perceptron algoritması - Doğrusal olarak ayrılabilir problemler için.
    Binary classification için step fonksiyonu kullanır.
    """
    
    def __init__(self, learning_rate=0.01, n_classes=2):
        """
        Perceptron'u başlatır.
        
        Args:
            learning_rate: Öğrenme oranı
            n_classes: Sınıf sayısı
        """
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
        
    def _initialize_parameters(self, n_features):
        """Ağırlıkları ve bias'ı başlatır."""
        # Multi-class için her sınıf için bir weight vektörü
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))
        
    def _step_function(self, x):
        """Step aktivasyon fonksiyonu."""
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        """
        Tahmin yapar.
        
        Args:
            X: Girdi matrisi (n_samples, n_features)
            
        Returns:
            Tahmin edilen sınıflar
        """
        if self.weights is None:
            return np.zeros(X.shape[0])
        
        # Linear combination
        z = np.dot(X, self.weights) + self.bias
        
        # Multi-class için argmax
        predictions = np.argmax(z, axis=1)
        return predictions
    
    def fit(self, X, y, epochs=100):
        """
        Modeli eğitir ve her epoch'ta yield ile durumunu döndürür.
        
        Args:
            X: Eğitim verisi (n_samples, n_features)
            y: Etiketler (n_samples,)
            epochs: Epoch sayısı
            
        Yields:
            (epoch, loss, model) her epoch için
        """
        n_samples, n_features = X.shape
        
        # Parametreleri başlat
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        # One-hot encoding for labels
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        
        for epoch in range(epochs):
            total_error = 0
            
            # Her örnek için
            for i in range(n_samples):
                xi = X[i:i+1]
                yi = y_onehot[i:i+1]
                
                # Forward pass
                z = np.dot(xi, self.weights) + self.bias
                prediction_onehot = np.zeros_like(z)
                prediction_onehot[0, np.argmax(z)] = 1
                
                # Hata hesapla
                error = yi - prediction_onehot
                total_error += np.sum(np.abs(error))
                
                # Ağırlıkları güncelle (Perceptron öğrenme kuralı)
                self.weights += self.learning_rate * np.dot(xi.T, error)
                self.bias += self.learning_rate * error
            
            # Ortalama hata
            avg_error = total_error / (n_samples * self.n_classes)
            
            # Yield current state
            yield epoch + 1, avg_error, self


class DeltaRule:
    """
    Delta Rule (Adaline) - Widrow-Hoff öğrenme kuralı.
    Sürekli aktivasyon fonksiyonu (doğrusal) kullanır ve MSE minimize eder.
    """
    
    def __init__(self, learning_rate=0.01, n_classes=2):
        """
        Delta Rule'u başlatır.
        
        Args:
            learning_rate: Öğrenme oranı
            n_classes: Sınıf sayısı
        """
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
        
    def _initialize_parameters(self, n_features):
        """Ağırlıkları ve bias'ı başlatır."""
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))
        
    def _activation(self, x):
        """Linear aktivasyon fonksiyonu."""
        return x
    
    def predict(self, X):
        """
        Tahmin yapar.
        
        Args:
            X: Girdi matrisi (n_samples, n_features)
            
        Returns:
            Tahmin edilen sınıflar
        """
        if self.weights is None:
            return np.zeros(X.shape[0])
        
        # Linear combination
        z = np.dot(X, self.weights) + self.bias
        a = self._activation(z)
        
        # Multi-class için argmax
        predictions = np.argmax(a, axis=1)
        return predictions
    
    def fit(self, X, y, epochs=100):
        """
        Modeli eğitir ve her epoch'ta yield ile durumunu döndürür.
        
        Args:
            X: Eğitim verisi (n_samples, n_features)
            y: Etiketler (n_samples,)
            epochs: Epoch sayısı
            
        Yields:
            (epoch, loss, model) her epoch için
        """
        n_samples, n_features = X.shape
        
        # Parametreleri başlat
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        # One-hot encoding for labels
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        
        for epoch in range(epochs):
            # Forward pass (tüm veri için)
            z = np.dot(X, self.weights) + self.bias
            a = self._activation(z)
            
            # Hata hesapla (MSE)
            error = y_onehot - a
            loss = np.mean(error ** 2)
            
            # Gradyan hesapla
            dW = -2 * np.dot(X.T, error) / n_samples
            db = -2 * np.mean(error, axis=0, keepdims=True)
            
            # Parametreleri güncelle (Gradient Descent)
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            # Yield current state
            yield epoch + 1, loss, self
