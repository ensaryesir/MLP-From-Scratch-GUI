"""
Single-Layer Neural Network Algorithms
Tek Katmanlı Sinir Ağı Algoritmaları

This module implements two classic single-layer learning algorithms:
Bu modül iki klasik tek katmanlı öğrenme algoritması içerir:

1. Perceptron (Rosenblatt, 1958)
   - First neural network learning algorithm
     İlk sinir ağı öğrenme algoritması
   - Uses step activation function
     Basamak aktivasyon fonksiyonu kullanır
   - Can only learn linearly separable patterns
     Sadece doğrusal olarak ayrılabilir örüntüleri öğrenebilir
   - Updates weights based on classification errors
     Sınıflandırma hatalarına göre ağırlıkları günceller

2. Delta Rule / ADALINE (Widrow-Hoff, 1960)
   - Adaptive Linear Neuron
     Uyarlanabilir Doğrusal Nöron
   - Uses linear activation function
     Doğrusal aktivasyon fonksiyonu kullanır
   - Minimizes Mean Squared Error (MSE)
     Ortalama Karesel Hata'yı (MSE) minimize eder
   - More stable learning than Perceptron
     Perceptron'dan daha kararlı öğrenme

Historical Significance / Tarihsel Önem:
    These algorithms laid the foundation for modern deep learning.
    Bu algoritmalar, modern derin öğrenmenin temelini attı.

    The Perceptron Convergence Theorem (1958) proved that the
    perceptron will always converge for linearly separable data.

    Perceptron Yakınsama Teoremi (1958), perceptron'un doğrusal olarak
    ayrılabilir veriler için her zaman yakınsayacağını kanıtladı.

Author: Developed for educational purposes
Date: 2024
"""

import numpy as np


class Perceptron:
    """
    Perceptron Algorithm (Rosenblatt, 1958)
    Perceptron Algoritması (Rosenblatt, 1958)

    The perceptron is the simplest form of a neural network, consisting of
    a single neuron with adjustable weights and a step activation function.
    It can learn to classify patterns that are linearly separable.

    Perceptron, tek bir nöron, ayarlanabilir ağırlıklar ve bir basamak
    aktivasyon fonksiyonundan oluşan en basit sinir ağı biçimidir.
    Doğrusal olarak ayrılabilir örüntüleri sınıflandırmayı öğrenebilir.

    Mathematical Model / Matematiksel Model:
        Output: y = step(w · x + b)
        where step(z) = 1 if z ≥ 0, else 0

        Çıktı: y = step(w · x + b)
        burada step(z) = z ≥ 0 ise 1, değilse 0

    Learning Rule (Perceptron Learning Algorithm):
    Öğrenme Kuralı (Perceptron Öğrenme Algoritması):
        Δw = η * (y_true - y_pred) * x
        Δb = η * (y_true - y_pred)

        where η is the learning rate
        burada η öğrenme oranıdır

    Key Properties / Temel Özellikler:
        - Guaranteed convergence for linearly separable data
          Doğrusal olarak ayrılabilir veriler için garanti edilmiş yakınsama
        - No convergence for non-linearly separable data (XOR problem)
          Doğrusal olarak ayrılamayan veriler için yakınsamaz (XOR problemi)
        - Online learning (updates after each sample)
          Çevrimiçi öğrenme (her örnekten sonra güncelleme)
        - Binary step function causes discontinuous updates
          İkili basamak fonksiyonu sürekli olmayan güncellemelere neden olur

    Attributes:
        learning_rate (float): Step size for weight updates / Ağırlık güncellemeleri için adım boyutu
        n_classes (int): Number of output classes / Çıktı sınıf sayısı
        weights (np.ndarray): Weight matrix, shape (n_features, n_classes)
                             Ağırlık matrisi, boyut (n_features, n_classes)
        bias (np.ndarray): Bias vector, shape (1, n_classes)
                          Sapma vektörü, boyut (1, n_classes)
    """

    def __init__(self, learning_rate=0.01, n_classes=2):
        """
        Initialize the Perceptron classifier.
        Perceptron sınıflandırıcısını başlatır.

        Args:
            learning_rate (float, optional): Learning rate for weight updates. Default: 0.01
                                            Typical range: 0.001 to 0.1

                                            Ağırlık güncellemeleri için öğrenme oranı. Varsayılan: 0.01
                                            Tipik aralık: 0.001 ile 0.1

            n_classes (int, optional): Number of output classes. Default: 2
                                      Çıktı sınıf sayısı. Varsayılan: 2
        """
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.weights = None
        self.bias = None

    def _initialize_parameters(self, n_features):
        """Ağırlıkları ve bias'ı başlatır."""
        # For multi-class classification, create one weight vector per class
        # Çok sınıflı sınıflandırma için, her sınıf için bir ağırlık vektörü oluştur
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))

    def _step_function(self, x):
        """
        Step (Heaviside) activation function.
        Basamak (Heaviside) aktivasyon fonksiyonu.

        Formula / Formül:
            step(x) = 1 if x ≥ 0, else 0

        This is the signature activation function of the Perceptron,
        creating a hard decision boundary.

        Bu, Perceptron'un imza aktivasyon fonksiyonudur ve
        sert bir karar sınırı oluşturur.

        Args:
            x (np.ndarray): Input values / Girdi değerleri

        Returns:
            np.ndarray: Binary output (0 or 1) / İkili çıktı (0 veya 1)
        """
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
            # Return zeros if model is not trained yet
            # Model henüz eğitilmemişse sıfırlar döndür
            return np.zeros(X.shape[0])

        # Compute weighted sum: z = w · x + b
        # Ağırlıklı toplamı hesapla: z = w · x + b
        z = np.dot(X, self.weights) + self.bias

        # For multi-class: select class with highest activation
        # Çok sınıflı için: en yüksek aktivasyona sahip sınıfı seç
        predictions = np.argmax(z, axis=1)
        return predictions

    def fit(self, X, y, epochs=100):
        """
        Train the Perceptron using the Perceptron Learning Algorithm.
        Perceptron Öğrenme Algoritmasını kullanarak Perceptron'u eğitir.
        
        The Perceptron Learning Algorithm:
        Perceptron Öğrenme Algoritması:
        
        For each training example:
        Her eğitim örneği için:
            1. Compute prediction: y_pred = step(w · x + b)
               Tahmin hesapla: y_pred = step(w · x + b)
            2. Compute error: error = y_true - y_pred
               Hata hesapla: hata = y_true - y_pred
            3. Update weights: w = w + η * error * x
               Ağırlıkları güncelle: w = w + η * hata * x
            4. Update bias: b = b + η * error
               Sapmayı güncelle: b = b + η * hata
        
        Convergence Theorem / Yakınsama Teoremi:
            If the data is linearly separable, the Perceptron algorithm
            is guaranteed to find a solution in finite time.
            
            Veri doğrusal olarak ayrılabilirse, Perceptron algoritması
            sonlu zamanda bir çözüm bulmayı garanti eder.
        
        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features)
                           Eğitim verisi, boyut (n_samples, n_features)
            y (np.ndarray): Training labels, shape (n_samples,)
                           Eğitim etiketleri, boyut (n_samples,)
            epochs (int, optional): Number of training epochs. Default: 100
                                   Eğitim epoch sayısı. Varsayılan: 100
            
        Yields:
            tuple: (epoch_number, average_error, self)
                epoch_number (int): Current epoch / Mevcut epoch
                average_error (float): Average classification error / Ortalama sınıflandırma hatası
                self (Perceptron): Model reference / Model referansı
        """
        n_samples, n_features = X.shape
        
        # Parametreleri başlat
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        # One-hot encoding for labels
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        
        for epoch in range(epochs):
            total_error = 0  # Accumulator for total classification errors / Toplam sınıflandırma hataları için toplayıcı
            
            # Online learning: Update weights after each sample
            # Çevrimiçi öğrenme: Her örnekten sonra ağırlıkları güncelle
            for i in range(n_samples):
                xi = X[i:i+1]  # Single training example / Tek eğitim örneği
                yi = y_onehot[i:i+1]  # True label (one-hot) / Gerçek etiket (one-hot)
                
                # STEP 1: Forward pass - Compute prediction
                # ADİM 1: İleri geçiş - Tahmin hesapla
                z = np.dot(xi, self.weights) + self.bias  # Weighted sum / Ağırlıklı toplam
                prediction_onehot = np.zeros_like(z)
                prediction_onehot[0, np.argmax(z)] = 1  # Apply step function / Basamak fonksiyonunu uygula
                
                # STEP 2: Compute classification error
                # ADİM 2: Sınıflandırma hatasını hesapla
                # error = y_true - y_pred (will be 0 if correct, non-zero if wrong)
                # hata = y_true - y_pred (doğruysa 0, yanlışsa sıfırdan farklı)
                error = yi - prediction_onehot
                total_error += np.sum(np.abs(error))  # Accumulate total error / Toplam hatayı biriktir
                
                # STEP 3 & 4: Update weights and biases using Perceptron rule
                # ADİM 3 & 4: Perceptron kuralını kullanarak ağırlıkları ve sapmaları güncelle
                # Δw = η * error * x
                self.weights += self.learning_rate * np.dot(xi.T, error)
                # Δb = η * error
                self.bias += self.learning_rate * error
            
            # Compute average error across all samples
            # Tüm örnekler boyunca ortalama hatayı hesapla
            avg_error = total_error / (n_samples * self.n_classes)
            
            # Yield current training state (Generator pattern)
            # Mevcut eğitim durumunu döndür (Generator deseni)
            yield epoch + 1, avg_error, self


class DeltaRule:
    """\n    Delta Rule / ADALINE (Adaptive Linear Neuron) - Widrow-Hoff, 1960\n    Delta Kuralı / ADALİNE (Uyarlanılabilir Doğrusal Nöron) - Widrow-Hoff, 1960\n    \n    ADALINE is an improvement over the Perceptron that uses a continuous\n    (linear) activation function and minimizes the Mean Squared Error (MSE)\n    rather than classification errors.\n    \n    ADALINE, Perceptron'ün geliştirilmiş halidir ve sürekli (doğrusal) bir\n    aktivasyon fonksiyonu kullanır ve sınıflandırma hataları yerine\n    Ortalama Karesel Hata'yı (MSE) minimize eder.\n    \n    Mathematical Model / Matematiksel Model:\n        Output: y = w · x + b  (linear activation)\n        Çıktı: y = w · x + b  (doğrusal aktivasyon)\n    \n    Learning Rule (Widrow-Hoff / Delta Rule):\n    Öğrenme Kuralı (Widrow-Hoff / Delta Kuralı):\n        Loss: L = (1/2m) * Σ(y_true - y_pred)²  (MSE)\n        \n        Gradient Descent Update:\n        Gradient Descent Güncellemesi:\n            Δw = -η * dL/dw = η * (1/m) * Σ(y_true - y_pred) * x\n            Δb = -η * dL/db = η * (1/m) * Σ(y_true - y_pred)\n    \n    Key Differences from Perceptron / Perceptron'dan Temel Farklar:\n        1. Continuous activation (linear) vs. discrete (step)\n           Sürekli aktivasyon (doğrusal) vs. ayrık (basamak)\n        2. Minimizes MSE vs. classification errors\n           MSE minimize eder vs. sınıflandırma hataları\n        3. Batch updates vs. online updates\n           Toplu güncellemeler vs. çevrimiçi güncellemeler\n        4. More stable gradient-based learning\n           Daha kararlı gradyan tabanlı öğrenme\n    \n    Advantages / Avantajlar:\n        - Smooth, continuous error surface\n          Düzgün, sürekli hata yüzeyi\n        - Well-defined gradient for optimization\n          Optimizasyon için iyi tanımlanmış gradyan\n        - Foundation for modern gradient descent methods\n          Modern gradient descent yöntemlerinin temeli\n    \n    Attributes:\n        learning_rate (float): Step size for gradient descent / Gradient descent için adım boyutu\n        n_classes (int): Number of output classes / Çıktı sınıf sayısı\n        weights (np.ndarray): Weight matrix / Ağırlık matrisi\n        bias (np.ndarray): Bias vector / Sapma vektörü\n    """
    
    def __init__(self, learning_rate=0.01, n_classes=2):
        """
        Initialize the Delta Rule (ADALINE) classifier.
        Delta Rule (ADALINE) sınıflandırıcısını başlatır.
        
        Args:
            learning_rate (float, optional): Learning rate for gradient descent. Default: 0.01\n                                            Gradient descent için öğrenme oranı. Varsayılan: 0.01\n            n_classes (int, optional): Number of output classes. Default: 2\n                                      Çıktı sınıf sayısı. Varsayılan: 2\n        """
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
        
    def _initialize_parameters(self, n_features):
        """Ağırlıkları ve bias'ı başlatır."""
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))
        
    def _activation(self, x):
        """
        Linear (identity) activation function.
        Doğrusal (kimlik) aktivasyon fonksiyonu.
        
        Formula / Formül: f(x) = x
        
        Unlike Perceptron's step function, this allows for gradient-based learning.
        Perceptron'un basamak fonksiyonunun aksine, bu gradyan tabanlı öğrenmeye izin verir.
        
        Args:
            x (np.ndarray): Input values / Girdi değerleri
            
        Returns:
            np.ndarray: Same as input (identity) / Girdiyle aynı (kimlik)
        """
        return x
    
    def predict(self, X):
        """
        Make predictions for input samples.
        Girdi örnekleri için tahminler yapar.
        
        Computes the weighted sum and applies step function to determine
        the predicted class for each sample.
        
        Ağırlıklı toplamı hesaplar ve her örnek için tahmin edilen
        sınıfı belirlemek üzere basamak fonksiyonunu uygular.
        
        Args:
            X (np.ndarray): Input feature matrix, shape (n_samples, n_features)
                           Girdi özellik matrisi, boyut (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,)
                       Tahmin edilen sınıf etiketleri, boyut (n_samples,)
        """
        if self.weights is None:
            # Return zeros if model is not trained yet
            # Model henüz eğitilmemişse sıfırlar döndür
            return np.zeros(X.shape[0])
        
        # Compute weighted sum and apply linear activation
        # Ağırlıklı toplamı hesapla ve doğrusal aktivasyon uygula
        z = np.dot(X, self.weights) + self.bias
        a = self._activation(z)  # Linear: a = z
        
        # For multi-class: select class with highest output
        # Çok sınıflı için: en yüksek çıktıya sahip sınıfı seç
        predictions = np.argmax(a, axis=1)
        return predictions
    
    def fit(self, X, y, epochs=100):
        """
        Train the ADALINE using batch gradient descent to minimize MSE.
        MSE'yi minimize etmek için batch gradient descent kullanarak ADALINE'i eğitir.
        
        The Widrow-Hoff Learning Rule (Delta Rule):
        Widrow-Hoff Öğrenme Kuralı (Delta Kuralı):
        
        For each epoch:
        Her epoch için:
            1. Forward pass: Compute predictions for all samples
               İleri geçiş: Tüm örnekler için tahminleri hesapla
            2. Compute error: error = y_true - y_pred
               Hatayı hesapla: hata = y_true - y_pred
            3. Compute MSE loss: L = mean(error²)
               MSE kaybını hesapla: L = mean(hata²)
            4. Compute gradients:
               Gradyanları hesapla:
               dW = -(2/m) * X^T @ error
               db = -(2/m) * sum(error)
            5. Update parameters:
               Parametreleri güncelle:
               W = W - η * dW
               b = b - η * db
        
        Batch Learning vs. Online Learning:
        Toplu Öğrenme vs. Çevrimiçi Öğrenme:
            ADALINE uses batch learning (updates after seeing all samples),\n            while Perceptron uses online learning (updates after each sample).\n            \n            ADALINE toplu öğrenme kullanır (tüm örnekleri gördükten sonra günceller),\n            Perceptron ise çevrimiçi öğrenme kullanır (her örnekten sonra günceller).\n        \n        Args:\n            X (np.ndarray): Training data, shape (n_samples, n_features)\n                           Eğitim verisi, boyut (n_samples, n_features)\n            y (np.ndarray): Training labels, shape (n_samples,)\n                           Eğitim etiketleri, boyut (n_samples,)\n            epochs (int, optional): Number of training epochs. Default: 100\n                                   Eğitim epoch sayısı. Varsayılan: 100\n            \n        Yields:\n            tuple: (epoch_number, mse_loss, self)\n                epoch_number (int): Current epoch / Mevcut epoch\n                mse_loss (float): Mean Squared Error / Ortalama Karesel Hata\n                self (DeltaRule): Model reference / Model referansı\n        """
        n_samples, n_features = X.shape
        
        # Parametreleri başlat
        if self.weights is None:
            self._initialize_parameters(n_features)
        
        # One-hot encoding for labels
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        
        for epoch in range(epochs):
            # STEP 1: Forward pass - Compute predictions for entire dataset
            # ADİM 1: İleri geçiş - Tüm veri seti için tahminleri hesapla
            # Batch processing: compute all predictions at once
            # Toplu işleme: tüm tahminleri bir kerede hesapla
            z = np.dot(X, self.weights) + self.bias  # Linear combination / Doğrusal kombinasyon
            a = self._activation(z)  # Linear activation: a = z / Doğrusal aktivasyon: a = z
            
            # STEP 2: Compute error and MSE loss
            # ADİM 2: Hatayı ve MSE kaybını hesapla
            # error = y_true - y_pred
            error = y_onehot - a
            # MSE = mean((y_true - y_pred)²)
            loss = np.mean(error ** 2)
            
            # STEP 3: Compute gradients using calculus
            # ADİM 3: Kalkülüs kullanarak gradyanları hesapla
            # 
            # For MSE loss L = (1/2m) * Σ(y - ŷ)²:
            # MSE kaybı L = (1/2m) * Σ(y - ŷ)² için:
            #   dL/dW = -(1/m) * X^T @ (y - ŷ)
            #   dL/db = -(1/m) * Σ(y - ŷ)
            #
            # Note: We use -2 instead of -1 for clearer gradient descent
            # Not: Daha net gradient descent için -1 yerine -2 kullanıyoruz
            dW = -2 * np.dot(X.T, error) / n_samples
            db = -2 * np.mean(error, axis=0, keepdims=True)
            
            # STEP 4: Update parameters using gradient descent
            # ADİM 4: Gradient descent kullanarak parametreleri güncelle
            # W = W - η * dW (move in opposite direction of gradient)
            # W = W - η * dW (gradyanın ters yönüne hareket et)
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            # Yield current training state (Generator pattern)
            # Mevcut eğitim durumunu döndür (Generator deseni)
            yield epoch + 1, loss, self
