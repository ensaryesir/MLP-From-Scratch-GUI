"""
Multi-Layer Perceptron (MLP) - Fully Connected Neural Network
Çok Katmanlı Perceptron (MLP) - Tam Bağlantılı Sinir Ağı

This module implements a Multi-Layer Perceptron from scratch using only NumPy.
The network supports multiple hidden layers, various activation functions, and
is trained using the backpropagation algorithm with gradient descent optimization.

Bu modül, yalnızca NumPy kullanarak sıfırdan bir Çok Katmanlı Perceptron uygulaması içerir.
Ağ, birden fazla gizli katmanı ve çeşitli aktivasyon fonksiyonlarını destekler ve
hata geri yayılımı (backpropagation) algoritması ile gradient descent optimizasyonu kullanılarak eğitilir.

Theoretical Background / Teorik Arka Plan:
    MLP is a feedforward artificial neural network that consists of at least three layers:
    an input layer, one or more hidden layers, and an output layer. Each node (neuron)
    uses a nonlinear activation function. MLP utilizes backpropagation for training.

    MLP, en az üç katmandan oluşan ileri beslemeli bir yapay sinir ağıdır:
    bir girdi katmanı, bir veya daha fazla gizli katman ve bir çıktı katmanı.
    Her düğüm (nöron) doğrusal olmayan bir aktivasyon fonksiyonu kullanır.
    MLP, eğitim için hata geri yayılımı kullanır.

Author: Developed for educational purposes
Date: 2024
"""

import numpy as np


class MLP:
    """
    Multi-Layer Perceptron (MLP) Classifier
    Çok Katmanlı Perceptron (MLP) Sınıflandırıcısı

    A feedforward artificial neural network implementation that uses backpropagation
    for training. This class supports multiple hidden layers, various activation
    functions (ReLU, Tanh, Sigmoid, Softmax), L2 regularization, and mini-batch
    gradient descent optimization.

    Eğitim için hata geri yayılımı kullanan bir ileri beslemeli yapay sinir ağı
    uygulaması. Bu sınıf, birden fazla gizli katmanı, çeşitli aktivasyon fonksiyonlarını
    (ReLU, Tanh, Sigmoid, Softmax), L2 düzenlileştirmeyi ve mini-batch gradient
    descent optimizasyonunu destekler.

    Mathematical Foundation / Matematiksel Temel:
        Forward Propagation: a[l] = g[l](W[l] * a[l-1] + b[l])
        where g[l] is the activation function of layer l

        Backward Propagation uses chain rule:
        dL/dW[l] = dL/da[l] * da[l]/dz[l] * dz[l]/dW[l]

    Attributes:
        layer_dims (list): Number of neurons in each layer including input and output.
                          Example: [2, 5, 3] means 2 inputs, 5 hidden neurons, 3 outputs.

                          Her katmandaki nöron sayısı (girdi ve çıktı dahil).
                          Örnek: [2, 5, 3] = 2 girdi, 5 gizli nöron, 3 çıktı.

        activation_funcs (list): Activation function name for each layer (except input).
                                Supported: 'relu', 'tanh', 'sigmoid', 'softmax', 'linear'
                                Example: ['relu', 'softmax'] for one hidden + one output layer

                                Her katman için aktivasyon fonksiyonu adı (girdi hariç).
                                Desteklenen: 'relu', 'tanh', 'sigmoid', 'softmax', 'linear'
                                Örnek: 1 gizli + 1 çıktı katmanı için ['relu', 'softmax']

        learning_rate (float): Step size for gradient descent optimization.
                              Typically ranges from 0.001 to 0.1

                              Gradient descent için öğrenme oranı. Varsayılan: 0.01
                              Genellikle 0.001 ile 0.1 arasında değişir

        l2_lambda (float): L2 regularization coefficient to prevent overfitting.
                          Higher values prevent overfitting but may underfit.
                          Typical range: 0.0001 to 0.1

                          L2 düzenlileştirme katsayısı. Varsayılan: 0.0
                          Yüksek değerler aşırı öğrenmeyi önler ama yetersiz öğrenmeye yol açabilir.
                          Tipik aralık: 0.0001 ile 0.1

        parameters (dict): Network weights and biases stored as {'W1', 'b1', 'W2', 'b2', ...}

                          Ağ ağırlıkları ve sapmalar {'W1', 'b1', 'W2', 'b2', ...} olarak saklanır.

        L (int): Number of layers in the network (excluding input layer).

                Ağdaki katman sayısı (girdi katmanı hariç).
    """

    def __init__(self, layer_dims, activation_funcs, learning_rate=0.01, l2_lambda=0.0):
        """
        Initialize the Multi-Layer Perceptron network.
        Çok Katmanlı Perceptron ağını başlatır.

        This constructor sets up the network architecture and initializes all parameters
        (weights and biases) using appropriate initialization schemes.

        Bu yapıcı, ağ mimarisini kurar ve tüm parametreleri (ağırlıklar ve sapmalar)
        uygun başlatma şemaları kullanarak başlatır.

        Args:
            layer_dims (list of int): Layer dimensions including input and output layers.
                                     Example: [2, 5, 3] creates a network with:
                                     - 2 input features
                                     - 5 neurons in hidden layer
                                     - 3 output classes

                                     Girdi ve çıktı katmanları dahil katman boyutları.
                                     Örnek: [2, 5, 3] şu şekilde bir ağ oluşturur:
                                     - 2 girdi özelliği
                                     - Gizli katmanda 5 nöron
                                     - 3 çıktı sınıfı

            activation_funcs (list of str): Activation function for each layer (excluding input).
                                           Supported: 'relu', 'tanh', 'sigmoid', 'softmax', 'linear'
                                           Example: ['relu', 'softmax'] for 1 hidden + 1 output layer

                                           Her katman için aktivasyon fonksiyonu (girdi hariç).
                                           Desteklenen: 'relu', 'tanh', 'sigmoid', 'softmax', 'linear'
                                           Örnek: 1 gizli + 1 çıktı katmanı için ['relu', 'softmax']

            learning_rate (float, optional): Learning rate for gradient descent. Default: 0.01
                                            Typically ranges from 0.001 to 0.1

                                            Gradient descent için öğrenme oranı. Varsayılan: 0.01
                                            Genellikle 0.001 ile 0.1 arasında değişir

            l2_lambda (float, optional): L2 regularization coefficient. Default: 0.0
                                        Higher values prevent overfitting but may underfit.
                                        Typical range: 0.0001 to 0.1

                                        L2 düzenlileştirme katsayısı. Varsayılan: 0.0
                                        Yüksek değerler aşırı öğrenmeyi önler ama yetersiz öğrenmeye yol açabilir.
                                        Tipik aralık: 0.0001 ile 0.1
        """
        self.layer_dims = layer_dims
        self.activation_funcs = activation_funcs
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.parameters = {}
        self.L = len(layer_dims) - 1  # Katman sayısı (girdi katmanı hariç)
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """
        Initialize network parameters using Xavier/He initialization.
        Ağırlıkları Xavier/He initialization yöntemi ile başlatır.
        
        Weight initialization is crucial for training deep neural networks. This method
        uses He initialization for ReLU activations and Xavier initialization for others.
        
        Ağırlık başlatma, derin sinir ağlarını eğitmek için kritik öneme sahiptir. Bu metod,
        ReLU aktivasyonları için He initialization, diğerleri için Xavier initialization kullanır.
        
        Mathematical Foundation / Matematiksel Temel:
            He Init (for ReLU):  W ~ N(0, sqrt(2/n_in))
            Xavier Init (others): W ~ N(0, sqrt(1/n_in))
            
            where n_in is the number of input neurons to the layer
            burada n_in katmana giren nöron sayısıdır
            
        These initialization schemes help:
        Bu başlatma şemaları şunlara yardımcı olur:
            1. Prevent vanishing/exploding gradients
               Kaybolan/patlayan gradyanları önler
            2. Ensure proper signal propagation through layers
               Katmanlar boyunca doğru sinyal yayılımını sağlar
            3. Speed up convergence during training
               Eğitim sırasında yakınsamayı hızlandırır
        """
        for l in range(1, self.L + 1):
            # Iterate through each layer to initialize weights and biases
            # Her katman için ağırlık ve sapmaları başlat
            
            # He initialization for ReLU activation (prevents dying ReLU problem)
            # ReLU aktivasyonu için He initialization (ölü ReLU problemini önler)
            if self.activation_funcs[l-1] == 'relu':
                # He initialization: multiply by sqrt(2/n_in) for better ReLU performance
                # He başlatma: ReLU performansı için sqrt(2/n_in) ile çarp
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l-1], self.layer_dims[l]
                ) * np.sqrt(2.0 / self.layer_dims[l-1])
            else:
                # Xavier/Glorot initialization: multiply by sqrt(1/n_in) for tanh/sigmoid
                # Xavier/Glorot başlatma: tanh/sigmoid için sqrt(1/n_in) ile çarp
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l-1], self.layer_dims[l]
                ) * np.sqrt(1.0 / self.layer_dims[l-1])
            
            # Initialize biases to zero (common practice)
            # Sapmaları sıfıra başlat (yaygın uygulama)
            self.parameters[f'b{l}'] = np.zeros((1, self.layer_dims[l]))
    
    # ==============================================================================
    #                         ACTIVATION FUNCTIONS
    #                         AKTİVASYON FONKSİYONLARI
    # ==============================================================================
    # Activation functions introduce non-linearity to the network, enabling it
    # to learn complex patterns. Each function has its forward pass and backward
    # pass (derivative) implementation for backpropagation.
    #
    # Aktivasyon fonksiyonları ağa doğrusal olmayan özellik kazandırarak
    # karmaşık örüntüleri öğrenmesini sağlar. Her fonksiyonun ileri geçiş ve
    # geri geçiş (türev) uygulaması backpropagation için mevcuttur.
    # ==============================================================================
    
    def _relu(self, Z):
        """
        ReLU (Rectified Linear Unit) activation function.
        ReLU (Doğrultulmuş Doğrusal Birim) aktivasyon fonksiyonu.
        
        Formula / Formül: ReLU(z) = max(0, z)
        
        Properties / Özellikler:
            - Computationally efficient / Hesaplama açısından verimli
            - Helps with vanishing gradient problem / Kaybolan gradyan problemine yardımcı
            - Can cause "dying ReLU" if many neurons output 0 / Çok nöron 0 çıkarsa "ölü ReLU" sorununa yol açabilir
        
        Args:
            Z (np.ndarray): Pre-activation values / Aktivasyon öncesi değerler
            
        Returns:
            np.ndarray: Activated values / Aktive edilmiş değerler
        """
        return np.maximum(0, Z)
    
    def _relu_backward(self, dA, Z):
        """
        Backward pass for ReLU activation (derivative).
        ReLU aktivasyonu için geri geçiş (türev).
        
        Derivative / Türev:
            dReLU(z)/dz = 1 if z > 0, else 0
        
        Args:
            dA (np.ndarray): Gradient of loss with respect to activation / Aktivasyona göre kayıp gradyanı
            Z (np.ndarray): Pre-activation values from forward pass / İleri geçişten gelen aktivasyon öncesi değerler
            
        Returns:
            np.ndarray: Gradient of loss with respect to Z / Z'ye göre kayıp gradyanı
        """
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ
    
    def _tanh(self, Z):
        """
        Hyperbolic Tangent (Tanh) activation function.
        Hiperbolik Tanjant (Tanh) aktivasyon fonksiyonu.
        
        Formula / Formül: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
        Range / Aralık: (-1, 1)
        
        Properties / Özellikler:
            - Zero-centered output / Sıfır merkezli çıktı
            - Stronger gradients than sigmoid / Sigmoid'den daha güçlü gradyanlar
            - Can still suffer from vanishing gradients / Yine de kaybolan gradyan sorunu yaşayabilir
        
        Args:
            Z (np.ndarray): Pre-activation values / Aktivasyon öncesi değerler
            
        Returns:
            np.ndarray: Activated values in range (-1, 1) / (-1, 1) aralığında aktive edilmiş değerler
        """
        return np.tanh(Z)
    
    def _tanh_backward(self, dA, Z):
        """
        Backward pass for Tanh activation (derivative).
        Tanh aktivasyonu için geri geçiş (türev).
        
        Derivative / Türev:
            dtanh(z)/dz = 1 - tanh²(z)
        
        This elegant derivative makes backpropagation efficient.
        Bu zarif türev, backpropagation'ı verimli hale getirir.
        
        Args:
            dA (np.ndarray): Gradient of loss with respect to activation / Aktivasyona göre kayıp gradyanı
            Z (np.ndarray): Pre-activation values from forward pass / İleri geçişten gelen aktivasyon öncesi değerler
            
        Returns:
            np.ndarray: Gradient of loss with respect to Z / Z'ye göre kayıp gradyanı
        """
        A = np.tanh(Z)
        dZ = dA * (1 - A ** 2)
        return dZ
    
    def _sigmoid(self, Z):
        """
        Sigmoid (Logistic) activation function.
        Sigmoid (Lojistik) aktivasyon fonksiyonu.
        
        Formula / Formül: σ(z) = 1 / (1 + e^(-z))
        Range / Aralık: (0, 1)
        
        Properties / Özellikler:
            - Smooth gradient / Düzgün gradyan
            - Output interpretable as probability / Çıktı olasılık olarak yorumlanabilir
            - Suffers from vanishing gradient problem / Kaybolan gradyan probleminden muzdarip
            - Not zero-centered / Sıfır merkezli değil
        
        Args:
            Z (np.ndarray): Pre-activation values / Aktivasyon öncesi değerler
            
        Returns:
            np.ndarray: Activated values in range (0, 1) / (0, 1) aralığında aktive edilmiş değerler
        """
        # Numerical stability için clip
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def _sigmoid_backward(self, dA, Z):
        """
        Backward pass for Sigmoid activation (derivative).
        Sigmoid aktivasyonu için geri geçiş (türev).
        
        Derivative / Türev:
            dσ(z)/dz = σ(z) * (1 - σ(z))
        
        This derivative has a nice property: it can be computed using
        only the forward pass output, making it computationally efficient.
        
        Bu türevin güzel bir özelliği var: sadece ileri geçiş çıktısı kullanılarak
        hesaplanabilir, bu da hesaplama açısından verimli hale getirir.
        
        Args:
            dA (np.ndarray): Gradient of loss with respect to activation / Aktivasyona göre kayıp gradyanı
            Z (np.ndarray): Pre-activation values from forward pass / İleri geçişten gelen aktivasyon öncesi değerler
            
        Returns:
            np.ndarray: Gradient of loss with respect to Z / Z'ye göre kayıp gradyanı
        """
        A = self._sigmoid(Z)
        dZ = dA * A * (1 - A)
        return dZ
    
    def _softmax(self, Z):
        """
        Softmax activation function for multi-class classification.
        Çok sınıflı sınıflandırma için Softmax aktivasyon fonksiyonu.
        
        Formula / Formül:
            softmax(z_i) = e^(z_i) / Σ(e^(z_j)) for all j
        
        Properties / Özellikler:
            - Converts raw scores to probabilities / Ham skorları olasılıklara dönüştürür
            - Output sums to 1.0 / Çıktı toplamı 1.0'a eşittir
            - Used in output layer for multi-class problems / Çok sınıflı problemlerde çıktı katmanında kullanılır
        
        Numerical Stability / Sayısal Stabilite:
            To prevent overflow, we subtract max(Z) before exponentiation.
            Taşmayı önlemek için üstelleme öncesi max(Z) çıkarılır.
        
        Args:
            Z (np.ndarray): Pre-activation values, shape (batch_size, num_classes)
                           Aktivasyon öncesi değerler, boyut (batch_size, num_classes)
            
        Returns:
            np.ndarray: Probability distribution over classes / Sınıflar üzerinde olasılık dağılımı
        """
        # Subtract max for numerical stability (prevents overflow in exp)
        # Sayısal stabilite için max çıkar (üstellemede taşmayı önler)
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def _softmax_backward(self, dA, Z):
        """
        Backward pass for Softmax + Cross-Entropy (combined derivative).
        Softmax + Cross-Entropy için geri geçiş (birleşik türev).
        
        When Softmax is used with Cross-Entropy loss, the combined derivative
        simplifies beautifully to: dL/dZ = y_pred - y_true
        
        Softmax, Cross-Entropy kayıp fonksiyonu ile kullanıldığında,
        birleşik türev güzel bir şekilde basitleşir: dL/dZ = y_pred - y_true
        
        Mathematical Derivation / Matematiksel Türetme:
            L = -Σ(y_true * log(y_pred))
            dL/dZ = y_pred - y_true
        
        This elegant result makes backpropagation very efficient for
        classification tasks.
        
        Bu zarif sonuç, sınıflandırma görevleri için backpropagation'ı
        çok verimli hale getirir.
        
        Args:
            dA (np.ndarray): Already contains (y_pred - y_true) from loss calculation
                            Kayıp hesaplamasından gelen (y_pred - y_true) içerir
            Z (np.ndarray): Not used, kept for consistency / Kullanılmaz, tutarlılık için tutulur
            
        Returns:
            np.ndarray: Gradient (simply returns dA) / Gradyan (sadece dA'yı döndürür)
        """
        return dA
    
    def _linear(self, Z):
        """
        Linear (Identity) activation function.
        Doğrusal (Kimlik) aktivasyon fonksiyonu.
        
        Formula / Formül: f(z) = z
        
        Properties / Özellikler:
            - No transformation applied / Hiçbir dönüşüm uygulanmaz
            - Used for regression problems / Regresyon problemleri için kullanılır
            - Output can be any real number / Çıktı herhangi bir gerçel sayı olabilir
        
        Args:
            Z (np.ndarray): Pre-activation values / Aktivasyon öncesi değerler
            
        Returns:
            np.ndarray: Same as input (identity) / Girdiyle aynı (kimlik)
        """
        return Z
    
    def _linear_backward(self, dA, Z):
        """
        Backward pass for Linear activation (derivative).
        Doğrusal aktivasyon için geri geçiş (türev).
        
        Derivative / Türev:
            df(z)/dz = 1
        
        Args:
            dA (np.ndarray): Gradient of loss with respect to activation / Aktivasyona göre kayıp gradyanı
            Z (np.ndarray): Not used, kept for consistency / Kullanılmaz, tutarlılık için tutulur
            
        Returns:
            np.ndarray: Gradient (simply returns dA) / Gradyan (sadece dA'yı döndürür)
        """
        return dA
    
    def _activate(self, Z, activation):
        """
        Apply the specified activation function.
        Belirtilen aktivasyon fonksiyonunu uygular.
        
        This is a dispatcher method that calls the appropriate activation function
        based on the string identifier.
        
        Bu, string tanımlayıcıya göre uygun aktivasyon fonksiyonunu çağıran bir
        yönlendirici metoddur.
        
        Args:
            Z (np.ndarray): Pre-activation values / Aktivasyon öncesi değerler
            activation (str): Activation function name ('relu', 'tanh', 'sigmoid', 'softmax', 'linear')
                             Aktivasyon fonksiyonu adı
                             
        Returns:
            np.ndarray: Activated output / Aktive edilmiş çıktı
            
        Raises:
            ValueError: If activation function is not recognized / Bilinmeyen aktivasyon fonksiyonu
        """
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
        """
        Compute the derivative of the specified activation function.
        Belirtilen aktivasyon fonksiyonunun türevini hesaplar.
        
        This dispatcher method calls the appropriate backward pass function
        for the given activation, implementing the chain rule.
        
        Bu yönlendirici metod, verilen aktivasyon için uygun geri geçiş fonksiyonunu
        çağırarak zincir kuralını uygular.
        
        Args:
            dA (np.ndarray): Gradient flowing from the next layer / Sonraki katmandan gelen gradyan
            Z (np.ndarray): Pre-activation values from forward pass / İleri geçişten gelen aktivasyon öncesi değerler
            activation (str): Activation function name / Aktivasyon fonksiyonu adı
                             
        Returns:
            np.ndarray: Gradient with respect to Z (dL/dZ) / Z'ye göre gradyan (dL/dZ)
            
        Raises:
            ValueError: If activation function is not recognized / Bilinmeyen aktivasyon fonksiyonu
        """
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
            raise ValueError(f"Unknown activation function / Bilinmeyen aktivasyon fonksiyonu: {activation}")
    
    # ==============================================================================
    #                         FORWARD PROPAGATION
    #                         İLERİ YAYILIM
    # ==============================================================================
    
    def _forward_propagation(self, X):
        """
        Perform forward propagation through the entire network.
        Tüm ağ boyunca ileri yayılım gerçekleştirir.
        
        This method implements the forward pass of the neural network, computing
        the output for each layer sequentially. For each layer l:
            Z[l] = W[l] · A[l-1] + b[l]  (Linear transformation)
            A[l] = g[l](Z[l])            (Activation function)
        
        Bu metod, sinir ağının ileri geçişini uygular ve her katmanın çıktısını
        sırayla hesaplar. Her l katmanı için:
            Z[l] = W[l] · A[l-1] + b[l]  (Doğrusal dönüşüm)
            A[l] = g[l](Z[l])            (Aktivasyon fonksiyonu)
        
        The method also stores intermediate values (caches) needed for backpropagation.
        Metod ayrıca backpropagation için gereken ara değerleri (cache'leri) saklar.
        
        Args:
            X (np.ndarray): Input data matrix, shape (n_samples, n_features)
                           Girdi veri matrisi, boyut (n_samples, n_features)
                           where n_samples is batch size and n_features is number of input features
                           burada n_samples batch boyutu ve n_features girdi özellik sayısıdır
            
        Returns:
            tuple: (A_final, caches)
                A_final (np.ndarray): Output of the final layer, shape (n_samples, n_output)
                                     Son katmanın çıktısı, boyut (n_samples, n_output)
                caches (list): List of dictionaries containing intermediate values for each layer:
                              Her katman için ara değerleri içeren sözlük listesi:
                              - A_prev: Activation from previous layer / Önceki katmanın aktivasyonu
                              - W: Weight matrix / Ağırlık matrisi
                              - b: Bias vector / Sapma vektörü
                              - Z: Pre-activation values / Aktivasyon öncesi değerler
                              - activation: Activation function name / Aktivasyon fonksiyonu adı
        """
        caches = []
        A = X  # Initial activation is the input itself / İlk aktivasyon girdinin kendisidir
        
        # Iterate through each layer in the network
        # Ağdaki her katman için döngü
        for l in range(1, self.L + 1):
            A_prev = A  # Save previous layer's activation / Önceki katmanın aktivasyonunu kaydet
            W = self.parameters[f'W{l}']  # Weight matrix of layer l / l. katmanın ağırlık matrisi
            b = self.parameters[f'b{l}']  # Bias vector of layer l / l. katmanın sapma vektörü
            
            # Linear transformation: Z = A_prev @ W + b
            # Doğrusal dönüşüm: Z = A_prev @ W + b
            # Matrix dimensions: (m, n[l-1]) @ (n[l-1], n[l]) + (1, n[l]) = (m, n[l])
            # Matris boyutları: (m, n[l-1]) @ (n[l-1], n[l]) + (1, n[l]) = (m, n[l])
            Z = np.dot(A_prev, W) + b
            
            # Apply non-linear activation function
            # Doğrusal olmayan aktivasyon fonksiyonunu uygula
            activation = self.activation_funcs[l-1]
            A = self._activate(Z, activation)
            
            # Store values needed for backpropagation (backward pass)
            # Backpropagation için gereken değerleri sakla (geri geçiş)
            cache = {
                'A_prev': A_prev,
                'W': W,
                'b': b,
                'Z': Z,
                'activation': activation
            }
            caches.append(cache)
        
        return A, caches
    
    # ==============================================================================
    #                         LOSS COMPUTATION
    #                         KAYIP HESAPLAMA
    # ==============================================================================
    
    def _compute_loss(self, A_final, Y):
        """
        Compute Cross-Entropy Loss with optional L2 regularization.
        Opsiyonel L2 düzenlileştirme ile Cross-Entropy Kaybını hesaplar.
        
        Cross-Entropy loss is commonly used for classification tasks. It measures
        the difference between predicted probability distribution and true distribution.
        
        Cross-Entropy kaybı sınıflandırma görevleri için yaygın olarak kullanılır.
        Tahmin edilen olasılık dağılımı ile gerçek dağılım arasındaki farkı ölçer.
        
        Formula / Formül:
            L = -(1/m) * Σ Σ (y_true * log(y_pred))
            
        With L2 Regularization / L2 Düzenlileştirme ile:
            L_total = L + (λ/(2m)) * Σ ||W||²
            
        where λ is the regularization coefficient (l2_lambda)
        burada λ düzenlileştirme katsayısıdır (l2_lambda)
        
        Args:
            A_final (np.ndarray): Model's output (predicted probabilities), shape (n_samples, n_classes)
                                 Modelin çıktısı (tahmin edilen olasılıklar), boyut (n_samples, n_classes)
            Y (np.ndarray): True labels (one-hot encoded), shape (n_samples, n_classes)
                           Gerçek etiketler (one-hot kodlanmış), boyut (n_samples, n_classes)
                           
        Returns:
            float: Cross-entropy loss value (scalar) / Cross-entropy kayıp değeri (skaler)
        """
        m = Y.shape[0]  # Number of training examples / Eğitim örneği sayısı
        
        # Compute Cross-Entropy Loss
        # Cross-Entropy Kaybını hesapla
        
        # Add small epsilon to prevent log(0) which would cause NaN
        # log(0)'ı önlemek için küçük epsilon ekle (NaN'a yol açar)
        epsilon = 1e-8
        A_final = np.clip(A_final, epsilon, 1 - epsilon)
        
        # Cross-entropy formula: L = -(1/m) * Σ(y * log(ŷ))
        # Cross-entropy formülü: L = -(1/m) * Σ(y * log(ŷ))
        loss = -np.sum(Y * np.log(A_final)) / m
        
        # Add L2 Regularization term (weight decay) to prevent overfitting
        # Aşırı öğrenmeyi önlemek için L2 Düzenlileştirme terimi ekle (ağırlık azalması)
        if self.l2_lambda > 0:
            l2_loss = 0
            # Sum of squared weights across all layers
            # Tüm katmanlardaki ağırlıkların karelerinin toplamı
            for l in range(1, self.L + 1):
                W = self.parameters[f'W{l}']
                l2_loss += np.sum(W ** 2)  # Frobenius norm squared / Frobenius norm karesi
            # Add regularization term: (λ/(2m)) * Σ||W||²
            # Düzenlileştirme terimini ekle: (λ/(2m)) * Σ||W||²
            loss += (self.l2_lambda / (2 * m)) * l2_loss
        
        return loss
    
    # ==============================================================================
    #                         BACKWARD PROPAGATION (BACKPROPAGATION)
    #                         GERİ YAYILIM (HATA GERİ YAYILIMI)
    # ==============================================================================
    
    def _backward_propagation(self, Y, caches):
        """
        Perform backward propagation (backpropagation) through the entire network.
        Tüm ağ boyunca geri yayılım (backpropagation) gerçekleştirir.
        
        This is the heart of neural network training. Backpropagation computes the
        gradient of the loss function with respect to each parameter (weights and biases)
        using the chain rule of calculus. These gradients are then used to update
        the parameters via gradient descent.
        
        Bu, sinir ağı eğitiminin kalbidir. Backpropagation, kalkülüsün zincir kuralını
        kullanarak kayıp fonksiyonunun her parametreye (ağırlık ve sapmalara) göre
        gradyanını hesaplar. Bu gradyanlar daha sonra gradient descent ile parametreleri
        güncellemek için kullanılır.
        
        ========================================================================
        MATHEMATICAL FOUNDATION: CHAIN RULE / MATEMATİKSEL TEMEL: ZİNCİR KURALI
        ========================================================================
        
        The chain rule allows us to compute derivatives of composite functions.
        Zincir kuralı, bileşik fonksiyonların türevlerini hesaplamamiza izin verir.
        
        For each layer l, we compute three gradients:
        Her l katmanı için, üç gradyan hesaplanır:
        
        1. dZ[l] = dL/dZ[l] - Gradient with respect to pre-activation
                             Aktivasyon öncesi değere göre gradyan
           
           For output layer with Softmax + Cross-Entropy:
           Softmax + Cross-Entropy ile çıktı katmanı için:
               dZ[L] = A[L] - Y  (elegant simplification!)
                              (zarif basitleştirme!)
           
           For hidden layers:
           Gizli katmanlar için:
               dZ[l] = dA[l] * g'(Z[l])
               where g'(Z[l]) is the derivative of activation function
               burada g'(Z[l]) aktivasyon fonksiyonunun türevidir
        
        2. dW[l] = dL/dW[l] - Gradient with respect to weights
                             Ağırlıklara göre gradyan
           
           Using chain rule / Zincir kuralı kullanarak:
               dL/dW[l] = dL/dZ[l] * dZ[l]/dW[l]
                        = dZ[l] * A[l-1]^T / m
           
           Matrix form / Matris formu:
               dW[l] = (1/m) * A[l-1]^T @ dZ[l]
           
           Dimensions / Boyutlar:
               (n[l-1], m) @ (m, n[l]) = (n[l-1], n[l])  ✓ Correct!
        
        3. db[l] = dL/db[l] - Gradient with respect to biases
                             Sapmalara göre gradyan
           
           Using chain rule / Zincir kuralı kullanarak:
               dL/db[l] = dL/dZ[l] * dZ[l]/db[l]
                        = sum(dZ[l]) / m
           
           We sum across all examples in the batch:
           Batch'teki tüm örnekler boyunca topluyoruz:
               db[l] = (1/m) * Σ dZ[l]
        
        4. dA[l-1] - Gradient flowing to previous layer
                    Önceki katmana akan gradyan
           
           Using chain rule / Zincir kuralı kullanarak:
               dA[l-1] = dZ[l] @ W[l]^T
           
           This gradient is then used to compute dZ[l-1] in the next iteration.
           Bu gradyan daha sonra bir sonraki iterasyonda dZ[l-1]'i hesaplamak için kullanılır.
        
        ========================================================================
        L2 REGULARIZATION GRADIENT / L2 DÜZENLİLEŞTİRME GRADYANI
        ========================================================================
        
        When using L2 regularization, the weight gradient includes an extra term:
        L2 düzenlileştirme kullanıldığında, ağırlık gradyanı ekstra bir terim içerir:
        
            dW[l] = dW[l]_cross_entropy + (λ/m) * W[l]
        
        This encourages smaller weights, preventing overfitting.
        Bu, daha küçük ağırlıkları teşvik eder ve aşırı öğrenmeyi önler.
        
        ========================================================================
        
        Args:
            Y (np.ndarray): True labels (one-hot encoded), shape (n_samples, n_classes)
                           Gerçek etiketler (one-hot kodlanmış), boyut (n_samples, n_classes)
            caches (list): Cached values from forward propagation
                          İleri yayılımdan gelen önbelleklenmiş değerler
                          
        Returns:
            dict: Dictionary containing gradients for all parameters
                  Tüm parametreler için gradyanları içeren sözlük
                  Keys: 'dW1', 'db1', 'dW2', 'db2', ...
        """
        gradients = {}
        m = Y.shape[0]  # Number of training examples / Eğitim örneği sayısı (batch size)
        L = len(caches)  # Number of layers / Katman sayısı
        
        # ======================================================================
        # STEP 1: Compute gradient for the output layer
        # ADİM 1: Çıktı katmanı için gradyanı hesapla
        # ======================================================================
        
        cache_L = caches[L-1]  # Cache of the last layer / Son katmanın cache'i
        A_L = self._activate(cache_L['Z'], cache_L['activation'])  # Output predictions / Çıktı tahminleri
        
        # For Softmax activation + Cross-Entropy loss, the gradient simplifies beautifully:
        # Softmax aktivasyonu + Cross-Entropy kaybı için gradyan güzel bir şekilde basitleşir:
        #
        #   dL/dZ[L] = A[L] - Y
        #
        # This is one of the most elegant results in neural network theory!
        # Bu, sinir ağı teorisindeki en zarif sonuçlardan biridir!
        #
        # Derivation: Starting from L = -Σ(y * log(a)), and using the chain rule
        # Türetme: L = -Σ(y * log(a))'dan başlayarak ve zincir kuralını kullanarak
        # through softmax, we get this simple form.
        # softmax'ten geçerek, bu basit formu elde ederiz.
        dZ = A_L - Y  # Shape: (m, n_classes) / Boyut: (m, n_classes)
        
        # ======================================================================
        # STEP 2: Iterate backwards through all layers
        # ADİM 2: Tüm katmanlar boyunca geriye doğru ilerle
        # ======================================================================
        
        for l in reversed(range(1, L + 1)):  # From last layer to first / Son katmandan ilke
            cache = caches[l-1]  # Get cached values for layer l / l. katman için cache'lenmiş değerleri al
            A_prev = cache['A_prev']  # Activation from previous layer / Önceki katmandan gelen aktivasyon
            W = cache['W']  # Weight matrix of current layer / Mevcut katmanın ağırlık matrisi
            
            # ------------------------------------------------------------------
            # Compute dW[l]: Gradient of loss with respect to weights
            # dW[l]'yi hesapla: Ağırlıklara göre kayıp gradyanı
            # ------------------------------------------------------------------
            # 
            # Chain rule application / Zincir kuralı uygulaması:
            #   dL/dW[l] = dL/dZ[l] * dZ[l]/dW[l]
            #
            # Since Z[l] = W[l] · A[l-1] + b[l], we have:
            # Z[l] = W[l] · A[l-1] + b[l] olduğu için:
            #   dZ[l]/dW[l] = A[l-1]
            #
            # Therefore / Dolayısıyla:
            #   dW[l] = (1/m) * A[l-1]^T @ dZ[l]
            #
            # Matrix dimensions / Matris boyutları:
            #   (n[l-1], m) @ (m, n[l]) = (n[l-1], n[l])  ✓
            #
            dW = np.dot(A_prev.T, dZ) / m
            
            # ------------------------------------------------------------------
            # Compute db[l]: Gradient of loss with respect to biases
            # db[l]'yi hesapla: Sapmalara göre kayıp gradyanı
            # ------------------------------------------------------------------
            #
            # Chain rule application / Zincir kuralı uygulaması:
            #   dL/db[l] = dL/dZ[l] * dZ[l]/db[l]
            #
            # Since Z[l] = W[l] · A[l-1] + b[l], we have:
            # Z[l] = W[l] · A[l-1] + b[l] olduğu için:
            #   dZ[l]/db[l] = 1
            #
            # Therefore / Dolayısıyla:
            #   db[l] = (1/m) * Σ dZ[l]  (sum across batch dimension)
            #                            (batch boyutu boyunca toplam)
            #
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # ------------------------------------------------------------------
            # Add L2 Regularization gradient to dW (if enabled)
            # dW'ye L2 Düzenlileştirme gradyanını ekle (aktifse)
            # ------------------------------------------------------------------
            #
            # L2 regularization adds (λ/(2m)) * Σ||W||² to the loss.
            # L2 düzenlileştirme, kayba (λ/(2m)) * Σ||W||² ekler.
            #
            # Its derivative with respect to W is:
            # W'ye göre türevi:
            #   d/dW [(λ/(2m)) * ||W||²] = (λ/m) * W
            #
            if self.l2_lambda > 0:
                dW += (self.l2_lambda / m) * W  # Add regularization term / Düzenlileştirme terimini ekle
            
            # Store computed gradients / Hesaplanan gradyanları sakla
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
            
            # ------------------------------------------------------------------
            # Compute dA[l-1]: Gradient flowing to the previous layer
            # dA[l-1]'yi hesapla: Önceki katmana akan gradyan
            # ------------------------------------------------------------------
            #
            # We need this gradient to compute dZ[l-1] in the next iteration.
            # Bir sonraki iterasyonda dZ[l-1]'i hesaplamak için bu gradyana ihtiyaç var.
            #
            if l > 1:  # Not needed for the first layer / İlk katman için gerekli değil
                # Chain rule application / Zincir kuralı uygulaması:
                #   dL/dA[l-1] = dL/dZ[l] * dZ[l]/dA[l-1]
                #
                # Since Z[l] = W[l] · A[l-1] + b[l], we have:
                # Z[l] = W[l] · A[l-1] + b[l] olduğu için:
                #   dZ[l]/dA[l-1] = W[l]^T
                #
                # Therefore / Dolayısıyla:
                #   dA[l-1] = dZ[l] @ W[l]^T
                #
                # Matrix dimensions / Matris boyutları:
                #   (m, n[l]) @ (n[l], n[l-1]) = (m, n[l-1])  ✓
                #
                dA_prev = np.dot(dZ, W.T)
                
                # ------------------------------------------------------------------
                # Compute dZ[l-1] for the previous layer using activation derivative
                # Aktivasyon türevini kullanarak önceki katman için dZ[l-1]'yi hesapla
                # ------------------------------------------------------------------
                #
                # Chain rule application / Zincir kuralı uygulaması:
                #   dL/dZ[l-1] = dL/dA[l-1] * dA[l-1]/dZ[l-1]
                #
                # where dA[l-1]/dZ[l-1] = g'(Z[l-1]) is the activation derivative
                # burada dA[l-1]/dZ[l-1] = g'(Z[l-1]) aktivasyon türevidir
                #
                # Therefore / Dolayısıyla:
                #   dZ[l-1] = dA[l-1] * g'(Z[l-1])
                #            (element-wise multiplication)
                #            (eleman bazında çarpma)
                #
                cache_prev = caches[l-2]  # Cache of previous layer / Önceki katmanın cache'i
                dZ = self._activate_backward(dA_prev, cache_prev['Z'], 
                                            cache_prev['activation'])
        
        return gradients
    
    # ==============================================================================
    #                         PARAMETER UPDATE
    #                         PARAMETRE GÜNCELLEME
    # ==============================================================================
    
    def _update_parameters(self, gradients):
        """
        Update network parameters using gradient descent optimization.
        Gradient descent optimizasyonu kullanarak ağ parametrelerini günceller.
        
        This method implements the parameter update step of gradient descent:
        Bu metod, gradient descent'in parametre güncelleme adımını uygular:
        
            θ_new = θ_old - α * dθ
        
        where:
            θ represents parameters (W and b)
            α is the learning rate
            dθ is the gradient computed by backpropagation
        
        burada:
            θ parametreleri temsil eder (W ve b)
            α öğrenme oranıdır
            dθ backpropagation tarafından hesaplanan gradyandır
        
        The learning rate controls the step size. Too large: may overshoot minimum.
        Too small: slow convergence.
        
        Öğrenme oranı adım boyutunu kontrol eder. Çok büyük: minimum'u aşabilir.
        Çok küçük: yavaş yakınsama.
        
        Args:
            gradients (dict): Computed gradients from backpropagation
                             Backpropagation'dan hesaplanan gradyanlar
                             Keys: 'dW1', 'db1', 'dW2', 'db2', ...
        """
        for l in range(1, self.L + 1):
            # Update weights: W = W - α * dW
            # Ağırlıkları güncelle: W = W - α * dW
            self.parameters[f'W{l}'] -= self.learning_rate * gradients[f'dW{l}']
            
            # Update biases: b = b - α * db
            # Sapmaları güncelle: b = b - α * db
            self.parameters[f'b{l}'] -= self.learning_rate * gradients[f'db{l}']
    
    # ==============================================================================
    #                         PREDICTION
    #                         TAHMİN
    # ==============================================================================
    
    def predict(self, X):
        """
        Make predictions for input data.
        Girdi verisi için tahminler yapar.
        
        This method performs a forward pass through the trained network and returns
        the predicted class labels (not probabilities).
        
        Bu metod, eğitilmiş ağ üzerinden bir ileri geçiş yapar ve tahmin edilen
        sınıf etiketlerini döndürür (olasılıklar değil).
        
        Args:
            X (np.ndarray): Input data matrix, shape (n_samples, n_features)
                           Girdi veri matrisi, boyut (n_samples, n_features)
                           
        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,)
                       Tahmin edilen sınıf etiketleri, boyut (n_samples,)
                       Each element is an integer from 0 to (n_classes-1)
                       Her eleman 0'dan (n_classes-1)'e kadar bir tamsayıdır
        """
        # Perform forward propagation to get output probabilities
        # Çıktı olasılıklarını elde etmek için ileri yayılım gerçekleştir
        A_final, _ = self._forward_propagation(X)
        
        # Convert probabilities to class labels by taking argmax
        # Argmax alarak olasılıkları sınıf etiketlerine dönüştür
        # For each sample, select the class with highest probability
        # Her örnek için, en yüksek olasılığa sahip sınıfı seç
        predictions = np.argmax(A_final, axis=1)
        return predictions
    
    # ==============================================================================
    #                         TRAINING (FIT METHOD)
    #                         EĞİTİM (FIT METODU)
    # ==============================================================================
    
    def fit(self, X, y, epochs=100, batch_size=32):
        """
        Train the neural network using mini-batch gradient descent.
        Mini-batch gradient descent kullanarak sinir ağını eğitir.
        
        This method implements the complete training loop:
        Bu metod, tam eğitim döngüsünü uygular:
        
        1. Convert labels to one-hot encoding
           Etiketleri one-hot kodlamaya dönüştür
        2. For each epoch:
           Her epoch için:
           a. Shuffle the training data (improves generalization)
              Eğitim verisini karıştır (genelleme yeteneğini artırır)
           b. Split data into mini-batches
              Veriyi mini-batch'lere böl
           c. For each mini-batch:
              Her mini-batch için:
              - Forward propagation: Compute predictions
                İleri yayılım: Tahminleri hesapla
              - Compute loss: Measure prediction error
                Kaybı hesapla: Tahmin hatasını ölç
              - Backward propagation: Compute gradients
                Geri yayılım: Gradyanları hesapla
              - Update parameters: Apply gradient descent
                Parametreleri güncelle: Gradient descent uygula
           d. Yield progress (epoch number, average loss, model)
              İlerlemeyi döndür (epoch numarası, ortalama kayıp, model)
        
        Mini-Batch Gradient Descent / Mini-Batch Gradient Descent:
            - Faster than batch gradient descent (full dataset)
              Batch gradient descent'ten (tüm veri seti) daha hızlı
            - More stable than stochastic gradient descent (single sample)
              Stochastic gradient descent'ten (tek örnek) daha kararlı
            - Typical batch sizes: 16, 32, 64, 128, 256
              Tipik batch boyutları: 16, 32, 64, 128, 256
        
        Generator Pattern / Generator Deseni:
            This method uses Python's generator pattern (yield) to return
            progress after each epoch. This allows the GUI to:
            - Display real-time training progress
            - Update visualizations during training
            - Implement early stopping if needed
            
            Bu metod, her epoch sonrası ilerlemeyi döndürmek için Python'un
            generator desenini (yield) kullanır. Bu, GUI'nin şunları yapmasını sağlar:
            - Gerçek zamanlı eğitim ilerlemesini göster
            - Eğitim sırasında görselleştirmeleri güncelle
            - Gerekirse erken durdurmayı uygula
        
        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features)
                           Eğitim verisi, boyut (n_samples, n_features)
            y (np.ndarray): Training labels (class indices), shape (n_samples,)
                           Eğitim etiketleri (sınıf indeksleri), boyut (n_samples,)
                           Values should be integers from 0 to (n_classes-1)
                           Değerler 0'dan (n_classes-1)'e kadar tamsayı olmalı
            epochs (int, optional): Number of training epochs. Default: 100
                                   Eğitim epoch sayısı. Varsayılan: 100
                                   One epoch = one pass through entire dataset
                                   Bir epoch = tüm veri setinden bir geçiş
            batch_size (int, optional): Mini-batch size. Default: 32
                                       Mini-batch boyutu. Varsayılan: 32
                                       
        Yields:
            tuple: (epoch_number, average_loss, self)
                epoch_number (int): Current epoch (1 to epochs)
                                   Mevcut epoch (1'den epochs'a)
                average_loss (float): Average loss across all batches in this epoch
                                     Bu epoch'taki tüm batch'ler boyunca ortalama kayıp
                self (MLP): Reference to the model (for visualization)
                           Model referansı (görselleştirme için)
        
        Example / Örnek:
            >>> model = MLP([2, 5, 3], ['relu', 'softmax'])
            >>> for epoch, loss, _ in model.fit(X_train, y_train, epochs=100):
            >>>     print(f"Epoch {epoch}, Loss: {loss:.4f}")
        """
        n_samples = X.shape[0]  # Total number of training samples / Toplam eğitim örneği sayısı
        
        # ======================================================================
        # Convert class labels to one-hot encoding
        # Sınıf etiketlerini one-hot kodlamaya dönüştür
        # ======================================================================
        # 
        # One-hot encoding: Each class is represented as a binary vector
        # One-hot kodlama: Her sınıf bir ikili vektör olarak temsil edilir
        #
        # Example / Örnek:
        #   Class 0: [1, 0, 0]
        #   Class 1: [0, 1, 0]
        #   Class 2: [0, 0, 1]
        #
        # This format is required for softmax + cross-entropy loss.
        # Bu format, softmax + cross-entropy kaybı için gereklidir.
        #
        n_classes = self.layer_dims[-1]  # Number of output neurons / Çıktı nöron sayısı
        Y = np.zeros((n_samples, n_classes))  # Initialize with zeros / Sıfırlarla başlat
        Y[np.arange(n_samples), y.astype(int)] = 1  # Set corresponding class to 1 / İlgili sınıfı 1'e ayarla
        
        # ======================================================================
        # Main Training Loop
        # Ana Eğitim Döngüsü
        # ======================================================================
        
        for epoch in range(epochs):
            # ------------------------------------------------------------------
            # Shuffle training data (important for good generalization)
            # Eğitim verisini karıştır (iyi genelleme için önemli)
            # ------------------------------------------------------------------
            # 
            # Shuffling prevents the model from learning the order of examples
            # and helps it generalize better to unseen data.
            # 
            # Karıştırma, modelin örneklerin sırasını öğrenmesini engeller
            # ve görülmemiş veriye daha iyi genelleme yapmasına yardımcı olur.
            #
            indices = np.random.permutation(n_samples)  # Random permutation of indices / İndekslerin rastgele permütasyonu
            X_shuffled = X[indices]  # Shuffle features / Özellikleri karıştır
            Y_shuffled = Y[indices]  # Shuffle labels / Etiketleri karıştır
            
            epoch_loss = 0  # Accumulator for total loss in this epoch / Bu epoch'taki toplam kayıp için toplayıcı
            n_batches = 0  # Counter for number of batches / Batch sayısı için sayaç
            
            # ------------------------------------------------------------------
            # Mini-Batch Gradient Descent
            # Mini-Batch Gradient Descent
            # ------------------------------------------------------------------
            # 
            # Process data in small batches rather than entire dataset at once.
            # Tüm veri setini bir kerede işlemek yerine küçük batch'ler halinde işle.
            #
            for i in range(0, n_samples, batch_size):
                # Extract mini-batch / Mini-batch'i çıkar
                X_batch = X_shuffled[i:i+batch_size]  # Shape: (batch_size, n_features)
                Y_batch = Y_shuffled[i:i+batch_size]  # Shape: (batch_size, n_classes)
                
                # STEP 1: Forward Propagation
                # ADIM 1: İleri Yayılım
                # Compute predictions for this batch
                # Bu batch için tahminleri hesapla
                A_final, caches = self._forward_propagation(X_batch)
                
                # STEP 2: Compute Loss
                # ADIM 2: Kaybı Hesapla
                # Measure how far predictions are from true labels
                # Tahminlerin gerçek etiketlerden ne kadar uzak olduğunu ölç
                batch_loss = self._compute_loss(A_final, Y_batch)
                epoch_loss += batch_loss  # Accumulate loss / Kaybı biriktir
                n_batches += 1  # Increment batch counter / Batch sayacını artır
                
                # STEP 3: Backward Propagation
                # ADIM 3: Geri Yayılım
                # Compute gradients using chain rule
                # Zincir kuralı kullanarak gradyanları hesapla
                gradients = self._backward_propagation(Y_batch, caches)
                
                # STEP 4: Parameter Update
                # ADIM 4: Parametre Güncellemesi
                # Update weights and biases using gradient descent
                # Gradient descent kullanarak ağırlıkları ve sapmaları güncelle
                self._update_parameters(gradients)
            
            # ------------------------------------------------------------------
            # Compute average loss for this epoch
            # Bu epoch için ortalama kaybı hesapla
            # ------------------------------------------------------------------
            avg_loss = epoch_loss / n_batches
            
            # ------------------------------------------------------------------
            # Yield progress (Generator pattern)
            # İlerlemeyi döndür (Generator deseni)
            # ------------------------------------------------------------------
            # 
            # This allows the calling code (GUI) to:
            # Bu, çağıran kodun (GUI) şunları yapmasını sağlar:
            # - Update visualizations in real-time
            #   Görselleştirmeleri gerçek zamanlı güncelle
            # - Display training progress
            #   Eğitim ilerlemesini göster
            # - Implement early stopping
            #   Erken durdurmayı uygula
            #
            yield epoch + 1, avg_loss, self  # (epoch_number, loss, model)
