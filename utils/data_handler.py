"""
Data Management Module
Veri Yönetim Modülü

This module provides data handling capabilities for the Neural Network Visualizer.
It manages user-added training points, class labels, and provides train/test splitting.

Bu modül, Sinir Ağı Görselleştiricisi için veri işleme yetenekleri sağlar.
Kullanıcı tarafından eklenen eğitim noktalarını, sınıf etiketlerini yönetir
ve train/test ayrımı sağlar.

Key Responsibilities / Temel Sorumluluklar:
    - Store and manage training data points
      Eğitim veri noktalarını sakla ve yönet
    - Manage class labels and colors
      Sınıf etiketlerini ve renklerini yönet
    - Provide train/test split with random shuffling
      Rastgele karıştırmayla train/test ayrımı sağla
    - Retrieve data by class for visualization
      Görselleştirme için sınıfa göre veri getir

Author: Developed for educational purposes
Date: 2024
"""

import numpy as np


class DataHandler:
    """
    Data Handler for Neural Network Training
    Sinir Ağı Eğitimi için Veri Yöneticisi

    This class manages all aspects of training data in the application.
    It provides an interface for adding, removing, and organizing data points,
    as well as splitting data for training and testing.

    Bu sınıf, uygulamadaki eğitim verisinin tüm yönlerini yönetir.
    Veri noktalarını ekleme, çıkarma ve düzenleme için bir arayüz sağlar,
    ayrıca veriyi eğitim ve test için ayırır.

    Data Structure / Veri Yapısı:
        Each data point is stored as a tuple: (x, y, class_id)
        Her veri noktası bir demet olarak saklanır: (x, y, class_id)
            - x: X-coordinate (float) / X-koordinatı (float)
            - y: Y-coordinate (float) / Y-koordinatı (float)
            - class_id: Class label index (int) / Sınıf etiketi indeksi (int)

    Attributes:
        data_points (list): List of (x, y, class_id) tuples
                           (x, y, class_id) demetlerinin listesi
        classes (list): List of class names (strings)
                       Sınıf adlarının listesi (string'ler)
        colors (list): Predefined colors for each class (hex codes)
                      Her sınıf için öntanımlı renkler (hex kodlar)
    """

    def __init__(self):
        """
        Initialize the Data Handler with empty data structures.
        Veri Yöneticisini boş veri yapılarıyla başlat.

        Sets up empty lists for data points and classes, and defines
        a color palette for visualization.

        Veri noktaları ve sınıflar için boş listeler kurar ve
        görselleştirme için bir renk paleti tanımlar.
        """
        # Storage for all training points: list of (x, y, class_id) tuples
        # Tüm eğitim noktaları için depolama: (x, y, class_id) demetlerinin listesi
        self.data_points = []  # [(x, y, class_id), ...]
        # List of class names (e.g., ['Class 0', 'Class 1'])
        # Sınıf adları listesi (örn., ['Class 0', 'Class 1'])
        self.classes = []
        # Predefined color palette for up to 6 classes (hex codes)
        # 6 sınıfa kadar öntanımlı renk paleti (hex kodlar)
        # Colors chosen for good visual distinction and aesthetics
        # Renkler iyi görsel ayırt edilebilirlik ve estetik için seçilmiştir
        self.colors = ['#FF6B6B',  # Red / Kırmızı
                      '#4ECDC4',  # Teal / Deniz mavisi
                      '#45B7D1',  # Blue / Mavi
                      '#FFA07A',  # Light Salmon / Açık somon
                      '#98D8C8',  # Mint / Nane yeşili
                      '#F7DC6F']  # Yellow / Sarı

    def add_class(self, class_name=None):
        """
        Add a new class label to the dataset.
        Veri setine yeni bir sınıf etiketi ekle.

        Args:
            class_name (str, optional): Name for the new class.
                                       If None, generates 'Class N' automatically.
                                       Yeni sınıf için ad.
                                       None ise, otomatik olarak 'Class N' oluşturur.
        """
        if class_name is None:
            class_name = f"Class {len(self.classes)}"
        if class_name not in self.classes:
            self.classes.append(class_name)

    def remove_class(self):
        """
        Remove the most recently added class and its associated data points.
        En son eklenen sınıfı ve ilişkili veri noktalarını kaldır.

        This method removes the last class from the classes list and
        deletes all data points belonging to that class.

        Bu metod, classes listesinden son sınıfı kaldırır ve
        o sınıfa ait tüm veri noktalarını siler.
        """
        if len(self.classes) > 0:
            # Remove all data points belonging to the last class
            # Son sınıfa ait tüm veri noktalarını kaldır
            class_id = len(self.classes) - 1
            self.data_points = [p for p in self.data_points if p[2] != class_id]
            self.classes.pop()

    def add_point(self, x, y, class_id):
        """
        Add a new training data point.
        Yeni bir eğitim veri noktası ekle.

        Args:
            x (float): X-coordinate of the point / Noktanın X-koordinatı
            y (float): Y-coordinate of the point / Noktanın Y-koordinatı
            class_id (int): Class label index (0 to n_classes-1)
                           Sınıf etiketi indeksi (0'dan n_classes-1'e)
        """
        # Validate class_id is within valid range
        # class_id'nin geçerli aralıkta olduğunu doğrula
        if 0 <= class_id < len(self.classes):
            self.data_points.append((x, y, class_id))

    def clear_data(self):
        """
        Clear all training data points.
        Tüm eğitim veri noktalarını temizle.

        Removes all data points while keeping class definitions intact.
        Sınıf tanımlarını koruyarak tüm veri noktalarını kaldırır.
        """
        self.data_points = []

    def get_data_by_class(self, class_id):
        """
        Retrieve all data points belonging to a specific class.
        Belirli bir sınıfa ait tüm veri noktalarını getir.

        Used for visualization to plot points of each class separately.
        Her sınıfın noktalarını ayrı ayrı çizmek için görselleştirmede kullanılır.

        Args:
            class_id (int): Class label index / Sınıf etiketi indeksi

        Returns:
            np.ndarray: Array of shape (n_points, 2) with x, y coordinates
                       (n_points, 2) boyutunda x, y koordinatları içeren dizi
                       Returns empty array if no points for this class
                       Bu sınıf için nokta yoksa boş dizi döndürür
        """
        points = [p for p in self.data_points if p[2] == class_id]
        if len(points) == 0:
            return np.array([]).reshape(0, 2)
        return np.array([(p[0], p[1]) for p in points])

    def get_all_data(self):
        """
        Get all training data as NumPy arrays.
        Tüm eğitim verisini NumPy dizileri olarak al.

        Converts the internal data structure to NumPy format suitable
        for training neural networks.

        Dahili veri yapısını sinir ağlarını eğitmeye uygun
        NumPy formatına dönüştürür.

        Returns:
            tuple: (X, y)
                X (np.ndarray): Feature matrix, shape (n_samples, 2)
                               Özellik matrisi, boyut (n_samples, 2)
                y (np.ndarray): Labels vector, shape (n_samples,)
                               Etiketler vektörü, boyut (n_samples,)
        """
        if len(self.data_points) == 0:
            return np.array([]).reshape(0, 2), np.array([])

        X = np.array([(p[0], p[1]) for p in self.data_points])
        y = np.array([p[2] for p in self.data_points])
        return X, y

    def get_train_test_split(self, test_ratio=0.2, random_state=42):
        """
        Split data into training and testing sets with random shuffling.
        Veriyi rastgele karıştırarak eğitim ve test setlerine ayır.

        This method implements a stratified split to ensure representative
        distribution of classes in both train and test sets.

        Bu metod, hem eğitim hem de test setlerinde sınıfların temsili
        dağılımını garantilemek için katmanlı ayrım uygular.

        Algorithm / Algoritma:
            1. Get all data as NumPy arrays
               Tüm veriyi NumPy dizileri olarak al
            2. Generate random permutation of indices
               İndekslerin rastgele permütasyonunu oluştur
            3. Split indices based on test_ratio
               test_ratio'ya göre indeksleri ayır
            4. Extract train/test subsets using indices
               İndeksleri kullanarak train/test alt kümelerini çıkar

        Args:
            test_ratio (float, optional): Fraction of data for testing (0.0-1.0).
                                        Default: 0.2 (20%)
                                        Test için veri oranı (0.0-1.0).
                                        Varsayılan: 0.2 (%20)

            random_state (int, optional): Random seed for reproducibility.
                                        Default: 42
                                        Tekrarlanabilirlik için rastgele tohum.
                                        Varsayılan: 42

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
                X_train (np.ndarray): Training features, shape (n_train, 2)
                                     Eğitim özellikleri, boyut (n_train, 2)
                X_test (np.ndarray): Testing features, shape (n_test, 2)
                                    Test özellikleri, boyut (n_test, 2)
                y_train (np.ndarray): Training labels, shape (n_train,)
                                     Eğitim etiketleri, boyut (n_train,)
                y_test (np.ndarray): Testing labels, shape (n_test,)
                                    Test etiketleri, boyut (n_test,)
        """
        if len(self.data_points) == 0:
            empty = np.array([]).reshape(0, 2)
            return empty, empty, np.array([]), np.array([])

        X, y = self.get_all_data()  # Get all data / Tüm veriyi al

        # Shuffle data using random permutation
        # Rastgele permütasyon kullanarak veriyi karıştır
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))

        # Calculate test set size based on ratio
        # Orana göre test seti boyutunu hesapla
        test_size = int(len(X) * test_ratio)

        # Split indices into test and train
        # İndeksleri test ve train olarak ayır
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test

    def get_color(self, class_id):
        """
        Get the display color for a specific class.
        Belirli bir sınıf için görüntüleme rengini al.

        Args:
            class_id (int): Class label index / Sınıf etiketi indeksi

        Returns:
            str: Hex color code (e.g., '#FF6B6B') / Hex renk kodu (örn., '#FF6B6B')
        """
        # Use modulo to wrap around if more classes than colors
        # Renk sayısından fazla sınıf varsa döngüsel olarak kullan
        return self.colors[class_id % len(self.colors)]

    def get_num_classes(self):
        """
        Get the total number of classes.
        Toplam sınıf sayısını al.

        Returns:
            int: Number of classes / Sınıf sayısı
        """
        return len(self.classes)

    def get_num_points(self):
        """
        Get the total number of data points.
        Toplam veri noktası sayısını al.

        Returns:
            int: Number of data points / Veri noktası sayısı
        """
        return len(self.data_points)
