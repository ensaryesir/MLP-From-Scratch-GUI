"""
Veri Yönetim Modülü
Kullanıcının eklediği veri noktalarını saklayan ve eğitim/test setlerine ayıran sınıf.
"""

import numpy as np


class DataHandler:
    """Veri noktalarını yöneten, saklayan ve hazırlayan sınıf."""
    
    def __init__(self):
        """DataHandler sınıfını başlatır."""
        self.data_points = []  # [(x, y, class_id), ...]
        self.classes = []  # Sınıf isimleri listesi
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
    def add_class(self, class_name=None):
        """Yeni bir sınıf ekler."""
        if class_name is None:
            class_name = f"Class {len(self.classes)}"
        if class_name not in self.classes:
            self.classes.append(class_name)
            
    def remove_class(self):
        """Son eklenen sınıfı kaldırır."""
        if len(self.classes) > 0:
            # Son sınıfa ait veri noktalarını kaldır
            class_id = len(self.classes) - 1
            self.data_points = [p for p in self.data_points if p[2] != class_id]
            self.classes.pop()
            
    def add_point(self, x, y, class_id):
        """Yeni bir veri noktası ekler."""
        if 0 <= class_id < len(self.classes):
            self.data_points.append((x, y, class_id))
            
    def clear_data(self):
        """Tüm veri noktalarını temizler."""
        self.data_points = []
        
    def get_data_by_class(self, class_id):
        """Belirli bir sınıfa ait tüm veri noktalarını döndürür."""
        points = [p for p in self.data_points if p[2] == class_id]
        if len(points) == 0:
            return np.array([]).reshape(0, 2)
        return np.array([(p[0], p[1]) for p in points])
    
    def get_all_data(self):
        """Tüm veri noktalarını numpy array olarak döndürür."""
        if len(self.data_points) == 0:
            return np.array([]).reshape(0, 2), np.array([])
        
        X = np.array([(p[0], p[1]) for p in self.data_points])
        y = np.array([p[2] for p in self.data_points])
        return X, y
    
    def get_train_test_split(self, test_ratio=0.2, random_state=42):
        """
        Veriyi eğitim ve test setlerine ayırır.
        
        Args:
            test_ratio: Test setinin oranı (0.0 - 1.0)
            random_state: Rastgelelik için seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if len(self.data_points) == 0:
            empty = np.array([]).reshape(0, 2)
            return empty, empty, np.array([]), np.array([])
        
        X, y = self.get_all_data()
        
        # Rastgele karıştır
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        
        # Test seti boyutunu hesapla
        test_size = int(len(X) * test_ratio)
        
        # Ayır
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def get_color(self, class_id):
        """Belirli bir sınıf için renk döndürür."""
        return self.colors[class_id % len(self.colors)]
    
    def get_num_classes(self):
        """Toplam sınıf sayısını döndürür."""
        return len(self.classes)
    
    def get_num_points(self):
        """Toplam veri noktası sayısını döndürür."""
        return len(self.data_points)
