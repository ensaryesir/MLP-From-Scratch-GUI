"""
Neural Network Visualizer - Ana Uygulama
Sinir ağlarını interaktif olarak görselleştiren masaüstü uygulaması.
"""

import customtkinter as ctk
import numpy as np
from tkinter import messagebox
import time

from utils.data_handler import DataHandler
from algorithms.single_layer import Perceptron, DeltaRule
from algorithms.mlp import MLP
from gui.control_panel import ControlPanel
from gui.visualization_frames import VisualizationFrame


class NeuralNetworkVisualizer(ctk.CTk):
    """Ana uygulama sınıfı - tüm bileşenleri koordine eder."""
    
    def __init__(self):
        """Uygulamayı başlatır."""
        super().__init__()
        
        # Pencere ayarları
        self.title("🧠 Neural Network Visualizer - MLP From Scratch")
        self.geometry("1400x800")
        
        # Tema ayarı
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Veri yöneticisi
        self.data_handler = DataHandler()
        
        # Başlangıçta 2 sınıf ekle
        self.data_handler.add_class("Class 0")
        self.data_handler.add_class("Class 1")
        
        # Eğitim durumu
        self.is_training = False
        self.trained_model = None
        
        # UI'ı kur
        self._setup_ui()
        
        # Kontrol panelindeki sınıf radio button'larını güncelle
        self._update_class_radios()
    
    def _setup_ui(self):
        """Kullanıcı arayüzünü kurar."""
        # Grid layout
        self.grid_columnconfigure(0, weight=3)  # Görselleştirme alanı
        self.grid_columnconfigure(1, weight=1)  # Kontrol paneli
        self.grid_rowconfigure(0, weight=1)
        
        # Görselleştirme paneli (sol)
        self.visualization_frame = VisualizationFrame(
            self,
            on_point_added_callback=self._on_point_added
        )
        self.visualization_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        # Kontrol paneli (sağ)
        self.control_panel = ControlPanel(
            self,
            on_add_class=self._on_add_class,
            on_remove_class=self._on_remove_class,
            on_clear_data=self._on_clear_data,
            on_start_training=self._on_start_training
        )
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    
    # ========== Event Handlers ==========
    
    def _on_point_added(self, x, y):
        """
        Kullanıcı eğitim grafiğine tıkladığında çağrılır.
        
        Args:
            x, y: Tıklanan koordinatlar
        """
        # Seçili sınıfı al
        class_id = self.control_panel.get_selected_class()
        
        # Veriyi ekle
        self.data_handler.add_point(x, y, class_id)
        
        # Görselleştirmeyi güncelle
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_add_class(self):
        """Yeni sınıf ekler."""
        # Maksimum 6 sınıf
        if self.data_handler.get_num_classes() >= 6:
            messagebox.showwarning("Uyarı", "Maksimum 6 sınıf ekleyebilirsiniz.")
            return
        
        class_name = f"Class {self.data_handler.get_num_classes()}"
        self.data_handler.add_class(class_name)
        self._update_class_radios()
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_remove_class(self):
        """Son sınıfı kaldırır."""
        # En az 2 sınıf olmalı
        if self.data_handler.get_num_classes() <= 2:
            messagebox.showwarning("Uyarı", "En az 2 sınıf bulunmalıdır.")
            return
        
        self.data_handler.remove_class()
        self._update_class_radios()
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_clear_data(self):
        """Tüm veri noktalarını temizler."""
        response = messagebox.askyesno("Onay", "Tüm veri noktalarını silmek istediğinizden emin misiniz?")
        if response:
            self.data_handler.clear_data()
            self.visualization_frame.update_train_view(self.data_handler)
            self.visualization_frame.clear_loss_history()
            self.control_panel.set_status("Veri temizlendi")
    
    def _on_start_training(self):
        """Eğitimi başlatır."""
        if self.is_training:
            messagebox.showinfo("Bilgi", "Eğitim zaten devam ediyor.")
            return
        
        # Veri kontrolü
        if self.data_handler.get_num_points() < 10:
            messagebox.showwarning("Uyarı", "En az 10 veri noktası ekleyin.")
            return
        
        # Eğitimi başlat
        self.is_training = True
        self.control_panel.enable_training(False)
        self.control_panel.set_status("Eğitim başlıyor...")
        
        # Loss geçmişini temizle
        self.visualization_frame.clear_loss_history()
        
        # Parametreleri al
        model_type = self.control_panel.get_model_type()
        learning_rate = self.control_panel.get_learning_rate()
        epochs = self.control_panel.get_epochs()
        batch_size = self.control_panel.get_batch_size()
        test_split = self.control_panel.get_test_split()
        
        # Veriyi hazırla
        X_train, X_test, y_train, y_test = self.data_handler.get_train_test_split(
            test_ratio=test_split
        )
        
        # Test verisini sakla
        self.X_test = X_test
        self.y_test = y_test
        
        # Modeli oluştur
        n_classes = self.data_handler.get_num_classes()
        
        if model_type == "Perceptron":
            model = Perceptron(learning_rate=learning_rate, n_classes=n_classes)
            batch_size = 1  # Perceptron için online learning
        elif model_type == "DeltaRule":
            model = DeltaRule(learning_rate=learning_rate, n_classes=n_classes)
        else:  # MLP
            architecture = self.control_panel.get_architecture()
            # İlk katman girdi boyutu, son katman çıktı boyutu olmalı
            architecture[0] = 2  # 2D girdi
            architecture[-1] = n_classes  # Sınıf sayısı kadar çıktı
            
            activation_funcs = self.control_panel.get_activation_functions()
            l2_lambda = self.control_panel.get_l2_lambda()
            
            model = MLP(
                layer_dims=architecture,
                activation_funcs=activation_funcs,
                learning_rate=learning_rate,
                l2_lambda=l2_lambda
            )
        
        # Eğitimi başlat (asenkron olarak)
        self.after(100, lambda: self._run_training(model, X_train, y_train, epochs, batch_size))
    
    def _run_training(self, model, X_train, y_train, epochs, batch_size):
        """
        Eğitimi çalıştırır ve her epoch'ta UI'ı günceller.
        
        Args:
            model: Eğitilecek model
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
            epochs: Epoch sayısı
            batch_size: Batch boyutu
        """
        # Modeli sakla (eğitim sonunda kullanmak için)
        self.current_model = model
        
        # Fit generator'ü oluştur
        # Perceptron ve DeltaRule batch_size kullanmaz
        if isinstance(model, (Perceptron, DeltaRule)):
            fit_generator = model.fit(X_train, y_train, epochs=epochs)
        else:  # MLP
            fit_generator = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        # Her epoch için
        self._train_next_epoch(fit_generator, X_train, y_train)
    
    def _train_next_epoch(self, fit_generator, X_train, y_train):
        """
        Bir sonraki epoch'u eğitir ve UI'ı günceller.
        
        Args:
            fit_generator: Model.fit() generator'ü
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
        """
        try:
            # Bir sonraki epoch'u al
            epoch, loss, model = next(fit_generator)
            
            # Modeli güncelle
            self.current_model = model
            
            # Durum güncelle
            self.control_panel.set_status(f"Epoch {epoch} - Loss: {loss:.4f}")
            
            # Loss grafiğini güncelle
            self.visualization_frame.update_loss_plot(epoch, loss)
            
            # Her 5 epoch'ta bir karar sınırını güncelle (performans için)
            if epoch % 5 == 0 or epoch == 1:
                self.visualization_frame.update_decision_boundary(
                    model, X_train, y_train, self.data_handler, tab_name='train'
                )
            
            # UI'ı güncelle
            self.update_idletasks()
            
            # Bir sonraki epoch için zamanlayıcı kur (50ms sonra)
            self.after(50, lambda: self._train_next_epoch(fit_generator, X_train, y_train))
            
        except StopIteration:
            # Eğitim tamamlandı - saklanan modeli kullan
            # Son eğitim karar sınırını çiz
            self.visualization_frame.update_decision_boundary(
                self.current_model, X_train, y_train, self.data_handler, tab_name='train'
            )
            self._on_training_completed(self.current_model)
    
    def _on_training_completed(self, model):
        """
        Eğitim tamamlandığında çağrılır.
        
        Args:
            model: Eğitilmiş model
        """
        self.is_training = False
        self.trained_model = model
        self.control_panel.enable_training(True)
        
        # Test setinde değerlendirme yap
        accuracy = None
        if len(self.X_test) > 0:
            # Test karar sınırını çiz
            self.visualization_frame.update_decision_boundary(
                model, self.X_test, self.y_test, self.data_handler, tab_name='test'
            )
            
            # Test accuracy hesapla
            y_pred = model.predict(self.X_test)
            accuracy = np.mean(y_pred == self.y_test) * 100
            
            self.control_panel.set_status(f"Eğitim tamamlandı! Test Accuracy: {accuracy:.2f}%")
            
            # Test sekmesine geç
            self.visualization_frame.switch_to_tab('test')
            
            # Başarı mesajı
            messagebox.showinfo("Başarılı", 
                               f"Eğitim başarıyla tamamlandı!\n"
                               f"Test Accuracy: {accuracy:.2f}%")
        else:
            self.control_panel.set_status("Eğitim tamamlandı!")
            messagebox.showinfo("Başarılı", "Eğitim başarıyla tamamlandı!")
    
    def _update_class_radios(self):
        """Kontrol panelindeki sınıf radio button'larını günceller."""
        classes = self.data_handler.classes
        colors = [self.data_handler.get_color(i) for i in range(len(classes))]
        self.control_panel.update_class_radios(classes, colors)


def main():
    """Ana fonksiyon - uygulamayı başlatır."""
    app = NeuralNetworkVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()
