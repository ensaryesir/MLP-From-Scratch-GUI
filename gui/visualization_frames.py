"""
Görselleştirme Paneli
Eğitim, Test ve Hata Grafiği sekmelerini içeren ana görselleştirme alanı.
"""

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


class VisualizationFrame(ctk.CTkFrame):
    """
    Üç sekme içeren görselleştirme çerçevesi:
    1. Eğitim (Train) - Eğitim verisi ve karar sınırları
    2. Test - Test verisi ve karar sınırları
    3. Hata Grafiği (Loss) - Epoch'lara göre loss grafiği
    """
    
    def __init__(self, master, on_point_added_callback=None, **kwargs):
        """
        VisualizationFrame'i başlatır.
        
        Args:
            master: Üst widget
            on_point_added_callback: Fare tıklamasında çağrılacak callback fonksiyonu
        """
        super().__init__(master, **kwargs)
        
        self.on_point_added_callback = on_point_added_callback
        self.loss_history = []
        
        # Tabbed view oluştur
        self.tabview = ctk.CTkTabview(self, width=700, height=600)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Sekmeleri ekle
        self.tabview.add("🎯 Eğitim (Train)")
        self.tabview.add("📊 Test")
        self.tabview.add("📈 Hata Grafiği (Loss)")
        
        # Her sekme için matplotlib figure'ları oluştur
        self._setup_train_tab()
        self._setup_test_tab()
        self._setup_loss_tab()
        
    def _setup_train_tab(self):
        """Eğitim sekmesini kurar."""
        tab = self.tabview.tab("🎯 Eğitim (Train)")
        
        # Matplotlib figure
        self.train_fig = Figure(figsize=(7, 6), dpi=100)
        self.train_ax = self.train_fig.add_subplot(111)
        self.train_ax.set_xlim(-1, 11)
        self.train_ax.set_ylim(-1, 11)
        self.train_ax.set_xlabel('X')
        self.train_ax.set_ylabel('Y')
        self.train_ax.set_title('Eğitim Verisi - Fare ile Nokta Ekleyin')
        self.train_ax.grid(True, alpha=0.3)
        
        # Canvas
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, tab)
        self.train_canvas.draw()
        self.train_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Fare tıklama olayını bağla
        self.train_canvas.mpl_connect('button_press_event', self._on_train_click)
        
    def _setup_test_tab(self):
        """Test sekmesini kurar."""
        tab = self.tabview.tab("📊 Test")
        
        # Matplotlib figure
        self.test_fig = Figure(figsize=(7, 6), dpi=100)
        self.test_ax = self.test_fig.add_subplot(111)
        self.test_ax.set_xlim(-1, 11)
        self.test_ax.set_ylim(-1, 11)
        self.test_ax.set_xlabel('X')
        self.test_ax.set_ylabel('Y')
        self.test_ax.set_title('Test Verisi ve Model Performansı')
        self.test_ax.grid(True, alpha=0.3)
        
        # Canvas
        self.test_canvas = FigureCanvasTkAgg(self.test_fig, tab)
        self.test_canvas.draw()
        self.test_canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def _setup_loss_tab(self):
        """Hata grafiği sekmesini kurar."""
        tab = self.tabview.tab("📈 Hata Grafiği (Loss)")
        
        # Matplotlib figure
        self.loss_fig = Figure(figsize=(7, 6), dpi=100)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Eğitim Sırasında Loss Değişimi')
        self.loss_ax.grid(True, alpha=0.3)
        
        # Canvas
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, tab)
        self.loss_canvas.draw()
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def _on_train_click(self, event):
        """
        Eğitim grafiğine tıklandığında çağrılır.
        
        Args:
            event: Matplotlib mouse event
        """
        # Tıklama grafiğin içindeyse
        if event.inaxes == self.train_ax and event.xdata and event.ydata:
            x, y = event.xdata, event.ydata
            
            # Callback'i çağır
            if self.on_point_added_callback:
                self.on_point_added_callback(x, y)
    
    def plot_data_points(self, data_handler, ax=None):
        """
        Veri noktalarını çizer.
        
        Args:
            data_handler: DataHandler instance
            ax: Matplotlib axes (None ise train_ax kullanılır)
        """
        if ax is None:
            ax = self.train_ax
        
        # Her sınıf için
        for class_id in range(data_handler.get_num_classes()):
            points = data_handler.get_data_by_class(class_id)
            if len(points) > 0:
                color = data_handler.get_color(class_id)
                ax.scatter(points[:, 0], points[:, 1], 
                          c=color, s=100, alpha=0.7, 
                          edgecolors='black', linewidth=1.5,
                          label=data_handler.classes[class_id])
        
        # Legend ekle
        if data_handler.get_num_classes() > 0:
            ax.legend(loc='upper right')
    
    def update_train_view(self, data_handler):
        """
        Eğitim görünümünü günceller.
        
        Args:
            data_handler: DataHandler instance
        """
        self.train_ax.clear()
        self.train_ax.set_xlim(-1, 11)
        self.train_ax.set_ylim(-1, 11)
        self.train_ax.set_xlabel('X')
        self.train_ax.set_ylabel('Y')
        self.train_ax.set_title('Eğitim Verisi - Fare ile Nokta Ekleyin')
        self.train_ax.grid(True, alpha=0.3)
        
        # Veri noktalarını çiz
        self.plot_data_points(data_handler, self.train_ax)
        
        self.train_canvas.draw()
    
    def update_decision_boundary(self, model, X, y, data_handler, tab_name='train'):
        """
        Karar sınırlarını günceller.
        
        Args:
            model: Eğitilmiş model
            X: Veri noktaları
            y: Etiketler
            data_handler: DataHandler instance
            tab_name: 'train' veya 'test'
        """
        # Hangi ax kullanılacak?
        if tab_name == 'train':
            ax = self.train_ax
            canvas = self.train_canvas
            title = 'Eğitim Verisi - Karar Sınırları'
        else:
            ax = self.test_ax
            canvas = self.test_canvas
            title = 'Test Verisi - Model Performansı'
        
        ax.clear()
        
        # Meshgrid oluştur
        x_min, x_max = -1, 11
        y_min, y_max = -1, 11
        h = 0.1  # Grid resolution
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Tahminler
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Karar sınırlarını çiz (contourf)
        n_classes = data_handler.get_num_classes()
        colors = [data_handler.get_color(i) for i in range(n_classes)]
        ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5,
                   colors=colors)
        
        # Veri noktalarını çiz
        if len(X) > 0:
            for class_id in range(n_classes):
                mask = y == class_id
                if np.any(mask):
                    color = data_handler.get_color(class_id)
                    ax.scatter(X[mask, 0], X[mask, 1],
                             c=color, s=100, alpha=0.8,
                             edgecolors='black', linewidth=1.5,
                             label=data_handler.classes[class_id])
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if n_classes > 0:
            ax.legend(loc='upper right')
        
        canvas.draw()
    
    def update_loss_plot(self, epoch, loss):
        """
        Hata grafiğini günceller.
        
        Args:
            epoch: Epoch numarası
            loss: Loss değeri
        """
        self.loss_history.append((epoch, loss))
        
        self.loss_ax.clear()
        
        if len(self.loss_history) > 0:
            epochs, losses = zip(*self.loss_history)
            self.loss_ax.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
            self.loss_ax.set_xlabel('Epoch')
            self.loss_ax.set_ylabel('Loss')
            self.loss_ax.set_title('Eğitim Sırasında Loss Değişimi')
            self.loss_ax.grid(True, alpha=0.3)
        
        self.loss_canvas.draw()
    
    def clear_loss_history(self):
        """Loss geçmişini temizler."""
        self.loss_history = []
        self.loss_ax.clear()
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Eğitim Sırasında Loss Değişimi')
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_canvas.draw()
    
    def switch_to_tab(self, tab_name):
        """
        Belirtilen sekmeye geçiş yapar.
        
        Args:
            tab_name: 'train', 'test', veya 'loss'
        """
        tab_mapping = {
            'train': "🎯 Eğitim (Train)",
            'test': "📊 Test",
            'loss': "📈 Hata Grafiği (Loss)"
        }
        
        if tab_name in tab_mapping:
            self.tabview.set(tab_mapping[tab_name])
