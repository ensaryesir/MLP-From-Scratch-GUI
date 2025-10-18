"""
Visualization Panel - Interactive Plotting Module
Görselleştirme Paneli - İnteraktif Çizim Modülü

This module implements the left panel containing interactive matplotlib plots
for visualizing training data, test results, and loss curves in real-time.

Bu modül, eğitim verisini, test sonuçlarını ve kayıp eğrilerini gerçek zamanlı
görselleştirmek için interaktif matplotlib grafikleri içeren sol paneli uygular.

Visualization Components / Görselleştirme Bileşenleri:
    - Training Tab: Interactive canvas for adding data points and viewing
                   decision boundaries during training
      Eğitim Sekmesi: Veri noktaları eklemek ve eğitim sırasında
                      karar sınırlarını görüntülemek için interaktif tuval

    - Test Tab: Display of model performance on test data with
               decision boundaries
      Test Sekmesi: Karar sınırlarıyla test verisi üzerinde
                   model performansının gösterimi

    - Loss Tab: Real-time loss curve showing training progress
      Kayıp Sekmesi: Eğitim ilerlemesini gösteren gerçek zamanlı kayıp eğrisi

Matplotlib Integration / Matplotlib Entegrasyonu:
    Uses FigureCanvasTkAgg to embed matplotlib figures in CustomTkinter GUI,
    enabling interactive plots with mouse events.

    Fare olaylarıyla interaktif grafikleri etkinleştirerek, matplotlib
    figürlerini CustomTkinter GUI'ye gömmek için FigureCanvasTkAgg kullanır.

Author: Developed for educational purposes
Date: 2024
"""

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


class VisualizationFrame(ctk.CTkFrame):
    """
    Visualization Frame - Interactive Plotting Component
    Görselleştirme Çerçevesi - İnteraktif Çizim Bileşeni

    This class implements a tabbed interface with three visualization panels,
    each containing matplotlib plots embedded in the CustomTkinter GUI.

    Bu sınıf, her biri CustomTkinter GUI'ye gömülü matplotlib grafikleri
    içeren üç görselleştirme paneli ile sekmeli bir arayüz uygular.

    Tabs / Sekmeler:
        1. Training Tab (Eğitim Sekmesi):
           - Interactive 2D scatter plot for data points
             Veri noktaları için interaktif 2D dağılım grafiği
           - Mouse click to add points
             Nokta eklemek için fare tıklaması
           - Real-time decision boundary visualization
             Gerçek zamanlı karar sınırı görselleştirmesi

        2. Test Tab (Test Sekmesi):
           - Display test data and model predictions
             Test verisini ve model tahminlerini göster
           - Decision boundaries on test set
             Test setinde karar sınırları

        3. Loss Tab (Kayıp Sekmesi):
           - Line plot of loss vs. epoch
             Kaybın epoch'a karşı çizgi grafiği
           - Updated in real-time during training
             Eğitim sırasında gerçek zamanlı güncellenir

    Matplotlib Components / Matplotlib Bileşenleri:
        - Figure: Container for plots / Grafikler için kap
        - Axes: Individual plot area / Bireysel grafik alanı
        - Canvas: Tkinter widget for embedding / Gömmek için Tkinter widget'ı

    Interactive Features / İnteraktif Özellikler:
        - Mouse click events on training canvas
          Eğitim tuvalinde fare tıklama olayları
        - Dynamic updates during training
          Eğitim sırasında dinamik güncellemeler
        - Tab switching for different views
          Farklı görünümler için sekme geçişi
    """

    def __init__(self, master, on_point_added_callback=None, **kwargs):
        """
        Initialize the Visualization Frame with matplotlib plots.
        Matplotlib grafikleriyle Görselleştirme Çerçevesini başlat.

        Sets up three tabbed views with embedded matplotlib figures,
        configures mouse event handlers, and initializes plot properties.

        Gömülü matplotlib figürleriyle üç sekmeli görünüm kurar,
        fare olay yöneticilerini yapılandırır ve grafik özelliklerini başlatır.

        Args:
            master: Parent widget (main application)
                   Üst widget (ana uygulama)

            on_point_added_callback (callable, optional):
                Callback function triggered on mouse click in training plot.
                Signature: callback(x, y) where x, y are coordinates.

                Eğitim grafiğinde fare tıklamasında tetiklenen callback fonksiyonu.
                İmza: callback(x, y) burada x, y koordinatlardır.

            **kwargs: Additional arguments for CTkFrame
                     CTkFrame için ek argümanlar
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
        """
        Setup the training tab with interactive matplotlib canvas.
        İnteraktif matplotlib tuvaliyle eğitim sekmesini kur.

        Creates a matplotlib figure with mouse click event handling
        for adding data points interactively.

        İnteraktif olarak veri noktaları eklemek için fare tıklama
        olay işlemeyle bir matplotlib figürü oluşturur.
        """
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
        """
        Setup the test tab with matplotlib plot.
        Matplotlib grafiğiyle test sekmesini kur.
        """
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
        """
        Setup the loss tab with matplotlib plot.
        Matplotlib grafiğiyle kayıp sekmesini kur.
        """
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
        Handle mouse click events on the training plot.
        Eğitim grafiğindeki fare tıklama olaylarını işle.

        This method is called when the user clicks on the training canvas.
        It extracts coordinates and triggers the callback to add a data point.

        Bu metod, kullanıcı eğitim tuvaline tıkladığında çağrılır.
        Koordinatları çıkarır ve bir veri noktası eklemek için callback'i tetikler.

        Args:
            event (matplotlib.backend_bases.MouseEvent):
                Mouse event containing click coordinates and plot information
                Tıklama koordinatlarını ve grafik bilgilerini içeren fare olayı
        """
        # Tıklama grafiğin içindeyse
        if event.inaxes == self.train_ax and event.xdata and event.ydata:
            x, y = event.xdata, event.ydata

            # Callback'i çağır
            if self.on_point_added_callback:
                self.on_point_added_callback(x, y)

    def plot_data_points(self, data_handler, ax=None):
        """
        Plot data points on the given axes.
        Veri noktalarını verilen eksenlerde çiz.

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
        Update the training view with new data.
        Eğitim görünümünü yeni verilerle güncelle.

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
        Update decision boundary visualization using trained model.
        Eğitilmiş model kullanarak karar sınırı görselleştirmesini güncelle.

        This method creates a dense grid of points, predicts their classes,
        and visualizes the decision regions using contour plots.

        Bu metod, yoğun bir nokta ızgarası oluşturur, sınıflarını tahmin eder
        ve kontur grafikleri kullanarak karar bölgelerini görselleştirir.

        Algorithm / Algoritma:
            1. Create meshgrid covering the plot area
               Grafik alanını kaplayan meshgrid oluştur
            2. Predict class for each grid point
               Her ızgara noktası için sınıfı tahmin et
            3. Draw filled contours for decision regions
               Karar bölgeleri için dolu konturlar çiz
            4. Overlay actual data points
               Gerçek veri noktalarını üst üste bindir

        Args:
            model: Trained model with predict() method
                  predict() metodu olan eğitilmiş model

            X (np.ndarray): Data points to display, shape (n_samples, 2)
                           Gösterilecek veri noktaları, boyut (n_samples, 2)

            y (np.ndarray): True labels, shape (n_samples,)
                           Gerçek etiketler, boyut (n_samples,)

            data_handler (DataHandler): Data handler for colors and classes
                                       Renkler ve sınıflar için veri yöneticisi

            tab_name (str): 'train' or 'test' to select which plot to update
                           Hangi grafiğin güncelleneceğini seçmek için 'train' veya 'test'
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
        Update the loss plot with new training progress.
        Yeni eğitim ilerlemesiyle kayıp grafiğini güncelle.
        
        Adds a new data point to the loss history and redraws the plot.
        This provides real-time feedback on training convergence.
        
        Kayıp geçmişine yeni bir veri noktası ekler ve grafiği yeniden çizer.
        Bu, eğitim yakınsaması hakkında gerçek zamanlı geri bildirim sağlar.
        
        Args:
            epoch (int): Current epoch number / Mevcut epoch numarası
            loss (float): Loss value at this epoch / Bu epoch'taki kayıp değeri
        """
        # Add new data point to history
        # Geçmişe yeni veri noktası ekle
        self.loss_history.append((epoch, loss))
        
        # Clear and redraw plot
        # Grafiği temizle ve yeniden çiz
        self.loss_ax.clear()
        
        if len(self.loss_history) > 0:
            # Unpack epochs and losses from history
            # Geçmişten epoch'ları ve kayıpları çıkar
            epochs, losses = zip(*self.loss_history)
            
            # Plot loss curve with blue line and markers
            # Mavi çizgi ve işaretleyicilerle kayıp eğrisini çiz
            self.loss_ax.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
            self.loss_ax.set_xlabel('Epoch')
            self.loss_ax.set_ylabel('Loss')
            self.loss_ax.set_title('Training Loss Over Time / Eğitim Sırasında Loss Değişimi')
            self.loss_ax.grid(True, alpha=0.3)
        
        # Refresh canvas to display changes
        # Değişiklikleri göstermek için tuvali yenile
        self.loss_canvas.draw()
    
    def clear_loss_history(self):
        """
        Clear the loss history and reset the plot.
        Kayıp geçmişini temizle ve grafiği sıfırla.
        
        Called when starting a new training session to remove
        old loss data from previous runs.
        
        Önceki çalışmalardan eski kayıp verisini kaldırmak için
        yeni bir eğitim oturumu başlatırken çağrılır.
        """
        self.loss_history = []
        self.loss_ax.clear()
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Eğitim Sırasında Loss Değişimi')
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_canvas.draw()
    
    def switch_to_tab(self, tab_name):
        """
        Programmatically switch to a specific tab.
        Belirtilen sekmeye programatık olarak geçiş yap.
        
        Used to automatically switch to relevant tabs (e.g., test tab
        after training completes).
        
        İlgili sekmelere otomatik olarak geçiş yapmak için kullanılır
        (örneğin, eğitim tamamlandıktan sonra test sekmesine).
        
        Args:
            tab_name (str): Tab identifier: 'train', 'test', or 'loss'
                           Sekme tanımlayıcısı: 'train', 'test' veya 'loss'
        """
        # Map simple identifiers to full tab names with emojis
        # Basit tanımlayıcıları emoji'li tam sekme adlarına eşle
        tab_mapping = {
            'train': "🎯 Eğitim (Train)",
            'test': "📊 Test",
            'loss': "📈 Hata Grafiği (Loss)"
        }
        
        # Switch to tab if valid identifier provided
        # Geçerli tanımlayıcı sağlanırsa sekmeye geç
        if tab_name in tab_mapping:
            self.tabview.set(tab_mapping[tab_name])
