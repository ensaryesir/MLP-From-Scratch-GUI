"""
Neural Network Visualizer - Main Application
Sinir Ağı Görselleştiricisi - Ana Uygulama

This is the main entry point of the Neural Network Visualizer application.
It orchestrates all components: GUI, algorithms, data handling, and visualization.

Bu, Sinir Ağı Görselleştiricisi uygulamasının ana giriş noktasıdır.
Tüm bileşenleri orkestre eder: GUI, algoritmalar, veri yönetimi ve görselleştirme.

Application Architecture / Uygulama Mimarisi:

    NeuralNetworkVisualizer (Main Orchestrator)
        |
        +-- DataHandler (Data Management)
        |   Manages training data points and class labels
        |   Eğitim veri noktalarını ve sınıf etiketlerini yönetir
        |
        +-- ControlPanel (Right Sidebar)
        |   User controls for model selection and hyperparameters
        |   Model seçimi ve hiperparametreler için kullanıcı kontrolleri
        |
        +-- VisualizationFrame (Left Panel)
        |   Interactive plots for training, testing, and loss curves
        |   Eğitim, test ve kayıp eğrileri için interaktif grafikler
        |
        +-- Algorithm Models (Perceptron, Delta Rule, MLP)
            Neural network algorithms implemented from scratch
            Sıfırdan uygulanmış sinir ağı algoritmaları

Key Features / Temel Özellikler:
    - Interactive data point addition via mouse clicks
      Fare tıklamalarıyla interaktif veri noktası ekleme
    - Real-time training visualization with live decision boundaries
      Canlı karar sınırlarıyla gerçek zamanlı eğitim görselleştirmesi
    - Support for 3 algorithms: Perceptron, Delta Rule, MLP
      3 algoritma desteği: Perceptron, Delta Rule, MLP
    - Customizable network architecture and hyperparameters
      Özelleştirilebilir ağ mimarisi ve hiperparametreler
    - Train/test split with performance metrics
      Performans metrikleriyle train/test ayrımı

Author: Developed for educational purposes
Date: 2024
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
    """
    Main Application Class - Neural Network Visualizer
    Ana Uygulama Sınıfı - Sinir Ağı Görselleştiricisi

    This class serves as the orchestrator for the entire application. It:

    1. Initializes the main window and GUI components
       Ana pencereyi ve GUI bileşenlerini başlatır
    2. Manages data collection and storage
       Veri toplama ve depolamayı yönetir
    3. Coordinates training process with visualization
       Eğitim sürecini görselleştirmeyle koordine eder
    4. Handles user interactions and callbacks
       Kullanıcı etkileşimlerini ve callback'leri işler
    5. Updates real-time visualizations during training
       Eğitim sırasında gerçek zamanlı görselleştirmeleri günceller

    Architecture Pattern / Mimari Desen:
        Uses Observer/Callback pattern for event-driven GUI updates.
        Olay odaklı GUI güncellemeleri için Observer/Callback desenini kullanır.

        User Action → Callback → Update Model → Update Visualization
        Kullanıcı Eylemi → Callback → Modeli Güncelle → Görselleştirmeyi Güncelle

    Attributes:
        data_handler (DataHandler): Manages training data points
                                   Eğitim veri noktalarını yönetir
        control_panel (ControlPanel): Right sidebar with controls
                                     Kontroller ile sağ kenar çubuğu
        visualization_frame (VisualizationFrame): Left panel with plots
                                                  Grafiklerle sol panel
        is_training (bool): Flag indicating if training is in progress
                           Eğitimin devam edip etmediğini belirten bayrak
        trained_model: Currently trained model instance
                      Şu anda eğitilmiş model örneği
        current_model: Model being trained (updated each epoch)
                      Eğitilmekte olan model (her epoch güncellenir)
    """

    def __init__(self):
        """
        Initialize the Neural Network Visualizer application.
        Sinir Ağı Görselleştiricisi uygulamasını başlatır.

        Sets up the main window, initializes all components, and establishes
        the event-driven architecture with callback functions.

        Ana pencereyi kurar, tüm bileşenleri başlatır ve callback fonksiyonlarıyla
        olay odaklı mimariyi oluşturur.
        """
        super().__init__()

        # ==================================================================
        # Window Configuration
        # Pencere Yapılandırması
        # ==================================================================
        self.title("🧠 Neural Network Visualizer - MLP From Scratch")
        self.geometry("1400x800")

        # ==================================================================
        # Theme Configuration
        # Tema Yapılandırması
        # ==================================================================
        
        # ==================================================================\n        # Theme Configuration\n        # Tema Yapılandırması\n        # ==================================================================
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # ==================================================================\n        # Data Handler Initialization\n        # Veri Yöneticisi Başlatma\n        # ==================================================================\n        # DataHandler manages all training data points and class labels\n        # DataHandler tüm eğitim veri noktalarını ve sınıf etiketlerini yönetir
        self.data_handler = DataHandler()
        
        # Initialize with 2 classes (binary classification by default)\n        # 2 sınıfla başlat (varsayılan olarak ikili sınıflandırma)
        self.data_handler.add_class("Class 0")
        self.data_handler.add_class("Class 1")
        
        # ==================================================================\n        # Training State Variables\n        # Eğitim Durum Değişkenleri\n        # ==================================================================
        self.is_training = False
        self.trained_model = None
        
        # ==================================================================\n        # Setup User Interface\n        # Kullanıcı Arayüzünü Kur\n        # ==================================================================
        self._setup_ui()
        
        # Update class selection radio buttons in control panel\n        # Kontrol panelindeki sınıf seçim radio button'larını güncelle
        self._update_class_radios()
    
    def _setup_ui(self):
        """
        Setup the user interface with two-panel layout.
        İki panelli düzende kullanıcı arayüzünü kurar.
        
        Creates a grid-based layout with:
        Şunlarla izgara tabanlı bir düzen oluşturur:
            - Left panel (70%): VisualizationFrame for interactive plots
              Sol panel (%70): İnteraktif grafikler için VisualizationFrame
            - Right panel (30%): ControlPanel for user controls
              Sağ panel (%30): Kullanıcı kontrolleri için ControlPanel
        
        Callback Pattern:
        Callback Deseni:
            Connects GUI events to handler methods using callback functions.
            GUI olaylarını callback fonksiyonları kullanarak handler metodlarına bağlar.
        """
        # Configure grid layout with proportional column weights
        # Oranlı sütun ağırlıklarıyla izgara düzenini yapılandır
        self.grid_columnconfigure(0, weight=3)  # Görselleştirme alanı
        self.grid_columnconfigure(1, weight=1)  # Kontrol paneli
        self.grid_rowconfigure(0, weight=1)
        
        # ==================================================================
        # Left Panel: Visualization Frame
        # Sol Panel: Görselleştirme Çerçevesi
        # ==================================================================
        # Contains three tabs: Training, Testing, and Loss plots
        # Üç sekme içerir: Eğitim, Test ve Kayıp grafikleri
        self.visualization_frame = VisualizationFrame(
            self,
            on_point_added_callback=self._on_point_added
        )
        self.visualization_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        # ==================================================================
        # Right Panel: Control Panel
        # Sağ Panel: Kontrol Paneli
        # ==================================================================
        # Contains controls for model selection, hyperparameters, and actions
        # Model seçimi, hiperparametreler ve eylemler için kontroller içerir
        self.control_panel = ControlPanel(
            self,
            on_add_class=self._on_add_class,
            on_remove_class=self._on_remove_class,
            on_clear_data=self._on_clear_data,
            on_start_training=self._on_start_training
        )
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    
    # ==============================================================================
    #                         EVENT HANDLERS (CALLBACKS)
    #                         OLAY YÖNETİCİLERİ (CALLBACK'LER)
    # ==============================================================================
    # These methods are called in response to user interactions with the GUI.
    # They implement the Observer pattern, reacting to events and updating state.
    #
    # Bu metodlar, kullanıcının GUI ile etkileşimlerine yanıt olarak çağrılır.
    # Observer desenini uygular, olaylara tepki verir ve durumu günceller.
    # ==============================================================================
    
    def _on_point_added(self, x, y):
        """
        Callback: Handle mouse click on training plot to add data point.
        Callback: Veri noktası eklemek için eğitim grafiğindeki fare tıklamasını işle.
        
        This callback is triggered when the user clicks on the training
        visualization canvas. It adds a new training point with the currently
        selected class label.
        
        Bu callback, kullanıcı eğitim görselleştirme tuvaline tıkladığında
        tetiklenir. Şu anda seçili sınıf etiketiyle yeni bir eğitim noktası ekler.
        
        Flow / Akış:
            1. Get currently selected class from control panel
               Kontrol panelinden şu anda seçili sınıfı al
            2. Add point to data handler
               Veri yöneticisine nokta ekle
            3. Update visualization to show new point
               Yeni noktayı göstermek için görselleştirmeyi güncelle
        
        Args:
            x (float): X coordinate of mouse click / Fare tıklamasının X koordinatı
            y (float): Y coordinate of mouse click / Fare tıklamasının Y koordinatı
        """
        # Get the currently selected class from control panel radio buttons
        # Kontrol paneli radio butonlarından şu anda seçili sınıfı al
        class_id = self.control_panel.get_selected_class()
        
        # Add the new data point to the data handler
        # Yeni veri noktasını veri yöneticisine ekle
        self.data_handler.add_point(x, y, class_id)
        
        # Refresh the training view to display the new point
        # Yeni noktayı göstermek için eğitim görünümünü yenile
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_add_class(self):
        """
        Callback: Add a new class label to the dataset.
        Callback: Veri setine yeni bir sınıf etiketi ekle.
        
        Allows users to extend the classification problem to multi-class
        (up to 6 classes maximum for visualization clarity).
        
        Kullanıcıların sınıflandırma problemini çok sınıflıya genişletmesine
        izin verir (görselleştirme netliği için maksimum 6 sınıf).
        """
        # Enforce maximum of 6 classes for visualization clarity
        # Görselleştirme netliği için maksimum 6 sınıf uygula
        if self.data_handler.get_num_classes() >= 6:
            messagebox.showwarning("Uyarı", "Maksimum 6 sınıf ekleyebilirsiniz.")
            return
        
        class_name = f"Class {self.data_handler.get_num_classes()}"
        self.data_handler.add_class(class_name)
        self._update_class_radios()
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_remove_class(self):
        """
        Callback: Remove the last class from the dataset.
        Callback: Veri setinden son sınıfı kaldır.
        
        Removes the most recently added class and all its associated
        data points. Maintains minimum of 2 classes for binary classification.
        
        En son eklenen sınıfı ve onunla ilişkili tüm veri noktalarını kaldırır.
        İkili sınıflandırma için minimum 2 sınıfı korur.
        """
        # Enforce minimum of 2 classes for binary classification
        # İkili sınıflandırma için minimum 2 sınıf uygula
        if self.data_handler.get_num_classes() <= 2:
            messagebox.showwarning("Uyarı", "En az 2 sınıf bulunmalıdır.")
            return
        
        self.data_handler.remove_class()
        self._update_class_radios()
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_clear_data(self):
        """
        Callback: Clear all data points from the dataset.
        Callback: Veri setinden tüm veri noktalarını temizle.
        
        Prompts user for confirmation before deleting all training data
        and resetting visualizations.
        
        Tüm eğitim verilerini silmeden ve görselleştirmeleri sıfırlamadan
        önce kullanıcıdan onay ister.
        """
        response = messagebox.askyesno("Onay", "Tüm veri noktalarını silmek istediğinizden emin misiniz?")
        if response:
            self.data_handler.clear_data()
            self.visualization_frame.update_train_view(self.data_handler)
            self.visualization_frame.clear_loss_history()
            self.control_panel.set_status("Veri temizlendi")
    
    def _on_start_training(self):
        """
        Callback: Initialize and start the training process.
        Callback: Eğitim sürecini başlat ve başlat.
        
        This is the main training orchestrator that:
        Bu, şunları yapan ana eğitim orkestratörüdür:
            1. Validates sufficient training data exists
               Yeterli eğitim verisinin varlığını doğrular
            2. Retrieves user-selected model and hyperparameters
               Kullanıcı tarafından seçilen modeli ve hiperparametreleri alır
            3. Prepares train/test split
               Train/test ayrımını hazırlar
            4. Instantiates the selected model (Perceptron/Delta Rule/MLP)
               Seçilen modeli örneklendirir (Perceptron/Delta Rule/MLP)
            5. Launches asynchronous training loop with real-time visualization
               Gerçek zamanlı görselleştirmeyle asenkron eğitim döngüsünü başlatır
        
        The training runs asynchronously to keep the GUI responsive,
        using Tkinter's after() method for non-blocking execution.
        
        Eğitim, GUI'ı duyarlı tutmak için asenkron olarak çalışır,
        bloklamayan yürütme için Tkinter'in after() metodunu kullanır.
        """
        # Prevent concurrent training sessions
        # Eş zamanlı eğitim oturumlarını engelle
        if self.is_training:
            messagebox.showinfo("Info / Bilgi", "Training already in progress. / Eğitim zaten devam ediyor.")
            return
        
        # Validate sufficient training data (minimum 10 points)
        # Yeterli eğitim verisini doğrula (minimum 10 nokta)
        if self.data_handler.get_num_points() < 10:
            messagebox.showwarning("Warning / Uyarı", 
                                  "Please add at least 10 data points. / En az 10 veri noktası ekleyin.")
            return
        
        # ==================================================================
        # Set training state and prepare UI
        # Eğitim durumunu ayarla ve UI'yı hazırla
        # ==================================================================
        self.is_training = True  # Flag to prevent concurrent training / Eş zamanlı eğitimi engelleyen bayrak
        self.control_panel.enable_training(False)  # Disable training button / Eğitim butonunu devre dışı bırak
        self.control_panel.set_status("Training starting... / Eğitim başlıyor...")
        
        # Clear previous training history
        # Önceki eğitim geçmişini temizle
        self.visualization_frame.clear_loss_history()
        
        # ==================================================================
        # Retrieve hyperparameters from control panel
        # Kontrol panelinden hiperparametreleri al
        # ==================================================================
        model_type = self.control_panel.get_model_type()  # Perceptron / Delta Rule / MLP
        learning_rate = self.control_panel.get_learning_rate()  # η (eta)
        epochs = self.control_panel.get_epochs()  # Number of training iterations / Eğitim iterasyon sayısı
        batch_size = self.control_panel.get_batch_size()  # Mini-batch size / Mini-batch boyutu
        test_split = self.control_panel.get_test_split()  # Train/test ratio / Train/test oranı
        
        # ==================================================================
        # Prepare train/test split
        # Train/test ayrımını hazırla
        # ==================================================================
        X_train, X_test, y_train, y_test = self.data_handler.get_train_test_split(
            test_ratio=test_split
        )
        
        # Store test data for evaluation after training
        # Eğitim sonrası değerlendirme için test verisini sakla
        self.X_test = X_test
        self.y_test = y_test
        
        # ==================================================================
        # Model Instantiation based on user selection
        # Kullanıcı seçimine göre Model Örneklendirme
        # ==================================================================
        n_classes = self.data_handler.get_num_classes()
        
        if model_type == "Perceptron":
            # Rosenblatt's Perceptron (1958) - Classic single-layer algorithm
            # Rosenblatt'ın Perceptron'u (1958) - Klasik tek katmanlı algoritma
            model = Perceptron(learning_rate=learning_rate, n_classes=n_classes)
            batch_size = 1  # Perceptron uses online learning (updates after each sample)
                           # Perceptron çevrimiçi öğrenme kullanır (her örnekten sonra günceller)
        
        elif model_type == "DeltaRule":
            # Widrow-Hoff Delta Rule/ADALINE (1960) - Gradient-based learning
            # Widrow-Hoff Delta Rule/ADALINE (1960) - Gradyan tabanlı öğrenme
            model = DeltaRule(learning_rate=learning_rate, n_classes=n_classes)
        
        else:  # MLP
            # Multi-Layer Perceptron with backpropagation
            # Backpropagation ile Çok Katmanlı Perceptron
            architecture = self.control_panel.get_architecture()
            
            # Ensure correct input/output dimensions
            # Doğru girdi/çıktı boyutlarını garanti et
            architecture[0] = 2  # 2D input features (x, y coordinates)
                                # 2D girdi özellikleri (x, y koordinatları)
            architecture[-1] = n_classes  # Output layer size = number of classes
                                         # Çıktı katmanı boyutu = sınıf sayısı
            
            activation_funcs = self.control_panel.get_activation_functions()
            l2_lambda = self.control_panel.get_l2_lambda()
            
            model = MLP(
                layer_dims=architecture,
                activation_funcs=activation_funcs,
                learning_rate=learning_rate,
                l2_lambda=l2_lambda
            )
        
        # ==================================================================
        # Launch asynchronous training
        # Asenkron eğitimi başlat
        # ==================================================================
        # Use Tkinter's after() to schedule training without blocking GUI
        # GUI'yi bloklamadan eğitimi planlamak için Tkinter'in after() metodunu kullan
        self.after(100, lambda: self._run_training(model, X_train, y_train, epochs, batch_size))
    
    def _run_training(self, model, X_train, y_train, epochs, batch_size):
        """
        Run the training loop with real-time visualization updates.
        Gerçek zamanlı görselleştirme güncellemeleriyle eğitim döngüsünü çalıştır.
        
        This method creates a generator from the model's fit() method and
        initiates the epoch-by-epoch training process. The generator pattern
        allows us to yield control back to the GUI after each epoch.
        
        Bu metod, modelin fit() metodundan bir generator oluşturur ve
        epoch-by-epoch eğitim sürecini başlatır. Generator deseni,
        her epoch sonrası kontrolü GUI'ye geri vermemizi sağlar.
        
        Generator Pattern Benefits / Generator Deseni Faydaları:
            - Non-blocking training: GUI remains responsive
              Bloklamayan eğitim: GUI duyarlı kalır
            - Real-time updates: Visualizations update each epoch
              Gerçek zamanlı güncellemeler: Görselleştirmeler her epoch güncellenir
            - Early stopping: User can interrupt training if needed
              Erken durdurma: Kullanıcı gerekirse eğitimi kesebilir
        
        Args:
            model: Model instance to train (Perceptron/DeltaRule/MLP)
                   Eğitilecek model örneği (Perceptron/DeltaRule/MLP)
            X_train (np.ndarray): Training features / Eğitim özellikleri
            y_train (np.ndarray): Training labels / Eğitim etiketleri
            epochs (int): Number of training epochs / Eğitim epoch sayısı
            batch_size (int): Batch size for mini-batch gradient descent
                             Mini-batch gradient descent için batch boyutu
        """
        # Store model reference (will be updated each epoch)
        # Model referansını sakla (her epoch güncellenecek)
        self.current_model = model
        
        # Create generator from model's fit() method
        # Modelin fit() metodundan generator oluştur
        # Note: Perceptron and DeltaRule don't use batch_size parameter
        # Not: Perceptron ve DeltaRule batch_size parametresini kullanmaz
        if isinstance(model, (Perceptron, DeltaRule)):
            fit_generator = model.fit(X_train, y_train, epochs=epochs)
        else:  # MLP
            fit_generator = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        # Start the epoch-by-epoch training loop
        # Epoch-by-epoch eğitim döngüsünü başlat
        self._train_next_epoch(fit_generator, X_train, y_train)
    
    def _train_next_epoch(self, fit_generator, X_train, y_train):
        """
        Train one epoch and update visualizations (recursive async pattern).
        Bir epoch eğit ve görselleştirmeleri güncelle (recursive async pattern).
        
        This method implements a recursive asynchronous training pattern:
        Bu metod, recursive asenkron eğitim desenini uygular:
            1. Get next epoch from generator (trains one epoch)
               Generator'den sonraki epoch'u al (bir epoch eğit)
            2. Update UI with progress (status, loss plot, decision boundary)
               İlerlemeyle UI'yı güncelle (durum, kayıp grafiği, karar sınırı)
            3. Schedule next epoch using after() (non-blocking)
               after() kullanarak sonraki epoch'u planla (bloklamayan)
            4. Repeat until StopIteration (training complete)
               StopIteration'a kadar tekrarla (eğitim tamamlandı)
        
        This pattern keeps the GUI responsive while training progresses.
        Bu desen, eğitim ilerlerken GUI'yı duyarlı tutar.
        
        Args:
            fit_generator: Python generator yielding (epoch, loss, model)
                          (epoch, loss, model) yield eden Python generator
            X_train (np.ndarray): Training features for visualization
                                 Görselleştirme için eğitim özellikleri
            y_train (np.ndarray): Training labels for visualization
                                 Görselleştirme için eğitim etiketleri
        """
        try:
            # ==============================================================
            # Get next epoch results from generator
            # Generator'den sonraki epoch sonuçlarını al
            # ==============================================================
            # This call trains one complete epoch and yields results
            # Bu çağrı bir tam epoch eğitir ve sonuçları verir
            epoch, loss, model = next(fit_generator)
            
            # Update model reference with latest trained state
            # Model referansını en son eğitilmiş durumla güncelle
            self.current_model = model
            
            # ==============================================================
            # Update UI with training progress
            # Eğitim ilerlemesiyle UI'yı güncelle
            # ==============================================================
            # Update status label with current epoch and loss
            # Mevcut epoch ve kayıpla durum etiketini güncelle
            self.control_panel.set_status(f"Epoch {epoch} - Loss: {loss:.4f}")
            
            # Update loss curve plot with new data point
            # Yeni veri noktasıyla kayıp eğrisi grafiğini güncelle
            self.visualization_frame.update_loss_plot(epoch, loss)
            
            # Update decision boundary visualization (every 5 epochs for performance)
            # Karar sınırı görselleştirmesini güncelle (performans için her 5 epoch'ta)
            # Note: Computing decision boundaries is expensive, so we don't do it every epoch
            # Not: Karar sınırlarını hesaplamak pahalıdır, bu yüzden her epoch yapmıyoruz
            if epoch % 5 == 0 or epoch == 1:
                self.visualization_frame.update_decision_boundary(
                    model, X_train, y_train, self.data_handler, tab_name='train'
                )
            
            # Process pending GUI events to keep interface responsive
            # Arayüzü duyarlı tutmak için bekleyen GUI olaylarını işle
            self.update_idletasks()
            
            # ==============================================================
            # Schedule next epoch (recursive async call)
            # Sonraki epoch'u planla (recursive async çağrı)
            # ==============================================================
            # Wait 50ms before next epoch to allow GUI to update
            # Sonraki epoch'tan önce GUI'nin güncellenmesine izin vermek için 50ms bekle
            self.after(50, lambda: self._train_next_epoch(fit_generator, X_train, y_train))
            
        except StopIteration:
            # ==============================================================
            # Training Complete
            # Eğitim Tamamlandı
            # ==============================================================
            # Generator exhausted - all epochs completed
            # Generator tükendi - tüm epoch'lar tamamlandı
            
            # Draw final decision boundary on training data
            # Eğitim verisi üzerinde son karar sınırını çiz
            self.visualization_frame.update_decision_boundary(
                self.current_model, X_train, y_train, self.data_handler, tab_name='train'
            )
            
            # Proceed to evaluation and cleanup
            # Değerlendirme ve temizliğe geç
            self._on_training_completed(self.current_model)
    
    def _on_training_completed(self, model):
        """
        Callback: Handle completion of training process.
        Callback: Eğitim sürecinin tamamlanmasını işle.
        
        This method is called when all training epochs are complete. It:
        Bu metod, tüm eğitim epoch'ları tamamlandığında çağrılır. Şunları yapar:
            1. Resets training state flags
               Eğitim durum bayraklarını sıfırlar
            2. Evaluates model on test set
               Modeli test setinde değerlendirir
            3. Updates visualizations with test results
               Test sonuçlarıyla görselleştirmeleri günceller
            4. Displays final accuracy metrics
               Son doğruluk metriklerini gösterir
            5. Re-enables training controls
               Eğitim kontrollerini yeniden etkinleştirir
        
        Args:
            model: Trained model instance / Eğitilmiş model örneği
        """
        # ==================================================================
        # Reset training state
        # Eğitim durumunu sıfırla
        # ==================================================================
        self.is_training = False  # Allow new training session / Yeni eğitim oturumuna izin ver
        self.trained_model = model  # Store trained model / Eğitilmiş modeli sakla
        self.control_panel.enable_training(True)  # Re-enable training button / Eğitim butonunu yeniden etkinleştir
        
        # ==================================================================
        # Test Set Evaluation
        # Test Seti Değerlendirmesi
        # ==================================================================
        accuracy = None
        if len(self.X_test) > 0:
            # Visualize model's decision boundary on test data
            # Modelin test verisi üzerindeki karar sınırını görselleştir
            self.visualization_frame.update_decision_boundary(
                model, self.X_test, self.y_test, self.data_handler, tab_name='test'
            )
            
            # Compute test accuracy (classification performance metric)
            # Test doğruluğunu hesapla (sınıflandırma performans metriği)
            y_pred = model.predict(self.X_test)  # Get predictions / Tahminleri al
            accuracy = np.mean(y_pred == self.y_test) * 100  # Percentage correct / Yüzde doğru
            
            # Update status with final results
            # Son sonuçlarla durumu güncelle
            self.control_panel.set_status(f"Training complete! Test Accuracy: {accuracy:.2f}%")
            
            # Switch to test tab to show results
            # Sonuçları göstermek için test sekmesine geç
            self.visualization_frame.switch_to_tab('test')
            
            # Display success message with accuracy
            # Doğrulukla başarı mesajını göster
            messagebox.showinfo("Success / Başarılı", 
                               f"Training completed successfully!\n"
                               f"Eğitim başarıyla tamamlandı!\n\n"
                               f"Test Accuracy: {accuracy:.2f}%")
        else:
            # No test data available
            # Test verisi mevcut değil
            self.control_panel.set_status("Training completed! / Eğitim tamamlandı!")
            messagebox.showinfo("Success / Başarılı", 
                               "Training completed successfully!\n"
                               "Eğitim başarıyla tamamlandı!")
    
    def _update_class_radios(self):
        """
        Update class selection radio buttons in control panel.
        Kontrol panelindeki sınıf seçim radio butonlarını güncelle.
        
        Synchronizes the radio buttons with the current class labels
        and colors from the data handler.
        
        Radio butonlarını veri yöneticisinden gelen mevcut sınıf etiketleri
        ve renkleriyle senkronize eder.
        """
        classes = self.data_handler.classes
        colors = [self.data_handler.get_color(i) for i in range(len(classes))]
        self.control_panel.update_class_radios(classes, colors)


def main():
    """
    Application entry point.
    Uygulama giriş noktası.
    
    Creates the main application window and starts the Tkinter event loop.
    This function is called when the script is run directly.
    
    Ana uygulama penceresini oluşturur ve Tkinter olay döngüsünü başlatır.
    Bu fonksiyon, script doğrudan çalıştırıldığında çağrılır.
    
    Args:
        None
    
    Returns:
        None
    """
    app = NeuralNetworkVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()
