"""
Control Panel - User Interface Module
Kontrol Paneli - Kullanıcı Arayüzü Modülü

This module implements the right sidebar control panel containing all user
controls for model selection, hyperparameter configuration, and training actions.

Bu modül, model seçimi, hiperparametre yapılandırması ve eğitim eylemleri
için tüm kullanıcı kontrollerini içeren sağ kenar çubuğu kontrol panelini uygular.

GUI Components / GUI Bileşenleri:
    - Class Management: Add/remove classes, select active class
      Sınıf Yönetimi: Sınıf ekle/çıkar, aktif sınıfı seç
    - Model Selection: Choose between Perceptron, Delta Rule, MLP
      Model Seçimi: Perceptron, Delta Rule, MLP arasında seçim
    - Hyperparameters: Learning rate, epochs, batch size, etc.
      Hiperparametreler: Öğrenme oranı, epoch'lar, batch boyutu, vb.
    - Action Buttons: Clear data, start training
      Eylem Butonları: Veriyi temizle, eğitimi başlat

Author: Developed for educational purposes
Date: 2024
"""

import customtkinter as ctk


class ControlPanel(ctk.CTkFrame):
    """
    Control Panel - Right Sidebar UI Component
    Kontrol Paneli - Sağ Kenar Çubuğu UI Bileşeni

    This class implements the control panel on the right side of the application.
    It provides all user controls and communicates with the main application
    through callback functions.

    Bu sınıf, uygulamanın sağ tarafındaki kontrol panelini uygular.
    Tüm kullanıcı kontrollerini sağlar ve callback fonksiyonları aracılığıyla
    ana uygulamayla iletişim kurar.

    UI Sections / UI Bölümleri:
        1. Class Management Section
           Sınıf Yönetim Bölümü:
           - Add/remove class buttons
             Sınıf ekle/çıkar butonları
           - Radio buttons for class selection
             Sınıf seçimi için radio butonlar

        2. Model Selection Section
           Model Seçim Bölümü:
           - Dropdown menu for algorithm choice
             Algoritma seçimi için açılır menü

        3. Hyperparameters Section
           Hiperparametreler Bölümü:
           - Input fields for all training parameters
             Tüm eğitim parametreleri için giriş alanları

        4. Control Buttons Section
           Kontrol Butonları Bölümü:
           - Clear data button
             Veri temizleme butonu
           - Start training button
             Eğitimi başlatma butonu
           - Status label
             Durum etiketi

    Callback Pattern / Callback Deseni:
        Uses callback functions to communicate with main application,
        implementing loose coupling and separation of concerns.

        Ana uygulamayla iletişim için callback fonksiyonları kullanır,
        gevşek bağlantı ve endişelerin ayrılmasını uygular.
    """

    def __init__(self, master, on_add_class=None, on_remove_class=None,
                 on_clear_data=None, on_start_training=None, **kwargs):
        """
        Initialize the Control Panel with all UI components and callbacks.
        Tüm UI bileşenleri ve callback'lerle Kontrol Panelini başlat.

        Sets up the control panel layout and stores callback functions
        for communication with the main application.

        Kontrol paneli düzenini kurar ve ana uygulamayla iletişim için
        callback fonksiyonlarını saklar.

        Args:
            master: Parent widget (main application window)
                   Üst widget (ana uygulama penceresi)

            on_add_class (callable, optional): Callback when add class button is clicked
                                              Sınıf ekle butonuna tıklandığında callback

            on_remove_class (callable, optional): Callback when remove class button is clicked
                                                 Sınıf çıkar butonuna tıklandığında callback

            on_clear_data (callable, optional): Callback when clear data button is clicked
                                               Veri temizle butonuna tıklandığında callback

            on_start_training (callable, optional): Callback when start training button is clicked
                                                   Eğitimi başlat butonuna tıklandığında callback

            **kwargs: Additional arguments passed to CTkFrame
                     CTkFrame'e geçirilen ek argümanlar
        """
        super().__init__(master, **kwargs)

        self.on_add_class = on_add_class
        self.on_remove_class = on_remove_class
        self.on_clear_data = on_clear_data
        self.on_start_training = on_start_training
        
        self.selected_class = ctk.IntVar(value=0)
        self.class_radio_buttons = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """
        Setup all UI components in the control panel.
        Kontrol panelindeki tüm UI bileşenlerini kur.
        
        Creates and organizes all widgets in a vertical layout with sections:
        Tüm widget'ları bölümlerle dikey düzende oluşturur ve düzenler:
            - Title / Başlık
            - Class Management / Sınıf Yönetimi
            - Model Selection / Model Seçimi
            - Hyperparameters / Hiperparametreler
            - Control Buttons / Kontrol Butonları
        """
        # Başlık
        title_label = ctk.CTkLabel(self, text="⚙️ Kontrol Paneli", 
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10, padx=10)
        
        # ========== Sınıf Yönetimi ==========
        class_frame = ctk.CTkFrame(self)
        class_frame.pack(fill="x", padx=10, pady=5)
        
        class_label = ctk.CTkLabel(class_frame, text="🎨 Sınıf Yönetimi",
                                   font=ctk.CTkFont(size=14, weight="bold"))
        class_label.pack(pady=5)
        
        # Sınıf butonları
        class_btn_frame = ctk.CTkFrame(class_frame)
        class_btn_frame.pack(pady=5)
        
        self.add_class_btn = ctk.CTkButton(class_btn_frame, text="+ Class",
                                          command=self._on_add_class_clicked,
                                          width=100)
        self.add_class_btn.pack(side="left", padx=5)
        
        self.remove_class_btn = ctk.CTkButton(class_btn_frame, text="- Class",
                                             command=self._on_remove_class_clicked,
                                             width=100)
        self.remove_class_btn.pack(side="left", padx=5)
        
        # Sınıf seçimi için radio button'lar
        self.class_radio_frame = ctk.CTkFrame(class_frame)
        self.class_radio_frame.pack(pady=5, fill="x", padx=5)
        
        # ========== Model Seçimi ==========
        model_frame = ctk.CTkFrame(self)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        model_label = ctk.CTkLabel(model_frame, text="🤖 Model Seçimi",
                                   font=ctk.CTkFont(size=14, weight="bold"))
        model_label.pack(pady=5)
        
        self.model_type = ctk.StringVar(value="MLP")
        self.model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=["Single-Layer (Perceptron)", 
                   "Single-Layer (Delta Rule)", 
                   "Multi-Layer (MLP)"],
            variable=self.model_type,
            command=self._on_model_changed
        )
        self.model_menu.pack(pady=5, padx=10, fill="x")
        
        # ========== Hiperparametreler ==========
        hyper_frame = ctk.CTkFrame(self)
        hyper_frame.pack(fill="x", padx=10, pady=5)
        
        hyper_label = ctk.CTkLabel(hyper_frame, text="⚡ Hiperparametreler",
                                   font=ctk.CTkFont(size=14, weight="bold"))
        hyper_label.pack(pady=5)
        
        # Katman Mimarisi
        self.architecture_frame = ctk.CTkFrame(hyper_frame)
        self.architecture_frame.pack(fill="x", padx=10, pady=2)
        
        arch_label = ctk.CTkLabel(self.architecture_frame, text="Katman Mimarisi:")
        arch_label.pack(side="left", padx=5)
        
        self.architecture_entry = ctk.CTkEntry(self.architecture_frame, width=150,
                                              placeholder_text="örn: 2,5,3")
        self.architecture_entry.pack(side="right", padx=5)
        self.architecture_entry.insert(0, "2,5,3")
        
        # Aktivasyon Fonksiyonu - Gizli Katmanlar
        self.activation_hidden_frame = ctk.CTkFrame(hyper_frame)
        self.activation_hidden_frame.pack(fill="x", padx=10, pady=2)
        
        activ_hidden_label = ctk.CTkLabel(self.activation_hidden_frame, 
                                          text="Gizli Katman Aktiv:")
        activ_hidden_label.pack(side="left", padx=5)
        
        self.activation_hidden_var = ctk.StringVar(value="relu")
        self.activation_hidden_menu = ctk.CTkOptionMenu(
            self.activation_hidden_frame,
            values=["relu", "tanh", "sigmoid", "linear"],
            variable=self.activation_hidden_var,
            width=150
        )
        self.activation_hidden_menu.pack(side="right", padx=5)
        
        # Aktivasyon Fonksiyonu - Çıktı Katmanı
        self.activation_output_frame = ctk.CTkFrame(hyper_frame)
        self.activation_output_frame.pack(fill="x", padx=10, pady=2)
        
        activ_output_label = ctk.CTkLabel(self.activation_output_frame, 
                                          text="Çıktı Katman Aktiv:")
        activ_output_label.pack(side="left", padx=5)
        
        self.activation_output_var = ctk.StringVar(value="softmax")
        self.activation_output_menu = ctk.CTkOptionMenu(
            self.activation_output_frame,
            values=["softmax", "sigmoid", "linear"],
            variable=self.activation_output_var,
            width=150
        )
        self.activation_output_menu.pack(side="right", padx=5)
        
        # Öğrenme Oranı
        lr_frame = ctk.CTkFrame(hyper_frame)
        lr_frame.pack(fill="x", padx=10, pady=2)
        
        lr_label = ctk.CTkLabel(lr_frame, text="Öğrenme Oranı:")
        lr_label.pack(side="left", padx=5)
        
        self.learning_rate_entry = ctk.CTkEntry(lr_frame, width=100)
        self.learning_rate_entry.pack(side="right", padx=5)
        self.learning_rate_entry.insert(0, "0.01")
        
        # Epochs
        epochs_frame = ctk.CTkFrame(hyper_frame)
        epochs_frame.pack(fill="x", padx=10, pady=2)
        
        epochs_label = ctk.CTkLabel(epochs_frame, text="Epochs:")
        epochs_label.pack(side="left", padx=5)
        
        self.epochs_entry = ctk.CTkEntry(epochs_frame, width=100)
        self.epochs_entry.pack(side="right", padx=5)
        self.epochs_entry.insert(0, "100")
        
        # Batch Size
        batch_frame = ctk.CTkFrame(hyper_frame)
        batch_frame.pack(fill="x", padx=10, pady=2)
        
        batch_label = ctk.CTkLabel(batch_frame, text="Batch Size:")
        batch_label.pack(side="left", padx=5)
        
        self.batch_size_entry = ctk.CTkEntry(batch_frame, width=100)
        self.batch_size_entry.pack(side="right", padx=5)
        self.batch_size_entry.insert(0, "32")
        
        # L2 Regularization
        l2_frame = ctk.CTkFrame(hyper_frame)
        l2_frame.pack(fill="x", padx=10, pady=2)
        
        l2_label = ctk.CTkLabel(l2_frame, text="L2 Regularization:")
        l2_label.pack(side="left", padx=5)
        
        self.l2_entry = ctk.CTkEntry(l2_frame, width=100)
        self.l2_entry.pack(side="right", padx=5)
        self.l2_entry.insert(0, "0.0")
        
        # Test Split
        test_frame = ctk.CTkFrame(hyper_frame)
        test_frame.pack(fill="x", padx=10, pady=2)
        
        test_label = ctk.CTkLabel(test_frame, text="Test Split (%):")
        test_label.pack(side="left", padx=5)
        
        self.test_split_entry = ctk.CTkEntry(test_frame, width=100)
        self.test_split_entry.pack(side="right", padx=5)
        self.test_split_entry.insert(0, "20")
        
        # ========== Kontrol Butonları ==========
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        self.clear_btn = ctk.CTkButton(control_frame, text="🗑️ Clear Data",
                                      command=self._on_clear_data_clicked,
                                      fg_color="#E74C3C", hover_color="#C0392B")
        self.clear_btn.pack(pady=5, padx=10, fill="x")
        
        self.train_btn = ctk.CTkButton(control_frame, text="▶️ START TRAINING",
                                      command=self._on_start_training_clicked,
                                      fg_color="#27AE60", hover_color="#229954",
                                      font=ctk.CTkFont(size=14, weight="bold"))
        self.train_btn.pack(pady=5, padx=10, fill="x")
        
        # Durum etiketi
        self.status_label = ctk.CTkLabel(control_frame, text="Hazır",
                                        font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=5)
    
    def _on_add_class_clicked(self):
        """Sınıf ekleme butonu tıklandığında."""
        if self.on_add_class:
            self.on_add_class()
    
    def _on_remove_class_clicked(self):
        """Sınıf silme butonu tıklandığında."""
        if self.on_remove_class:
            self.on_remove_class()
    
    def _on_clear_data_clicked(self):
        """Veri temizleme butonu tıklandığında."""
        if self.on_clear_data:
            self.on_clear_data()
    
    def _on_start_training_clicked(self):
        """Eğitim başlatma butonu tıklandığında."""
        if self.on_start_training:
            self.on_start_training()
    
    def _on_model_changed(self, choice):
        """Model seçimi değiştiğinde."""
        # MLP dışı modeller için bazı parametreleri gizle/göster
        if "Multi-Layer" in choice:
            self.architecture_frame.pack(fill="x", padx=10, pady=2)
            self.activation_hidden_frame.pack(fill="x", padx=10, pady=2)
            self.activation_output_frame.pack(fill="x", padx=10, pady=2)
        else:
            self.architecture_frame.pack_forget()
            self.activation_hidden_frame.pack_forget()
            self.activation_output_frame.pack_forget()
    
    def update_class_radios(self, classes, colors):
        """
        Update class selection radio buttons dynamically.
        Sınıf seçim radio butonlarını dinamik olarak güncelle.
        
        Recreates radio buttons when classes are added or removed,
        ensuring UI stays synchronized with data state.
        
        Sınıflar eklenip çıkarıldığında radio butonları yeniden oluşturur,
        UI'nın veri durumuyla senkronize kalmasını sağlar.
        
        Args:
            classes (list): List of class names / Sınıf adları listesi
            colors (list): List of hex color codes for each class / Her sınıf için hex renk kodları listesi
        """
        # Eski radio button'ları temizle
        for widget in self.class_radio_frame.winfo_children():
            widget.destroy()
        self.class_radio_buttons = []
        
        # Yeni radio button'lar oluştur
        for i, (class_name, color) in enumerate(zip(classes, colors)):
            radio = ctk.CTkRadioButton(
                self.class_radio_frame,
                text=f"  {class_name}",
                variable=self.selected_class,
                value=i,
                fg_color=color,
                border_color=color
            )
            radio.pack(anchor="w", padx=10, pady=2)
            self.class_radio_buttons.append(radio)
        
        # İlk sınıfı seçili yap
        if len(classes) > 0:
            self.selected_class.set(0)
    
    def get_selected_class(self):
        """
        Get the currently selected class ID.
        Şu anda seçili sınıf ID'sini al.
        
        Returns:
            int: Selected class index / Seçili sınıf indeksi
        """
        return self.selected_class.get()
    
    def get_model_type(self):
        """
        Get the selected model type from dropdown.
        Açılır menüden seçilen model tipini al.
        
        Returns:
            str: 'Perceptron', 'DeltaRule', or 'MLP'
        """
        model_str = self.model_menu.get()
        if "Perceptron" in model_str:
            return "Perceptron"
        elif "Delta Rule" in model_str:
            return "DeltaRule"
        else:
            return "MLP"
    
    def get_architecture(self):
        """Katman mimarisini liste olarak döndürür."""
        try:
            arch_str = self.architecture_entry.get()
            return [int(x.strip()) for x in arch_str.split(',')]
        except:
            return [2, 5, 3]
    
    def get_activation_functions(self):
        """Aktivasyon fonksiyonlarını liste olarak döndürür."""
        try:
            hidden_activation = self.activation_hidden_var.get()
            output_activation = self.activation_output_var.get()
            return [hidden_activation, output_activation]
        except:
            return ['relu', 'softmax']
    
    def get_learning_rate(self):
        """Öğrenme oranını döndürür."""
        try:
            return float(self.learning_rate_entry.get())
        except:
            return 0.01
    
    def get_epochs(self):
        """Epoch sayısını döndürür."""
        try:
            return int(self.epochs_entry.get())
        except:
            return 100
    
    def get_batch_size(self):
        """Batch size'ı döndürür."""
        try:
            return int(self.batch_size_entry.get())
        except:
            return 32
    
    def get_l2_lambda(self):
        """L2 regularization katsayısını döndürür."""
        try:
            return float(self.l2_entry.get())
        except:
            return 0.0
    
    def get_test_split(self):
        """Test split oranını döndürür (0-1 arası)."""
        try:
            return float(self.test_split_entry.get()) / 100.0
        except:
            return 0.2
    
    def set_status(self, status_text):
        """
        Update the status label text.
        Durum etiketi metnini güncelle.
        
        Used to display training progress and messages to the user.
        Eğitim ilerlemesini ve mesajları kullanıcıya göstermek için kullanılır.
        
        Args:
            status_text (str): Text to display / Gösterilecek metin
        """
        self.status_label.configure(text=status_text)
    
    def enable_training(self, enabled=True):
        """
        Enable or disable the training button.
        Eğitim butonunu etkinleştir veya devre dışı bırak.
        
        Used to prevent multiple concurrent training sessions.
        Birden fazla eş zamanlı eğitim oturumunu önlemek için kullanılır.
        
        Args:
            enabled (bool): True to enable, False to disable / Etkinleştirmek için True, devre dışı bırakmak için False
        """
        if enabled:
            self.train_btn.configure(state="normal")
        else:
            self.train_btn.configure(state="disabled")
