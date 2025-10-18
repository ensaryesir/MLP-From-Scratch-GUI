"""
Control panel UI - right sidebar with hyperparameters.
"""

import customtkinter as ctk


class ControlPanel(ctk.CTkFrame):
    """Right sidebar with all controls and hyperparameter inputs."""

    def __init__(self, master, on_add_class=None, on_remove_class=None,
                 on_clear_data=None, on_start_training=None, **kwargs):
        super().__init__(master, **kwargs)

        self.on_add_class = on_add_class
        self.on_remove_class = on_remove_class
        self.on_clear_data = on_clear_data
        self.on_start_training = on_start_training
        
        self.selected_class = ctk.IntVar(value=0)
        self.class_radio_buttons = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup all UI widgets."""
        # title
        title_label = ctk.CTkLabel(self, text="⚙️ Control Panel", 
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10, padx=10)
        
        # Class Management
        class_frame = ctk.CTkFrame(self)
        class_frame.pack(fill="x", padx=10, pady=5)
        
        class_label = ctk.CTkLabel(class_frame, text="🎨 Class Management",
                                   font=ctk.CTkFont(size=14, weight="bold"))
        class_label.pack(pady=5)
        
        # class buttons
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
        
        # radio buttons
        self.class_radio_frame = ctk.CTkFrame(class_frame)
        self.class_radio_frame.pack(pady=5, fill="x", padx=5)
        
        # Model Selection
        model_frame = ctk.CTkFrame(self)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        model_label = ctk.CTkLabel(model_frame, text="🤖 Model Selection",
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
        
        # Hyperparameters
        hyper_frame = ctk.CTkFrame(self)
        hyper_frame.pack(fill="x", padx=10, pady=5)
        
        hyper_label = ctk.CTkLabel(hyper_frame, text="⚡ Hyperparameters",
                                   font=ctk.CTkFont(size=14, weight="bold"))
        hyper_label.pack(pady=5)
        
        # layer architecture
        self.architecture_frame = ctk.CTkFrame(hyper_frame)
        self.architecture_frame.pack(fill="x", padx=10, pady=2)
        
        arch_label = ctk.CTkLabel(self.architecture_frame, text="Layer Architecture:")
        arch_label.pack(side="left", padx=5)
        
        self.architecture_entry = ctk.CTkEntry(self.architecture_frame, width=150,
                                              placeholder_text="e.g.: 2,5,3")
        self.architecture_entry.pack(side="right", padx=5)
        self.architecture_entry.insert(0, "2,5,3")
        
        # hidden activation
        self.activation_hidden_frame = ctk.CTkFrame(hyper_frame)
        self.activation_hidden_frame.pack(fill="x", padx=10, pady=2)
        
        activ_hidden_label = ctk.CTkLabel(self.activation_hidden_frame, 
                                          text="Hidden Layer Activ:")
        activ_hidden_label.pack(side="left", padx=5)
        
        self.activation_hidden_var = ctk.StringVar(value="relu")
        self.activation_hidden_menu = ctk.CTkOptionMenu(
            self.activation_hidden_frame,
            values=["relu", "tanh", "sigmoid", "linear"],
            variable=self.activation_hidden_var,
            width=150
        )
        self.activation_hidden_menu.pack(side="right", padx=5)
        
        # output activation
        self.activation_output_frame = ctk.CTkFrame(hyper_frame)
        self.activation_output_frame.pack(fill="x", padx=10, pady=2)
        
        activ_output_label = ctk.CTkLabel(self.activation_output_frame, 
                                          text="Output Layer Activ:")
        activ_output_label.pack(side="left", padx=5)
        
        self.activation_output_var = ctk.StringVar(value="softmax")
        self.activation_output_menu = ctk.CTkOptionMenu(
            self.activation_output_frame,
            values=["softmax", "sigmoid", "linear"],
            variable=self.activation_output_var,
            width=150
        )
        self.activation_output_menu.pack(side="right", padx=5)
        
        # learning rate
        lr_frame = ctk.CTkFrame(hyper_frame)
        lr_frame.pack(fill="x", padx=10, pady=2)
        
        lr_label = ctk.CTkLabel(lr_frame, text="Learning Rate:")
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
        
        # Control Buttons
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
        
        # status label
        self.status_label = ctk.CTkLabel(control_frame, text="Ready",
                                        font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=5)
    
    def _on_add_class_clicked(self):
        if self.on_add_class:
            self.on_add_class()
    
    def _on_remove_class_clicked(self):
        if self.on_remove_class:
            self.on_remove_class()
    
    def _on_clear_data_clicked(self):
        if self.on_clear_data:
            self.on_clear_data()
    
    def _on_start_training_clicked(self):
        if self.on_start_training:
            self.on_start_training()
    
    def _on_model_changed(self, choice):
        """Show/hide MLP-specific params."""
        if "Multi-Layer" in choice:
            self.architecture_frame.pack(fill="x", padx=10, pady=2)
            self.activation_hidden_frame.pack(fill="x", padx=10, pady=2)
            self.activation_output_frame.pack(fill="x", padx=10, pady=2)
        else:
            self.architecture_frame.pack_forget()
            self.activation_hidden_frame.pack_forget()
            self.activation_output_frame.pack_forget()
    
    def update_class_radios(self, classes, colors):
        """Recreate radio buttons when classes change."""
        # clear old
        for widget in self.class_radio_frame.winfo_children():
            widget.destroy()
        self.class_radio_buttons = []
        
        # create new
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
        
        # select first
        if len(classes) > 0:
            self.selected_class.set(0)
    
    def get_selected_class(self):
        return self.selected_class.get()
    
    def get_model_type(self):
        model_str = self.model_menu.get()
        if "Perceptron" in model_str:
            return "Perceptron"
        elif "Delta Rule" in model_str:
            return "DeltaRule"
        else:
            return "MLP"
    
    def get_architecture(self):
        try:
            arch_str = self.architecture_entry.get()
            return [int(x.strip()) for x in arch_str.split(',')]
        except:
            return [2, 5, 3]
    
    def get_activation_functions(self):
        try:
            hidden_activation = self.activation_hidden_var.get()
            output_activation = self.activation_output_var.get()
            return [hidden_activation, output_activation]
        except:
            return ['relu', 'softmax']
    
    def get_learning_rate(self):
        try:
            return float(self.learning_rate_entry.get())
        except:
            return 0.01
    
    def get_epochs(self):
        try:
            return int(self.epochs_entry.get())
        except:
            return 100
    
    def get_batch_size(self):
        try:
            return int(self.batch_size_entry.get())
        except:
            return 32
    
    def get_l2_lambda(self):
        try:
            return float(self.l2_entry.get())
        except:
            return 0.0
    
    def get_test_split(self):
        try:
            return float(self.test_split_entry.get()) / 100.0
        except:
            return 0.2
    
    def set_status(self, status_text):
        self.status_label.configure(text=status_text)
    
    def enable_training(self, enabled=True):
        if enabled:
            self.train_btn.configure(state="normal")
        else:
            self.train_btn.configure(state="disabled")
