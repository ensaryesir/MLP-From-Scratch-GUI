import customtkinter as ctk
from config import UI_SAFETY_DEFAULTS

class ControlPanelInputs:
    def __init__(self, control_panel):
        self.cp = control_panel

    def get_architecture(self):
        try:
            arch_str = self.cp.architecture_entry.get()
            arch = [int(x.strip()) for x in arch_str.split(',')]
            
            task = self.cp.task_type.get()
            
            def update_ui(new_arch):
                self.cp.architecture_entry.delete(0, 'end')
                self.cp.architecture_entry.insert(0, ','.join(map(str, new_arch)))

            if task == "Regression":
                if arch[-1] != 1:
                    print(f"Warning: Regression requires output dim 1 (got {arch[-1]}). Auto-fixing.")
                    arch[-1] = 1
                    update_ui(arch)
            
            elif task == "Classification" and self.cp.dataset_mode.get() == 'Manual':
                num_classes = len(self.cp.class_radio_buttons)
                if num_classes > 0 and arch[-1] != num_classes:
                    print(f"Update: Setting output dim to {num_classes} to match classes.")
                    arch[-1] = num_classes
                    update_ui(arch)

            return arch
        except:
            return UI_SAFETY_DEFAULTS['architecture']

    def get_learning_rate(self):
        try:
            return float(self.cp.learning_rate_entry.get())
        except:
            return UI_SAFETY_DEFAULTS['learning_rate']
            
    def get_epochs(self):
        try:
            return int(self.cp.epochs_entry.get())
        except:
            return UI_SAFETY_DEFAULTS['epochs']
    
    def get_stopping_criteria(self):
        criteria = self.cp.stopping_criteria.get()
        try:
            max_epochs = int(self.cp.epochs_entry.get()) if criteria == "epochs" else 10000000
        except:
            max_epochs = 10000
        try:
            min_error = float(self.cp.min_error_entry.get()) if criteria == "error" else 0.0
        except:
            min_error = UI_SAFETY_DEFAULTS['min_error']
        return criteria, max_epochs, min_error

    def get_batch_size(self):
        try:
            return int(self.cp.batch_size_entry.get())
        except:
            return UI_SAFETY_DEFAULTS['batch_size']
        
    def get_l2_lambda(self):
        try:
            return float(self.cp.l2_entry.get())
        except:
            return UI_SAFETY_DEFAULTS['l2_lambda']
    
    def get_test_split(self):
        try:
            val = float(self.cp.test_split_entry.get())
            return val / 100.0
        except:
            return UI_SAFETY_DEFAULTS['test_split'] / 100.0
    
    def get_momentum_config(self):
        try:
            factor = float(self.cp.momentum_entry.get())
        except:
            factor = 0.9
        
        checkbox_value = self.cp.use_momentum_var.get()
        
        if checkbox_value:
            return True, factor
        else:
            return True, 0.0

    def get_encoder_architecture(self):
        try:
            arch_str = self.cp.encoder_architecture_entry.get()
            return [int(x.strip()) for x in arch_str.split(',')]
        except:
            return UI_SAFETY_DEFAULTS['encoder_architecture']
    
    def get_ae_epochs(self):
        try:
            return int(self.cp.ae_epochs_entry.get())
        except:
            return UI_SAFETY_DEFAULTS['ae_epochs']
            
    def get_ae_stopping_config(self):
        criteria = self.cp.ae_stopping_criteria.get()
        try:
            epochs = int(self.cp.ae_epochs_entry.get())
        except:
            epochs = UI_SAFETY_DEFAULTS['ae_epochs']
            
        try:
            min_error = float(self.cp.ae_min_error_entry.get())
        except:
            min_error = UI_SAFETY_DEFAULTS.get('ae_min_error', 0.001)
            
        if criteria == 'epochs':
            return 'epochs', epochs, 0.0
        else:
            return 'error', 10000, min_error
    
    def get_freeze_encoder(self):
        return self.cp.freeze_encoder_var.get()
    
    def get_recon_samples(self):
        try:
            return int(self.cp.recon_samples_entry.get())
        except:
            return UI_SAFETY_DEFAULTS['recon_samples']

    def get_activation_functions(self):
        try:
            hidden_activation = self.cp.activation_hidden_var.get()
            output_activation = self.cp.activation_output_var.get()
            return [hidden_activation, output_activation]
        except:
            return ['relu', 'softmax']



class ControlPanel(ctk.CTkFrame):
    def __init__(self, parent, on_add_class=None, 
                 on_remove_class=None, on_clear_data=None, 
                 on_start_training=None, on_stop_training=None, on_task_changed_callback=None,
                 on_dataset_changed_callback=None,
                 on_generate_xor=None, on_generate_circles=None,
                 on_generate_moons=None, on_generate_blobs=None,
                 on_generate_sine=None, on_generate_parabola=None,
                 on_generate_linear=None, on_generate_abs=None, 
                 on_save_model=None, on_load_model=None, on_test_handwriting=None,
                 on_save_encoder=None, on_load_encoder=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.on_add_class = on_add_class
        self.on_remove_class = on_remove_class
        self.on_clear_data = on_clear_data
        self.on_start_training = on_start_training
        self.on_stop_training = on_stop_training
        self.on_save_model = on_save_model
        self.on_load_model = on_load_model
        self.on_test_handwriting = on_test_handwriting
        self.on_save_encoder = on_save_encoder
        self.on_load_encoder = on_load_encoder
        self.on_task_changed_callback = on_task_changed_callback
        self.on_dataset_changed_callback = on_dataset_changed_callback

        self.on_generate_xor = on_generate_xor
        self.on_generate_circles = on_generate_circles
        self.on_generate_moons = on_generate_moons
        self.on_generate_blobs = on_generate_blobs
        
        self.on_generate_sine = on_generate_sine
        self.on_generate_parabola = on_generate_parabola
        self.on_generate_linear = on_generate_linear
        self.on_generate_abs = on_generate_abs
        
        self.selected_class = ctk.IntVar(value=0)
        self.class_radio_buttons = []
        self.dataset_mode = ctk.StringVar(value="Manual")
        
        self.inputs = ControlPanelInputs(self)
        
        self._setup_ui()
    
    def _setup_ui(self):
        title_label = ctk.CTkLabel(self, text="⚙️ Control Panel", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10, padx=10)
        
        task_frame = ctk.CTkFrame(self)
        task_frame.pack(fill="x", padx=10, pady=5)
        
        task_label = ctk.CTkLabel(task_frame, text="🎯 Task Type", font=ctk.CTkFont(size=14, weight="bold"))
        task_label.pack(pady=5)
        
        self.task_type = ctk.StringVar(value="Classification")
        self.task_switch = ctk.CTkSegmentedButton(task_frame, values=["Classification", "Regression"], variable=self.task_type, command=self._on_task_changed)
        self.task_switch.pack(pady=5, padx=10, fill="x")

        dataset_frame = ctk.CTkFrame(self)
        dataset_frame.pack(fill="x", padx=10, pady=5)

        dataset_label = ctk.CTkLabel(dataset_frame, text="📚 Dataset", font=ctk.CTkFont(size=14, weight="bold"))
        dataset_label.pack(pady=5)

        self.dataset_switch = ctk.CTkSegmentedButton(dataset_frame, values=["Manual", "MNIST"], variable=self.dataset_mode, command=self._on_dataset_changed)
        self.dataset_switch.pack(pady=5, padx=10, fill="x")

        self.class_management_frame = ctk.CTkFrame(self)
        self.class_management_frame.pack(fill="x", padx=10, pady=5)
        
        self.class_label = ctk.CTkLabel(self.class_management_frame, text="🎨 Class Management", font=ctk.CTkFont(size=14, weight="bold"))
        self.class_label.pack(pady=5)
        
        class_btn_frame = ctk.CTkFrame(self.class_management_frame)
        class_btn_frame.pack(pady=5)
        
        self.add_class_btn = ctk.CTkButton(class_btn_frame, text="+ Class", command=self._on_add_class_clicked, width=100)
        self.add_class_btn.pack(side="left", padx=5)
        
        self.remove_class_btn = ctk.CTkButton(class_btn_frame, text="- Class", command=self._on_remove_class_clicked, width=100)
        self.remove_class_btn.pack(side="left", padx=5)
        
        self.class_radio_frame = ctk.CTkFrame(self.class_management_frame)
        self.class_radio_frame.pack(pady=5, fill="x", padx=5)

        self.preset_classification_frame = ctk.CTkFrame(self.class_management_frame)
        self.preset_classification_frame.pack(fill="x", padx=5, pady=5)
        
        preset_label_cls = ctk.CTkLabel(self.preset_classification_frame, text="⚡ Classification Presets", font=ctk.CTkFont(size=12, weight="bold"))
        preset_label_cls.pack(pady=2)
        
        preset_grid_cls = ctk.CTkFrame(self.preset_classification_frame)
        preset_grid_cls.pack(fill="x", padx=2, pady=2)
        
        self.btn_xor = ctk.CTkButton(preset_grid_cls, text="XOR", width=60, command=self._on_xor_clicked)
        self.btn_xor.grid(row=0, column=0, padx=2, pady=2)
        
        self.btn_circle = ctk.CTkButton(preset_grid_cls, text="Circles", width=60, command=self._on_circles_clicked)
        self.btn_circle.grid(row=0, column=1, padx=2, pady=2)
        
        self.btn_moon = ctk.CTkButton(preset_grid_cls, text="Moons", width=60, command=self._on_moons_clicked)
        self.btn_moon.grid(row=1, column=0, padx=2, pady=2)
        
        self.btn_blob = ctk.CTkButton(preset_grid_cls, text="Blobs", width=60, command=self._on_blobs_clicked)
        self.btn_blob.grid(row=1, column=1, padx=2, pady=2)
        
        self.preset_regression_frame = ctk.CTkFrame(self.class_management_frame)
        
        preset_label_reg = ctk.CTkLabel(self.preset_regression_frame, text="⚡ Regression Presets", font=ctk.CTkFont(size=12, weight="bold"))
        preset_label_reg.pack(pady=2)
        
        preset_grid_reg = ctk.CTkFrame(self.preset_regression_frame)
        preset_grid_reg.pack(fill="x", padx=2, pady=2)
        
        self.btn_sine = ctk.CTkButton(preset_grid_reg, text="Sine", width=60, command=self._on_sine_clicked)
        self.btn_sine.grid(row=0, column=0, padx=2, pady=2)
        
        self.btn_parabola = ctk.CTkButton(preset_grid_reg, text="Parabola", width=60, command=self._on_parabola_clicked)
        self.btn_parabola.grid(row=0, column=1, padx=2, pady=2)
        
        self.btn_linear = ctk.CTkButton(preset_grid_reg, text="Linear", width=60, command=self._on_linear_clicked)
        self.btn_linear.grid(row=1, column=0, padx=2, pady=2)
        
        self.btn_abs = ctk.CTkButton(preset_grid_reg, text="Abs(x)", width=60, command=self._on_abs_clicked)
        self.btn_abs.grid(row=1, column=1, padx=2, pady=2)
        
        model_frame = ctk.CTkFrame(self)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        model_label = ctk.CTkLabel(model_frame, text="🤖 Model Selection", font=ctk.CTkFont(size=14, weight="bold"))
        model_label.pack(pady=5)
        
        self.model_type = ctk.StringVar(value="Single-Layer (Perceptron)")
        self.model_menu = ctk.CTkOptionMenu(model_frame, values=["Single-Layer (Perceptron)", "Single-Layer (Delta Rule)", "Multi-Layer (MLP)"], variable=self.model_type, command=self._on_model_changed)
        self.model_menu.pack(pady=5, padx=10, fill="x")
        
        hyper_frame = ctk.CTkScrollableFrame(self, label_text="⚡ Hyperparameters", label_font=ctk.CTkFont(size=14, weight="bold"))
        hyper_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.architecture_frame = ctk.CTkFrame(hyper_frame)
        self.architecture_frame.pack(fill="x", padx=10, pady=2)
        
        arch_label = ctk.CTkLabel(self.architecture_frame, text="Layer Architecture:")
        arch_label.pack(side="left", padx=5)
        
        self.architecture_entry = ctk.CTkEntry(self.architecture_frame, width=150, placeholder_text="e.g.: 2,5,3")
        self.architecture_entry.pack(side="right", padx=5)
        self.architecture_entry.insert(0, "2,5,3")
        
        self.activation_hidden_frame = ctk.CTkFrame(hyper_frame)
        self.activation_hidden_frame.pack(fill="x", padx=10, pady=2)
        
        activ_hidden_label = ctk.CTkLabel(self.activation_hidden_frame, text="Hidden Layer Activ:")
        activ_hidden_label.pack(side="left", padx=5)
        
        self.activation_hidden_var = ctk.StringVar(value="relu")
        self.activation_hidden_menu = ctk.CTkOptionMenu(self.activation_hidden_frame, values=["relu", "tanh", "sigmoid", "linear"], variable=self.activation_hidden_var, width=150)
        self.activation_hidden_menu.pack(side="right", padx=5)
        
        self.activation_output_frame = ctk.CTkFrame(hyper_frame)
        self.activation_output_frame.pack(fill="x", padx=10, pady=2)
        
        activ_output_label = ctk.CTkLabel(self.activation_output_frame, text="Output Layer Activ:")
        activ_output_label.pack(side="left", padx=5)
        
        self.activation_output_var = ctk.StringVar(value="softmax")
        self.activation_output_menu = ctk.CTkOptionMenu(self.activation_output_frame, values=["softmax", "sigmoid", "linear"], variable=self.activation_output_var, width=150)
        self.activation_output_menu.pack(side="right", padx=5)
        
        lr_frame = ctk.CTkFrame(hyper_frame)
        lr_frame.pack(fill="x", padx=10, pady=2)
        
        lr_label = ctk.CTkLabel(lr_frame, text="Learning Rate:")
        lr_label.pack(side="left", padx=5)
        
        self.learning_rate_entry = ctk.CTkEntry(lr_frame, width=100)
        self.learning_rate_entry.pack(side="right", padx=5)
        self.learning_rate_entry.insert(0, "0.1")
        
        stopping_frame = ctk.CTkFrame(hyper_frame)
        stopping_frame.pack(fill="x", padx=10, pady=5)
        
        stopping_label = ctk.CTkLabel(stopping_frame, text="🛑 Stopping Criteria", font=ctk.CTkFont(size=12, weight="bold"))
        stopping_label.pack(pady=2)
        
        self.stopping_criteria = ctk.StringVar(value="error")
        stopping_radio_frame = ctk.CTkFrame(stopping_frame)
        stopping_radio_frame.pack(pady=2)
        
        epochs_radio = ctk.CTkRadioButton(stopping_radio_frame, text="Max Epochs", variable=self.stopping_criteria, value="epochs", command=self._on_stopping_criteria_changed)
        epochs_radio.pack(side="left", padx=5)
        
        error_radio = ctk.CTkRadioButton(stopping_radio_frame, text="Min Error", variable=self.stopping_criteria, value="error", command=self._on_stopping_criteria_changed)
        error_radio.pack(side="left", padx=5)
        
        self.epochs_frame = ctk.CTkFrame(stopping_frame)
        self.epochs_frame.pack(fill="x", padx=10, pady=2)
        
        epochs_label = ctk.CTkLabel(self.epochs_frame, text="Max Epochs:")
        epochs_label.pack(side="left", padx=5)
        
        self.epochs_entry = ctk.CTkEntry(self.epochs_frame, width=100)
        self.epochs_entry.pack(side="right", padx=5)
        self.epochs_entry.insert(0, "100")
        
        self.min_error_frame = ctk.CTkFrame(stopping_frame)
        self.min_error_frame.pack(fill="x", padx=10, pady=2)
        
        min_error_label = ctk.CTkLabel(self.min_error_frame, text="Min Error:")
        min_error_label.pack(side="left", padx=5)
        
        self.min_error_entry = ctk.CTkEntry(self.min_error_frame, width=100)
        self.min_error_entry.pack(side="right", padx=5)
        self.min_error_entry.insert(0, "0.001")
        
        self.min_error_frame.pack_forget()
        
        self.batch_frame = ctk.CTkFrame(hyper_frame)
        self.batch_frame.pack(fill="x", padx=10, pady=2)
        
        batch_label = ctk.CTkLabel(self.batch_frame, text="Batch Size:")
        batch_label.pack(side="left", padx=5)
        
        self.batch_size_entry = ctk.CTkEntry(self.batch_frame, width=100)
        self.batch_size_entry.pack(side="right", padx=5)
        self.batch_size_entry.insert(0, "32")
        
        self.l2_frame = ctk.CTkFrame(hyper_frame)
        self.l2_frame.pack(fill="x", padx=10, pady=2)
        
        l2_label = ctk.CTkLabel(self.l2_frame, text="L2 Regularization:")
        l2_label.pack(side="left", padx=5)
        
        self.l2_entry = ctk.CTkEntry(self.l2_frame, width=100)
        self.l2_entry.pack(side="right", padx=5)
        self.l2_entry.insert(0, "0.0")
        
        self.momentum_frame = ctk.CTkFrame(hyper_frame)
        self.momentum_frame.pack(fill="x", padx=10, pady=2)
        
        self.use_momentum_var = ctk.BooleanVar(value=False)
        self.momentum_check = ctk.CTkCheckBox(self.momentum_frame, text="Use Momentum", variable=self.use_momentum_var, width=20)
        self.momentum_check.pack(side="left", padx=5)
        
        self.momentum_entry = ctk.CTkEntry(self.momentum_frame, width=80)
        self.momentum_entry.pack(side="right", padx=5)
        self.momentum_entry.insert(0, "0.9")
        
        mom_label = ctk.CTkLabel(self.momentum_frame, text="Factor:")
        mom_label.pack(side="right", padx=2)
        
        self.autoencoder_frame = ctk.CTkFrame(hyper_frame)
        
        ae_title = ctk.CTkLabel(self.autoencoder_frame, text="🔧 Autoencoder Config", font=ctk.CTkFont(size=12, weight="bold"))
        ae_title.pack(pady=2)
        
        encoder_arch_frame = ctk.CTkFrame(self.autoencoder_frame)
        encoder_arch_frame.pack(fill="x", padx=10, pady=2)
        
        encoder_arch_label = ctk.CTkLabel(encoder_arch_frame, text="Encoder Layers:")
        encoder_arch_label.pack(side="left", padx=5)
        
        self.encoder_architecture_entry = ctk.CTkEntry(encoder_arch_frame, width=150, placeholder_text="e.g.: 784,128,32")
        self.encoder_architecture_entry.pack(side="right", padx=5)
        self.encoder_architecture_entry.insert(0, "784,128,32")
        
        ae_stop_frame = ctk.CTkFrame(self.autoencoder_frame)
        ae_stop_frame.pack(fill="x", padx=10, pady=2)
        
        ae_stop_label = ctk.CTkLabel(ae_stop_frame, text="Stop AE Train By:")
        ae_stop_label.pack(side="left", padx=5)
        
        self.ae_stopping_criteria = ctk.StringVar(value="epochs")
        self.ae_stopping_switch = ctk.CTkSegmentedButton(
            ae_stop_frame, 
            values=["epochs", "error"], 
            variable=self.ae_stopping_criteria, 
            command=self._on_ae_stopping_changed,
            width=100
        )
        self.ae_stopping_switch.pack(side="right", padx=5)

        self.ae_epochs_frame = ctk.CTkFrame(self.autoencoder_frame)
        self.ae_epochs_frame.pack(fill="x", padx=10, pady=2)
        
        ae_epochs_label = ctk.CTkLabel(self.ae_epochs_frame, text="AE Pre-train Epochs:")
        ae_epochs_label.pack(side="left", padx=5)
        
        self.ae_epochs_entry = ctk.CTkEntry(self.ae_epochs_frame, width=100)
        self.ae_epochs_entry.pack(side="right", padx=5)
        self.ae_epochs_entry.insert(0, "50")
        
        self.ae_min_error_frame = ctk.CTkFrame(self.autoencoder_frame)
        
        ae_error_label = ctk.CTkLabel(self.ae_min_error_frame, text="AE Min Error:")
        ae_error_label.pack(side="left", padx=5)
        
        self.ae_min_error_entry = ctk.CTkEntry(self.ae_min_error_frame, width=100)
        self.ae_min_error_entry.pack(side="right", padx=5)
        self.ae_min_error_entry.insert(0, "0.001")
        
        freeze_frame = ctk.CTkFrame(self.autoencoder_frame)
        freeze_frame.pack(fill="x", padx=10, pady=2)
        
        self.freeze_encoder_var = ctk.BooleanVar(value=True)
        self.freeze_encoder_check = ctk.CTkCheckBox(freeze_frame, text="Freeze Encoder (faster training)", variable=self.freeze_encoder_var, width=20)
        self.freeze_encoder_check.pack(side="left", padx=5)
        
        recon_samples_frame = ctk.CTkFrame(self.autoencoder_frame)
        recon_samples_frame.pack(fill="x", padx=10, pady=2)
        
        recon_label = ctk.CTkLabel(recon_samples_frame, text="Reconstruction Samples:")
        recon_label.pack(side="left", padx=5)
        
        self.recon_samples_entry = ctk.CTkEntry(recon_samples_frame, width=100)
        self.recon_samples_entry.pack(side="right", padx=5)
        self.recon_samples_entry.insert(0, "10")
        
        
        self.test_split_frame = ctk.CTkFrame(hyper_frame)
        self.test_split_frame.pack(fill="x", padx=10, pady=2)
        
        test_label = ctk.CTkLabel(self.test_split_frame, text="Test Split (%):")
        test_label.pack(side="left", padx=5)
        
        self.test_split_entry = ctk.CTkEntry(self.test_split_frame, width=100)
        self.test_split_entry.pack(side="right", padx=5)
        self.test_split_entry.insert(0, "20")
        
        self.model_io_frame = ctk.CTkFrame(hyper_frame)
        self.model_io_frame.pack(fill="x", padx=10, pady=5)
        
        io_label = ctk.CTkLabel(self.model_io_frame, text="💾 Model I/O", font=ctk.CTkFont(size=12, weight="bold"))
        io_label.pack(pady=2)
        
        io_btn_frame = ctk.CTkFrame(self.model_io_frame)
        io_btn_frame.pack(fill="x", padx=2, pady=2)
        
        self.save_btn = ctk.CTkButton(io_btn_frame, text="Save Model", width=80, command=self._on_save_clicked)
        self.save_btn.pack(side="left", padx=5, expand=True)
        
        self.load_btn = ctk.CTkButton(io_btn_frame, text="Load Model", width=80, command=self._on_load_clicked)
        self.load_btn.pack(side="right", padx=5, expand=True)
        
        self.test_handwriting_btn = ctk.CTkButton(
            hyper_frame, 
            text="✍️ Test Handwriting", 
            fg_color="#8E44AD", 
            hover_color="#732D91", 
            command=self._on_test_handwriting_clicked
        )
        self.test_handwriting_btn.pack(fill="x", padx=10, pady=5)
        self.test_handwriting_btn.pack_forget() # Hidden by default
        
        # Encoder I/O Frame (for Autoencoder mode)
        self.encoder_io_frame = ctk.CTkFrame(hyper_frame)
        self.encoder_io_frame.pack(fill="x", padx=10, pady=5)
        
        encoder_io_label = ctk.CTkLabel(self.encoder_io_frame, text="💾 Encoder I/O", font=ctk.CTkFont(size=12, weight="bold"))
        encoder_io_label.pack(pady=2)
        
        encoder_io_btn_frame = ctk.CTkFrame(self.encoder_io_frame)
        encoder_io_btn_frame.pack(fill="x", padx=2, pady=2)
        
        self.save_encoder_btn = ctk.CTkButton(encoder_io_btn_frame, text="Save Encoder", width=80, command=self._on_save_encoder_clicked)
        self.save_encoder_btn.pack(side="left", padx=5, expand=True)
        
        self.load_encoder_btn = ctk.CTkButton(encoder_io_btn_frame, text="Load Encoder", width=80, command=self._on_load_encoder_clicked)
        self.load_encoder_btn.pack(side="right", padx=5, expand=True)
        
        self.encoder_io_frame.pack_forget()  # Hidden by default
        
        control_frame = ctk.CTkFrame(hyper_frame)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        self.clear_btn = ctk.CTkButton(control_frame, text="🗑️ Clear Data", command=self._on_clear_data_clicked, fg_color="#E74C3C", hover_color="#C0392B")
        self.clear_btn.pack(pady=5, padx=10, fill="x")
        
        self.train_btn = ctk.CTkButton(control_frame, text="▶️ START TRAINING", command=self._on_start_training_clicked, fg_color="#27AE60", hover_color="#229954", font=ctk.CTkFont(size=14, weight="bold"))
        self.train_btn.pack(pady=5, padx=10, fill="x")
        
        self.stop_btn = ctk.CTkButton(control_frame, text="⏹️ STOP TRAINING", command=self._on_stop_training_clicked, fg_color="#E74C3C", hover_color="#C0392B", font=ctk.CTkFont(size=14, weight="bold"))
        
        self.status_label = ctk.CTkLabel(control_frame, text="Ready", font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=5)
        
        self._on_model_changed(self.model_menu.get())
        self._on_stopping_criteria_changed()
        self._apply_default_hyperparams()
    
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
            
    def _on_stop_training_clicked(self):
        if self.on_stop_training:
            self.on_stop_training()

    def _on_save_clicked(self):
        if self.on_save_model:
            self.on_save_model()
            
    def _on_load_clicked(self):
        if self.on_load_model:
            self.on_load_model()
            
    def _on_test_handwriting_clicked(self):
        if self.on_test_handwriting:
            self.on_test_handwriting()
    
    def _on_save_encoder_clicked(self):
        if hasattr(self, 'on_save_encoder') and self.on_save_encoder:
            self.on_save_encoder()
    
    def _on_load_encoder_clicked(self):
        if hasattr(self, 'on_load_encoder') and self.on_load_encoder:
            self.on_load_encoder()

    def _on_xor_clicked(self):
        if self.on_generate_xor: self.on_generate_xor()
    
    def _on_circles_clicked(self):
        if self.on_generate_circles: self.on_generate_circles()
        
    def _on_moons_clicked(self):
        if self.on_generate_moons: self.on_generate_moons()

    def _on_blobs_clicked(self):
        if self.on_generate_blobs: self.on_generate_blobs()

    def _on_sine_clicked(self):
        if self.on_generate_sine: self.on_generate_sine()

    def _on_parabola_clicked(self):
        if self.on_generate_parabola: self.on_generate_parabola()
        
    def _on_linear_clicked(self):
        if self.on_generate_linear: self.on_generate_linear()
        
    def _on_abs_clicked(self):
        if self.on_generate_abs: self.on_generate_abs()

    def _on_dataset_changed(self, choice):
        self._update_preset_visibility()
        self._apply_default_hyperparams()
        if self.on_dataset_changed_callback:
            self.on_dataset_changed_callback(choice)
    
    def _on_task_changed(self, choice):
        is_regression = (choice == "Regression")
        
        self._update_preset_visibility()
        
        if is_regression:
            self.add_class_btn.pack_forget()
            self.remove_class_btn.pack_forget()
            self.class_label.configure(text="📊 Output")
            
            for i, radio in enumerate(self.class_radio_buttons):
                if i == 0:
                    radio.configure(text="  Continuous Output")
                else:
                    radio.pack_forget()
            
            self.activation_output_var.set("linear")
            
            if hasattr(self, 'architecture_entry'):
                try:
                    arch_str = self.architecture_entry.get()
                    arch = [int(x.strip()) for x in arch_str.split(',')]
                    if len(arch) >= 2 and arch[-1] != 1:
                        arch[-1] = 1
                        new_arch_str = ','.join(map(str, arch))
                        self.architecture_entry.delete(0, 'end')
                        self.architecture_entry.insert(0, new_arch_str)
                except:
                    pass
        
        else:
            self.add_class_btn.pack(side="left", padx=5)
            self.remove_class_btn.pack(side="left", padx=5)
            self.add_class_btn.configure(text="+ Class")
            self.remove_class_btn.configure(text="- Class")
            self.class_label.configure(text="🎨 Class Management")
            
            for i, radio in enumerate(self.class_radio_buttons):
                radio.configure(text=f"  Class {i}")
                radio.pack(anchor="w", padx=10, pady=2)
            
            if len(self.class_radio_buttons) > 0:
                self.selected_class.set(0)
            
            self.activation_output_var.set("softmax")
            
        self._apply_default_hyperparams()
        
        if self.on_task_changed_callback:
            self.on_task_changed_callback(choice)

    def _on_stopping_criteria_changed(self):
        if self.stopping_criteria.get() == "epochs":
            self.epochs_frame.pack(fill="x", padx=10, pady=2)
            self.min_error_frame.pack_forget()
        else:
            self.epochs_frame.pack_forget()
            self.min_error_frame.pack(fill="x", padx=10, pady=2)

    def _on_ae_stopping_changed(self, choice):
        if choice == "epochs":
            self.ae_epochs_frame.pack(fill="x", padx=10, pady=2, after=self.ae_stopping_switch.master)
            self.ae_min_error_frame.pack_forget()
        else:
            self.ae_epochs_frame.pack_forget()
            self.ae_min_error_frame.pack(fill="x", padx=10, pady=2, after=self.ae_stopping_switch.master)

    def _on_model_changed(self, choice):
        if "Autoencoder-Based MLP" in choice:
            self.architecture_frame.pack(fill="x", padx=10, pady=2)
            self.activation_hidden_frame.pack(fill="x", padx=10, pady=2)
            self.activation_output_frame.pack(fill="x", padx=10, pady=2)
            self.batch_frame.pack(fill="x", padx=10, pady=2)
            self.l2_frame.pack(fill="x", padx=10, pady=2)
            self.momentum_frame.pack(fill="x", padx=10, pady=2)
            self.autoencoder_frame.pack(fill="x", padx=10, pady=5)
        elif "Multi-Layer" in choice:
            self.architecture_frame.pack(fill="x", padx=10, pady=2)
            self.activation_hidden_frame.pack(fill="x", padx=10, pady=2)
            self.activation_output_frame.pack(fill="x", padx=10, pady=2)
            self.batch_frame.pack(fill="x", padx=10, pady=2)
            self.l2_frame.pack(fill="x", padx=10, pady=2)
            self.momentum_frame.pack(fill="x", padx=10, pady=2)
            self.autoencoder_frame.pack_forget()
        else:
            self.architecture_frame.pack_forget()
            self.activation_hidden_frame.pack_forget()
            self.activation_output_frame.pack_forget()
            self.batch_frame.pack_forget()
            self.l2_frame.pack_forget()
            self.momentum_frame.pack_forget()
            self.autoencoder_frame.pack_forget()
        
        self._apply_default_hyperparams()
        
        self._update_preset_visibility()

        if hasattr(self, 'on_model_changed_mnist_callback'):
            if self.on_model_changed_mnist_callback:
                self.on_model_changed_mnist_callback()

    def _update_preset_visibility(self):
        model = self.get_model_type()
        dataset = self.get_dataset_mode()
        task = self.get_task_type()
        
        should_show = (model == 'MLP') and (dataset == 'manual')
        
        if dataset == 'mnist':
            if hasattr(self, 'test_split_frame'):
                self.test_split_frame.pack_forget()
            
            # Only show Test Handwriting and Model I/O for standard MLP (not Autoencoder)
            if model == 'MLP':
                if hasattr(self, 'test_handwriting_btn'):
                    self.test_handwriting_btn.pack(fill="x", padx=10, pady=5)
                if hasattr(self, 'model_io_frame'):
                    self.model_io_frame.pack(fill="x", padx=10, pady=5)
                if hasattr(self, 'encoder_io_frame'):
                    self.encoder_io_frame.pack_forget()
            elif model == 'AutoencoderMLP':
                if hasattr(self, 'test_handwriting_btn'):
                    self.test_handwriting_btn.pack_forget()
                if hasattr(self, 'model_io_frame'):
                    self.model_io_frame.pack_forget()
                if hasattr(self, 'encoder_io_frame'):
                    self.encoder_io_frame.pack(fill="x", padx=10, pady=5)
            else:
                if hasattr(self, 'test_handwriting_btn'):
                    self.test_handwriting_btn.pack_forget()
                if hasattr(self, 'model_io_frame'):
                    self.model_io_frame.pack_forget()
                if hasattr(self, 'encoder_io_frame'):
                    self.encoder_io_frame.pack_forget()
        else:
            if hasattr(self, 'test_handwriting_btn'):
                self.test_handwriting_btn.pack_forget()
            if hasattr(self, 'model_io_frame'):
                self.model_io_frame.pack_forget()
            if hasattr(self, 'encoder_io_frame'):
                self.encoder_io_frame.pack_forget()
            if hasattr(self, 'test_split_frame'):
                self.test_split_frame.pack(fill="x", padx=10, pady=2)
        
        if not should_show:
            self.preset_classification_frame.pack_forget()
            self.preset_regression_frame.pack_forget()
        else:
            if task == 'regression':
                self.preset_classification_frame.pack_forget()
                self.preset_regression_frame.pack(fill="x", padx=5, pady=5, after=self.class_radio_frame)
            else:
                self.preset_regression_frame.pack_forget()
                self.preset_classification_frame.pack(fill="x", padx=5, pady=5, after=self.class_radio_frame)

    def _apply_default_hyperparams(self):
        from config.default_hyperparams import get_defaults
        
        task = self.get_task_type()
        model = self.get_model_type()
        dataset_mode = self.get_dataset_mode()
        
        def set_entry(entry, value):
            try:
                entry.delete(0, 'end')
                entry.insert(0, str(value))
            except Exception:
                pass
        
        try:
            defaults = get_defaults(model, task, dataset_mode)
        except Exception as e:
            print(f"Warning: Could not load defaults from config: {e}")
            return
        
        if 'learning_rate' in defaults:
            set_entry(self.learning_rate_entry, defaults['learning_rate'])
        if 'epochs' in defaults:
            set_entry(self.epochs_entry, defaults['epochs'])
        if 'min_error' in defaults:
            set_entry(self.min_error_entry, defaults['min_error'])
        if 'stopping_criteria' in defaults:
            self.stopping_criteria.set(defaults['stopping_criteria'])
            self._on_stopping_criteria_changed()
        
        if model in ['MLP', 'AutoencoderMLP']:
            if 'architecture' in defaults and hasattr(self, 'architecture_entry'):
                set_entry(self.architecture_entry, defaults['architecture'])
            if 'activation_hidden' in defaults:
                self.activation_hidden_var.set(defaults['activation_hidden'])
            if 'activation_output' in defaults:
                self.activation_output_var.set(defaults['activation_output'])
            if 'batch_size' in defaults:
                set_entry(self.batch_size_entry, defaults['batch_size'])
            if 'l2_lambda' in defaults:
                set_entry(self.l2_entry, defaults['l2_lambda'])
            if 'use_momentum' in defaults:
                self.use_momentum_var.set(defaults['use_momentum'])
            if 'momentum_factor' in defaults:
                set_entry(self.momentum_entry, defaults['momentum_factor'])
        
        if model == 'AutoencoderMLP':
            if 'encoder_architecture' in defaults and hasattr(self, 'encoder_architecture_entry'):
                set_entry(self.encoder_architecture_entry, defaults['encoder_architecture'])
            if 'ae_epochs' in defaults and hasattr(self, 'ae_epochs_entry'):
                set_entry(self.ae_epochs_entry, defaults['ae_epochs'])
            
            if 'ae_stopping_criteria' in defaults and hasattr(self, 'ae_stopping_criteria'):
                self.ae_stopping_criteria.set(defaults['ae_stopping_criteria'])
                self._on_ae_stopping_changed(defaults['ae_stopping_criteria'])

            if 'ae_min_error' in defaults and hasattr(self, 'ae_min_error_entry'):
                set_entry(self.ae_min_error_entry, defaults['ae_min_error'])

            if 'freeze_encoder' in defaults and hasattr(self, 'freeze_encoder_var'):
                self.freeze_encoder_var.set(defaults['freeze_encoder'])
            if 'recon_samples' in defaults and hasattr(self, 'recon_samples_entry'):
                set_entry(self.recon_samples_entry, defaults['recon_samples'])  

    def update_class_radios(self, classes, colors):
        for widget in self.class_radio_frame.winfo_children():
            widget.destroy()
        self.class_radio_buttons = []
        
        if len(classes) > 0:
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
            self.selected_class.set(0)
        
        elif self.get_task_type() == 'regression':
            from config import COLOR_PALETTE
            color = COLOR_PALETTE[0] if len(colors) > 0 else "#1F6AA5"
            radio = ctk.CTkRadioButton(
                self.class_radio_frame,
                text="  Continuous Output",
                variable=self.selected_class,
                value=0,
                fg_color=color,
                border_color=color
            )
            radio.pack(anchor="w", padx=10, pady=2)
            self.class_radio_buttons.append(radio)
            self.selected_class.set(0)
    
    def get_selected_class(self):
        return self.selected_class.get()
    
    def get_model_type(self):
        model_str = self.model_menu.get()
        if "Perceptron" in model_str:
            return "Perceptron"
        elif "Delta Rule" in model_str:
            return "DeltaRule"
        elif "Autoencoder" in model_str:
            return "AutoencoderMLP"
        else:
            return "MLP"
    
    
    def get_task_type(self):
        return self.task_type.get().lower()
    
    def get_dataset_mode(self):
        return self.dataset_mode.get().lower()

    def set_status(self, status_text):
        self.status_label.configure(text=status_text)
    
    def enable_training(self, enabled=True):
        if enabled:
            self.train_btn.pack(pady=5, padx=10, fill="x")
            self.stop_btn.pack_forget()
            self.train_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")
        else:
            self.train_btn.pack_forget()
            self.stop_btn.pack(pady=5, padx=10, fill="x")
            self.clear_btn.configure(state="disabled")

    def apply_mnist_mode(self):
        self.class_management_frame.pack_forget()

        self.task_type.set("Classification")
        self.task_switch.configure(state="disabled")
        self._on_task_changed("Classification")

        self.model_menu.configure(values=["Multi-Layer (MLP)", "Autoencoder-Based MLP"])
        self.model_type.set("Multi-Layer (MLP)")
        self._on_model_changed("Multi-Layer (MLP)")

        try:
            from config.default_hyperparams import get_defaults
            defaults = get_defaults('MLP', dataset_mode='mnist')
            
            def set_entry(entry, value):
                try:
                    entry.delete(0, 'end')
                    entry.insert(0, str(value))
                except Exception:
                    pass

            if 'learning_rate' in defaults:
                set_entry(self.learning_rate_entry, defaults['learning_rate'])
            
            if 'test_split' in defaults:
                set_entry(self.test_split_entry, defaults['test_split'])

            self.stopping_criteria.set("epochs")
            self._on_stopping_criteria_changed()
            if 'epochs' in defaults:
                set_entry(self.epochs_entry, defaults['epochs'])
            
            if 'min_error' in defaults:
                set_entry(self.min_error_entry, defaults['min_error'])

            if 'architecture' in defaults:
                set_entry(self.architecture_entry, defaults['architecture'])

            if 'activation_hidden' in defaults:
                self.activation_hidden_var.set(defaults['activation_hidden'])
            if 'activation_output' in defaults:
                self.activation_output_var.set(defaults['activation_output'])

            if 'batch_size' in defaults:
                set_entry(self.batch_size_entry, defaults['batch_size'])

            if 'l2_lambda' in defaults:
                set_entry(self.l2_entry, defaults['l2_lambda'])

            if 'use_momentum' in defaults:
                self.use_momentum_var.set(defaults['use_momentum'])
            if 'momentum_factor' in defaults:
                set_entry(self.momentum_entry, defaults['momentum_factor'])

        except Exception as e:
            print(f"Warning: Could not load defaults for MNIST mode: {e}")

    def apply_manual_mode(self):
        self.class_management_frame.pack(fill="x", padx=10, pady=5, before=self.model_menu.master)
        
        self.add_class_btn.configure(state="normal")
        self.remove_class_btn.configure(state="normal")

        self.task_switch.configure(state="normal")

        self.model_menu.configure(values=["Single-Layer (Perceptron)", "Single-Layer (Delta Rule)", "Multi-Layer (MLP)"])
        
        self._apply_default_hyperparams()
