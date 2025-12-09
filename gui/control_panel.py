"""
Control panel UI - right sidebar with hyperparameters.
"""

import customtkinter as ctk


class ControlPanel(ctk.CTkFrame):
    """Right sidebar with all controls and hyperparameter inputs."""

    def __init__(self, master, on_add_class=None, on_remove_class=None,
                 on_clear_data=None, on_start_training=None, on_task_changed_callback=None,
                 on_dataset_changed_callback=None, **kwargs):
        super().__init__(master, **kwargs)

        self.on_add_class = on_add_class
        self.on_remove_class = on_remove_class
        self.on_clear_data = on_clear_data
        self.on_start_training = on_start_training
        self.on_task_changed_callback = on_task_changed_callback
        self.on_dataset_changed_callback = on_dataset_changed_callback
        
        self.selected_class = ctk.IntVar(value=0)
        self.class_radio_buttons = []
        self.dataset_mode = ctk.StringVar(value="Manual")
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup all UI widgets."""
        # title
        title_label = ctk.CTkLabel(self, text="⚙️ Control Panel", 
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10, padx=10)
        
        # Task Selection
        task_frame = ctk.CTkFrame(self)
        task_frame.pack(fill="x", padx=10, pady=5)
        
        task_label = ctk.CTkLabel(task_frame, text="🎯 Task Type",
                                  font=ctk.CTkFont(size=14, weight="bold"))
        task_label.pack(pady=5)
        
        self.task_type = ctk.StringVar(value="Classification")
        self.task_switch = ctk.CTkSegmentedButton(
            task_frame,
            values=["Classification", "Regression"],
            variable=self.task_type,
            command=self._on_task_changed
        )
        self.task_switch.pack(pady=5, padx=10, fill="x")

        dataset_frame = ctk.CTkFrame(self)
        dataset_frame.pack(fill="x", padx=10, pady=5)

        dataset_label = ctk.CTkLabel(dataset_frame, text="📚 Dataset",
                                     font=ctk.CTkFont(size=14, weight="bold"))
        dataset_label.pack(pady=5)

        self.dataset_switch = ctk.CTkSegmentedButton(
            dataset_frame,
            values=["Manual", "MNIST"],
            variable=self.dataset_mode,
            command=self._on_dataset_changed,
        )
        self.dataset_switch.pack(pady=5, padx=10, fill="x")

        # Class Management
        class_frame = ctk.CTkFrame(self)
        class_frame.pack(fill="x", padx=10, pady=5)
        
        self.class_label = ctk.CTkLabel(class_frame, text="🎨 Class Management",
                                   font=ctk.CTkFont(size=14, weight="bold"))
        self.class_label.pack(pady=5)
        
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
        
        self.model_type = ctk.StringVar(value="Single-Layer (Perceptron)")
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
        self.learning_rate_entry.insert(0, "0.1")
        
        # Stopping Criteria
        stopping_frame = ctk.CTkFrame(hyper_frame)
        stopping_frame.pack(fill="x", padx=10, pady=5)
        
        stopping_label = ctk.CTkLabel(stopping_frame, text="🛑 Stopping Criteria",
                                     font=ctk.CTkFont(size=12, weight="bold"))
        stopping_label.pack(pady=2)
        
        self.stopping_criteria = ctk.StringVar(value="error")
        stopping_radio_frame = ctk.CTkFrame(stopping_frame)
        stopping_radio_frame.pack(pady=2)
        
        epochs_radio = ctk.CTkRadioButton(
            stopping_radio_frame,
            text="Max Epochs",
            variable=self.stopping_criteria,
            value="epochs",
            command=self._on_stopping_criteria_changed
        )
        epochs_radio.pack(side="left", padx=5)
        
        error_radio = ctk.CTkRadioButton(
            stopping_radio_frame,
            text="Min Error",
            variable=self.stopping_criteria,
            value="error",
            command=self._on_stopping_criteria_changed
        )
        error_radio.pack(side="left", padx=5)
        
        # Epochs
        self.epochs_frame = ctk.CTkFrame(hyper_frame)
        self.epochs_frame.pack(fill="x", padx=10, pady=2)
        
        epochs_label = ctk.CTkLabel(self.epochs_frame, text="Max Epochs:")
        epochs_label.pack(side="left", padx=5)
        
        self.epochs_entry = ctk.CTkEntry(self.epochs_frame, width=100)
        self.epochs_entry.pack(side="right", padx=5)
        self.epochs_entry.insert(0, "100")
        
        # Min Error
        self.min_error_frame = ctk.CTkFrame(hyper_frame)
        self.min_error_frame.pack(fill="x", padx=10, pady=2)
        
        min_error_label = ctk.CTkLabel(self.min_error_frame, text="Min Error:")
        min_error_label.pack(side="left", padx=5)
        
        self.min_error_entry = ctk.CTkEntry(self.min_error_frame, width=100)
        self.min_error_entry.pack(side="right", padx=5)
        self.min_error_entry.insert(0, "0.001")
        
        # Initially hide min_error_frame (visibility updated below)
        self.min_error_frame.pack_forget()
        
        # Batch Size (MLP only)
        self.batch_frame = ctk.CTkFrame(hyper_frame)
        self.batch_frame.pack(fill="x", padx=10, pady=2)
        
        batch_label = ctk.CTkLabel(self.batch_frame, text="Batch Size:")
        batch_label.pack(side="left", padx=5)
        
        self.batch_size_entry = ctk.CTkEntry(self.batch_frame, width=100)
        self.batch_size_entry.pack(side="right", padx=5)
        self.batch_size_entry.insert(0, "32")
        
        # L2 Regularization (MLP only)
        self.l2_frame = ctk.CTkFrame(hyper_frame)
        self.l2_frame.pack(fill="x", padx=10, pady=2)
        
        l2_label = ctk.CTkLabel(self.l2_frame, text="L2 Regularization:")
        l2_label.pack(side="left", padx=5)
        
        self.l2_entry = ctk.CTkEntry(self.l2_frame, width=100)
        self.l2_entry.pack(side="right", padx=5)
        self.l2_entry.insert(0, "0.0")
        
        # Momentum (MLP only)
        self.momentum_frame = ctk.CTkFrame(hyper_frame)
        self.momentum_frame.pack(fill="x", padx=10, pady=2)
        
        self.use_momentum_var = ctk.BooleanVar(value=False)
        self.momentum_check = ctk.CTkCheckBox(
            self.momentum_frame, 
            text="Use Momentum",
            variable=self.use_momentum_var,
            width=20
        )
        self.momentum_check.pack(side="left", padx=5)
        
        self.momentum_entry = ctk.CTkEntry(self.momentum_frame, width=80)
        self.momentum_entry.pack(side="right", padx=5)
        self.momentum_entry.insert(0, "0.9")
        
        mom_label = ctk.CTkLabel(self.momentum_frame, text="Factor:")
        mom_label.pack(side="right", padx=2)

        
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
        
        # Set initial visibility based on default model type
        self._on_model_changed(self.model_menu.get())
        # Set initial visibility for stopping criteria
        self._on_stopping_criteria_changed()
        # Apply initial hyperparameter presets for default task/model
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

    def _on_dataset_changed(self, choice):
        if self.on_dataset_changed_callback:
            self.on_dataset_changed_callback(choice)
    
    def _on_task_changed(self, choice):
        """Handle task switching (Classification vs Regression)."""
        is_regression = (choice == "Regression")
        
        # Regression: Disable class management, force single output
        if is_regression:
            # Hide/disable class buttons for single-layer regression
            self.add_class_btn.pack_forget()
            self.remove_class_btn.pack_forget()
            self.class_label.configure(text="📊 Output")
            
            # DON'T delete classes - just hide UI
            # This preserves class data when switching back to classification
            
            # Update radio to show "Output" instead of "Class 0"
            for i, radio in enumerate(self.class_radio_buttons):
                if i == 0:
                    radio.configure(text="  Continuous Output")
                else:
                    radio.pack_forget()  # Hide extra classes
            
            # Auto-select linear activation for regression
            self.activation_output_var.set("linear")
            
            # Fix layer architecture: Last value should be 1 for regression
            if hasattr(self, 'architecture_entry'):
                try:
                    arch_str = self.architecture_entry.get()
                    arch = [int(x.strip()) for x in arch_str.split(',')]
                    if len(arch) >= 2 and arch[-1] != 1:
                        # Change last value to 1
                        arch[-1] = 1
                        new_arch_str = ','.join(map(str, arch))
                        self.architecture_entry.delete(0, 'end')
                        self.architecture_entry.insert(0, new_arch_str)
                except:
                    pass  # Invalid format, will be handled by get_architecture
        
        # Classification: Show class management
        else:
            self.add_class_btn.pack(side="left", padx=5)
            self.remove_class_btn.pack(side="left", padx=5)
            self.add_class_btn.configure(text="+ Class")
            self.remove_class_btn.configure(text="- Class")
            self.class_label.configure(text="🎨 Class Management")
            
            # Update radio texts to "Class X" and restore visibility
            for i, radio in enumerate(self.class_radio_buttons):
                radio.configure(text=f"  Class {i}")
                # Restore visibility for hidden radios
                radio.pack(anchor="w", padx=10, pady=2)
            
            # Reset selected class to first class (important after regression mode)
            if len(self.class_radio_buttons) > 0:
                self.selected_class.set(0)
            
            # Auto-select softmax for classification
            self.activation_output_var.set("softmax")
            
        # Re-apply hyperparameter presets when task changes
        self._apply_default_hyperparams()
        
        # Notify callback about task change
        if self.on_task_changed_callback:
            self.on_task_changed_callback(choice)

    def _on_stopping_criteria_changed(self):
        """Show/hide stopping criteria inputs based on selection."""
        if self.stopping_criteria.get() == "epochs":
            self.epochs_frame.pack(fill="x", padx=10, pady=2)
            self.min_error_frame.pack_forget()
        else:
            self.epochs_frame.pack_forget()
            self.min_error_frame.pack(fill="x", padx=10, pady=2)

    def _on_model_changed(self, choice):
        """Show/hide model-specific parameters based on selection."""
        if "Multi-Layer" in choice:
            # MLP: Show all parameters
            self.architecture_frame.pack(fill="x", padx=10, pady=2)
            self.activation_hidden_frame.pack(fill="x", padx=10, pady=2)
            self.activation_output_frame.pack(fill="x", padx=10, pady=2)
            self.batch_frame.pack(fill="x", padx=10, pady=2)
            self.l2_frame.pack(fill="x", padx=10, pady=2)
            self.momentum_frame.pack(fill="x", padx=10, pady=2)
        else:
            # Single-Layer (Perceptron/Delta Rule): Hide MLP-only parameters
            self.architecture_frame.pack_forget()
            self.activation_hidden_frame.pack_forget()
            self.activation_output_frame.pack_forget()
            self.batch_frame.pack_forget()  # Hide batch size for single-layer
            self.l2_frame.pack_forget()  # Hide L2 regularization for single-layer
            self.momentum_frame.pack_forget()
        
        # Apply presets whenever model changes
        self._apply_default_hyperparams()

    def _apply_default_hyperparams(self):
        """Apply model- and task-specific default hyperparameters."""
        task = self.get_task_type()  # 'classification' or 'regression'
        model = self.get_model_type()  # 'Perceptron', 'DeltaRule', 'MLP'

        # Helper to safely set an entry widget
        def set_entry(entry, value):
            try:
                entry.delete(0, 'end')
                entry.insert(0, str(value))
            except Exception:
                pass

        # Perceptron presets
        if model == 'Perceptron':
            if task == 'classification':
                set_entry(self.learning_rate_entry, 0.01)
                set_entry(self.epochs_entry, 100)
                set_entry(self.min_error_entry, 0.01) 
            else:  # regression
                set_entry(self.learning_rate_entry, 0.01)  
                set_entry(self.epochs_entry, 200)
                set_entry(self.min_error_entry, 0.01) 

        # Delta Rule presets
        elif model == 'DeltaRule':
            if task == 'classification':
                set_entry(self.learning_rate_entry, 0.01) 
                set_entry(self.epochs_entry, 100)  
                set_entry(self.min_error_entry, 0.01) 
            else:  # regression
                set_entry(self.learning_rate_entry, 0.01)  
                set_entry(self.epochs_entry, 200) 
                set_entry(self.min_error_entry, 0.01) 

        # MLP presets
        else:  # 'MLP'
            if task == 'classification':
                # Basic, smooth decision boundary
                if hasattr(self, 'architecture_entry'):
                    set_entry(self.architecture_entry, '2,10,2')  
                self.activation_hidden_var.set('tanh')  
                self.activation_output_var.set('softmax')
                set_entry(self.learning_rate_entry, 0.01)  
                set_entry(self.batch_size_entry, 16) 
                set_entry(self.l2_entry, 0.001)
                self.use_momentum_var.set(True)
                set_entry(self.momentum_entry, 0.9)
                set_entry(self.epochs_entry, 500)  
                set_entry(self.min_error_entry, 0.002)  
            else:  # regression
                if hasattr(self, 'architecture_entry'):
                    set_entry(self.architecture_entry, '1,10,1')  
                self.activation_hidden_var.set('tanh')  
                self.activation_output_var.set('linear')
                set_entry(self.learning_rate_entry, 0.01) 
                set_entry(self.batch_size_entry, 16)  
                set_entry(self.l2_entry, 0.001)
                self.use_momentum_var.set(True)
                set_entry(self.momentum_entry, 0.9)
                set_entry(self.epochs_entry, 500)  
                set_entry(self.min_error_entry, 0.002)  

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
        """Get layer architecture with automatic regression validation."""
        try:
            arch_str = self.architecture_entry.get()
            arch = [int(x.strip()) for x in arch_str.split(',')]
            
            # Validate for regression: Last value must be 1
            if self.get_task_type() == 'regression' and len(arch) >= 2:
                if arch[-1] != 1:
                    # Auto-correct and update entry
                    arch[-1] = 1
                    new_arch_str = ','.join(map(str, arch))
                    self.architecture_entry.delete(0, 'end')
                    self.architecture_entry.insert(0, new_arch_str)
            
            return arch
        except:
            # Default fallback
            return [2, 5, 3] if self.get_task_type() == 'classification' else [2, 5, 1]
    
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
    
    def get_stopping_criteria(self):
        """Returns ('epochs' or 'error', max_epochs, min_error)"""
        criteria = self.stopping_criteria.get()
        try:
            max_epochs = int(self.epochs_entry.get()) if criteria == "epochs" else 10000  # Large default for error-based
        except:
            max_epochs = 10000
        
        try:
            min_error = float(self.min_error_entry.get()) if criteria == "error" else 0.0
        except:
            min_error = 0.001
        
        return criteria, max_epochs, min_error
    
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
            
    def get_task_type(self):
        return self.task_type.get().lower()
    
    def get_dataset_mode(self):
        return self.dataset_mode.get().lower()
        
    def get_momentum_config(self):
        use_momentum = self.use_momentum_var.get()
        try:
            factor = float(self.momentum_entry.get())
        except:
            factor = 0.9
        return use_momentum, factor
    
    def set_status(self, status_text):
        self.status_label.configure(text=status_text)
    
    def enable_training(self, enabled=True):
        if enabled:
            self.train_btn.configure(state="normal")
        else:
            self.train_btn.configure(state="disabled")

    def apply_mnist_mode(self):
        """Configure UI for MNIST dataset mode (classification + MLP only)."""
        # Disable manual class management (fixed MNIST labels)
        self.add_class_btn.configure(state="disabled")
        self.remove_class_btn.configure(state="disabled")

        # Force classification task and disable task switching
        self.task_type.set("Classification")
        self.task_switch.configure(state="disabled")
        # Reset task-dependent UI
        self._on_task_changed("Classification")

        # Restrict model selection to MLP only
        self.model_menu.configure(
            values=["Multi-Layer (MLP)"]
        )
        self.model_type.set("Multi-Layer (MLP)")
        # Apply model-specific UI changes
        self._on_model_changed("Multi-Layer (MLP)")

        # Apply MNIST-specific hyperparameter presets
        try:
            # Learning rate - INCREASED for faster MNIST training
            self.learning_rate_entry.delete(0, 'end')
            self.learning_rate_entry.insert(0, "0.1")

            # Test split
            self.test_split_entry.delete(0, 'end')
            self.test_split_entry.insert(0, "20")

            # Stopping criteria: Epochs (more predictable for MNIST)
            self.stopping_criteria.set("epochs")
            self._on_stopping_criteria_changed()
            self.epochs_entry.delete(0, 'end')
            self.epochs_entry.insert(0, "100")
            
            # Also set min error for fallback
            self.min_error_entry.delete(0, 'end')
            self.min_error_entry.insert(0, "0.1") 

            # Architecture for MNIST - IMPROVED with more capacity
            self.architecture_entry.delete(0, 'end')
            self.architecture_entry.insert(0, "784,128,64,10")

            # Activations - CHANGED to ReLU for better performance
            self.activation_hidden_var.set("relu")
            self.activation_output_var.set("softmax")

            # Batch size
            self.batch_size_entry.delete(0, 'end')
            self.batch_size_entry.insert(0, "32")

            # L2 regularization - REDUCED for more flexibility
            self.l2_entry.delete(0, 'end')
            self.l2_entry.insert(0, "0.0001")

            # Momentum
            self.use_momentum_var.set(True)
            self.momentum_entry.delete(0, 'end')
            self.momentum_entry.insert(0, "0.9")
        except Exception:
            pass

    def apply_manual_mode(self):
        """Restore UI settings for manual dataset mode."""
        # Re-enable class management buttons
        self.add_class_btn.configure(state="normal")
        self.remove_class_btn.configure(state="normal")

        # Re-enable task switching
        self.task_switch.configure(state="normal")

        # Restore full model list
        self.model_menu.configure(
            values=[
                "Single-Layer (Perceptron)",
                "Single-Layer (Delta Rule)",
                "Multi-Layer (MLP)",
            ]
        )
        
        # Reset hyperparameters to manual mode defaults
        # This ensures MNIST values don't persist when switching back
        self._apply_default_hyperparams()
