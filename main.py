"""
Neural Network Visualizer - Main Application
Interactive GUI for training and visualizing neural networks from scratch.
"""

import customtkinter as ctk
from tkinter import messagebox

from utils.data_handler import DataHandler
from algorithms.single_layer import Perceptron, DeltaRule
from algorithms.mlp import MLP
from algorithms.autoencoder import Autoencoder
from algorithms.mlp_with_encoder import MLPWithEncoder
from utils.load_mnist import load_mnist_dataset
from gui.control_panel import ControlPanel
from gui.visualization_frames import VisualizationFrame


class NeuralNetworkVisualizer(ctk.CTk):
    """Main application orchestrating GUI, algorithms, and visualization."""

    def __init__(self):
        super().__init__()

        # window config
        self.title("🧠 Neural Network Visualizer - MLP From Scratch")
        self.geometry("1400x800")

        # theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # data handler
        self.dataset_mode = 'manual'  # 'manual' or 'mnist'
        self.data_handler = DataHandler()
        self.data_handler.add_class("Class 0")
        self.data_handler.add_class("Class 1")
        
        # training state
        self.is_training = False
        self.stop_requested = False  # Flag to interrupt training
        self.trained_model = None
        self.trained_autoencoder = None  # NEW: Store trained autoencoder
        
        # setup UI
        self._setup_ui()
        self._update_class_radios()
    
    def _setup_ui(self):
        """Setup two-panel layout: visualization (left) and controls (right)."""
        # grid layout
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # visualization frame (left)
        self.visualization_frame = VisualizationFrame(
            self,
            on_point_added_callback=self._on_point_added
        )
        self.visualization_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        # control panel (right)
        self.control_panel = ControlPanel(
            self,
            on_add_class=self._on_add_class,
            on_remove_class=self._on_remove_class,
            on_clear_data=self._on_clear_data,
            on_start_training=self._on_start_training,
            on_stop_training=self._on_stop_training,
            on_task_changed_callback=self._on_task_changed,
            on_dataset_changed_callback=self._on_dataset_changed,
            on_generate_xor=self._on_generate_xor,
            on_generate_circles=self._on_generate_circles,
            on_generate_moons=self._on_generate_moons,
            on_generate_blobs=self._on_generate_blobs,
            on_generate_sine=self._on_generate_sine,
            on_generate_parabola=self._on_generate_parabola,
            on_generate_linear=self._on_generate_linear,
            on_generate_abs=self._on_generate_abs,
        )
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        
        # Set callback for model changes in MNIST mode
        self.control_panel.on_model_changed_mnist_callback = self._on_model_changed_mnist
    
    # Event Handlers
    def _on_point_added(self, x, y):
        """Add data point on mouse click."""
        # Prevent adding points during training
        if self.is_training:
            return
        
        class_id = self.control_panel.get_selected_class()
        self.data_handler.add_point(x, y, class_id)
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_add_class(self):
        """Add a new class (max 6)."""
        if self.data_handler.get_num_classes() >= 6:
            messagebox.showwarning("Warning", "Maximum 6 classes allowed.")
            return
        
        class_name = f"Class {self.data_handler.get_num_classes()}"
        self.data_handler.add_class(class_name)
        self._update_class_radios()
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_remove_class(self):
        """Remove last class (min 2)."""
        if self.data_handler.get_num_classes() <= 2:
            messagebox.showwarning("Warning", "Minimum 2 classes required.")
            return
        
        self.data_handler.remove_class()
        self._update_class_radios()
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_clear_data(self):
        """Clear all data points after confirmation."""
        response = messagebox.askyesno("Confirm", "Are you sure you want to clear all data points?")
        if response:
            self.data_handler.clear_data()
            self.visualization_frame.update_train_view(self.data_handler)
            self.visualization_frame.clear_test_view()
            self.visualization_frame.clear_loss_history()
            self.control_panel.set_status("Data cleared")
    
    def _on_task_changed(self, task_choice):
        """Handle task type changes (Classification/Regression)."""
        # Update visualization frame's task mode
        task = 'regression' if task_choice == 'Regression' else 'classification'
        self.visualization_frame.current_task = task
        
        # Refresh visualization to show correct point colors
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_dataset_changed(self, dataset_choice):
        """Handle dataset source switching (Manual vs MNIST)."""
        mode = 'mnist' if dataset_choice == 'MNIST' else 'manual'
        self.dataset_mode = mode

        if mode == 'mnist':
            # Disable manual point adding when using MNIST
            self.visualization_frame.enable_clicking(False)
            self.control_panel.set_status("MNIST mode: using dataset/MNIST, click disabled.")
            # Get current model type
            model_type = self.control_panel.get_model_type()
            # Reconfigure tabs for MNIST (model type determines if reconstruction/latent tabs shown)
            self.visualization_frame.configure_for_dataset_mode('mnist', model_type)
            # Delegate detailed UI configuration to control panel
            self.control_panel.apply_mnist_mode()
        else:
            # Re-enable manual interaction
            self.visualization_frame.enable_clicking(True)
            self.control_panel.set_status("Manual mode: click to add training points.")
            # Reconfigure tabs for Manual (show Training/Test, hide Reconstruction/Latent)
            self.visualization_frame.configure_for_dataset_mode('manual')
            # Delegate manual mode UI restoration
            self.control_panel.apply_manual_mode()
    
    def _on_model_changed_mnist(self):
        """Handle model changes in MNIST mode - update tabs."""
        if self.dataset_mode == 'mnist':
            model_type = self.control_panel.get_model_type()
            # Force tab reconfiguration (even if mode hasn't changed)
            self.visualization_frame.current_dataset_mode = None  # Force refresh
            self.visualization_frame.configure_for_dataset_mode('mnist', model_type)
    
        if self.control_panel.get_task_type() == 'regression':
             # For regression, we might want to plot the points differently or just as single class
             pass
        self.visualization_frame.update_train_view(self.data_handler)

    # Classification Presets
    def _on_generate_xor(self):
        if self.is_training: return
        self.data_handler.generate_xor()
        self._on_preset_generated()

    def _on_generate_circles(self):
        if self.is_training: return
        self.data_handler.generate_circles()
        self._on_preset_generated()
        
    def _on_generate_moons(self):
        if self.is_training: return
        self.data_handler.generate_moons()
        self._on_preset_generated()
        
    def _on_generate_blobs(self):
        if self.is_training: return
        self.data_handler.generate_blobs()
        self._on_preset_generated()

    # Regression Presets
    def _on_generate_sine(self):
        if self.is_training: return
        self.data_handler.generate_sine()
        self._on_preset_generated()

    def _on_generate_parabola(self):
        if self.is_training: return
        self.data_handler.generate_parabola()
        self._on_preset_generated()
        
    def _on_generate_linear(self):
        if self.is_training: return
        self.data_handler.generate_linear()
        self._on_preset_generated()
        
    def _on_generate_abs(self):
        if self.is_training: return
        self.data_handler.generate_abs()
        self._on_preset_generated()
        self._on_preset_generated()

    def _on_preset_generated(self):
        """Common update logic after preset generation."""
        self.control_panel.update_class_radios(
            self.data_handler.classes, 
            self.data_handler.colors
        )
        self.visualization_frame.update_train_view(self.data_handler)

    def _on_start_training(self):
        """Initialize and start training process."""
        if self.is_training:
            messagebox.showinfo("Info", "Training already in progress.")
            return
        
        # For manual mode, require some points; MNIST mode provides its own data
        if getattr(self, 'dataset_mode', 'manual') == 'manual':
            if self.data_handler.get_num_points() < 10:
                messagebox.showwarning("Warning", "Please add at least 10 data points.")
                return
        
        # prepare training
        self.is_training = True
        self.stop_requested = False
        self.control_panel.enable_training(False)
        self.control_panel.set_status("Initializing training...")
        self.visualization_frame.clear_loss_history()
        # Disable clicking during training
        self.visualization_frame.enable_clicking(False)
        
        # For MNIST, switch to error graph tab (training/test tabs are empty for high-D data)
        if getattr(self, 'dataset_mode', 'manual') == 'mnist':
            self.visualization_frame.switch_to_tab('loss')
        
        # 1) Collect hyperparameters from control panel
        (
            model_type,
            task,
            learning_rate,
            stopping_criteria,
            epochs,
            min_error,
            batch_size,
            test_split,
        ) = self._collect_hyperparameters()

        # Store stopping criteria for use in training loop
        self.stopping_criteria = stopping_criteria
        self.min_error = min_error

        # 2) Prepare dataset (manual 2D points vs MNIST images)
        X_train, X_test, y_train, y_test, task = self._prepare_training_data(
            task, test_split
        )
        self.X_test = X_test
        self.y_test = y_test
        self.current_task = task
        # Update visualization task mode (for correct point rendering)
        self.visualization_frame.current_task = task

        # 3) Determine number of classes / outputs
        if getattr(self, 'dataset_mode', 'manual') == 'mnist':
            n_classes = 10
            task = 'classification'
        else:
            n_classes = self.data_handler.get_num_classes()
            task = self.control_panel.get_task_type()
        self.current_task = task

        if task == 'regression':
            n_output_nodes = 1
        else:
            n_output_nodes = n_classes

        # 4) Build model based on selected type
        model, batch_size = self._build_model(
            model_type=model_type,
            learning_rate=learning_rate,
            task=task,
            n_output_nodes=n_output_nodes,
            batch_size=batch_size,
        )

        # 5) Start training asynchronously
        self.after(
            100,
            lambda: self._run_training(
                model, X_train, y_train, epochs, batch_size, stopping_criteria, min_error, model_type
            ),
        )

    def _on_stop_training(self):
        """Set flag to stop training."""
        if self.is_training:
            self.stop_requested = True
            self.control_panel.set_status("Stop requested. Finishing current epoch...")
        else:
            messagebox.showinfo("Info", "No training in progress to stop.")

    def _finish_training(self):
        """Common cleanup after training stops (either completed or interrupted)."""
        self.is_training = False
        self.stop_requested = False
        self.control_panel.enable_training(True)
        self.visualization_frame.enable_clicking(True)
        self.control_panel.set_status("Training stopped.")
        messagebox.showinfo("Info", "Training stopped.")

    def _collect_hyperparameters(self):
        """Collect core hyperparameters from the control panel."""
        model_type = self.control_panel.get_model_type()
        learning_rate = self.control_panel.inputs.get_learning_rate()
        stopping_criteria, max_epochs, min_error = self.control_panel.inputs.get_stopping_criteria()
        batch_size = self.control_panel.inputs.get_batch_size()
        test_split = self.control_panel.inputs.get_test_split()
        task = self.control_panel.get_task_type()
        epochs = max_epochs  # Use max_epochs for fit() call (limited by stopping criteria)

        return (
            model_type,
            task,
            learning_rate,
            stopping_criteria,
            epochs,
            min_error,
            batch_size,
            test_split,
        )

    def _prepare_training_data(self, task, test_split):
        """Prepare X_train, X_test, y_train, y_test based on dataset mode (no validation set)."""
        if getattr(self, 'dataset_mode', 'manual') == 'mnist':
            # MNIST: classification with 10 classes, 784-dim input
            # Use a small, class-balanced subset for faster GUI training
            # No validation set - only train and test
            (X_train, y_train), (X_test, y_test) = load_mnist_dataset(
                per_class_train=100,  # 100 samples per digit for training
                per_class_test=10     # 10 samples per digit for testing
            )
            task = 'classification'
        else:
            # Manual 2D points via DataHandler
            # Simple train/test split (no validation)
            X_train, X_test, y_train, y_test = self.data_handler.get_train_test_split(
                test_ratio=test_split,
                task=task,
            )

        # No validation set stored
        self.X_val = []
        self.y_val = []
        
        return X_train, X_test, y_train, y_test, task

    def _build_model(self, model_type, learning_rate, task, n_output_nodes, batch_size):
        """Create the selected model (Perceptron, DeltaRule, or MLP)."""
        if model_type == "Perceptron":
            model = Perceptron(
                learning_rate=learning_rate,
                n_classes=n_output_nodes,
                task=task,
            )
            batch_size = 1
        elif model_type == "DeltaRule":
            model = DeltaRule(
                learning_rate=learning_rate,
                n_classes=n_output_nodes,
                task=task,
            )
        elif model_type == "AutoencoderMLP":
            # Autoencoder-based MLP: Two-stage training
            # Model will be created in _run_training_autoencoder
            model = None
        else:  # MLP
            architecture = self.control_panel.inputs.get_architecture()
            # Set input size based on task and dataset
            if task == 'regression':
                architecture[0] = 1  # Single feature (X coordinate)
            else:
                if getattr(self, 'dataset_mode', 'manual') == 'mnist':
                    architecture[0] = 784  # Flattened 28x28 image
                else:
                    architecture[0] = 2  # Two features (X, Y coordinates)

            architecture[-1] = n_output_nodes

            # Update architecture entry to reflect changes (important for consistency)
            arch_str = ','.join(map(str, architecture))
            self.control_panel.architecture_entry.delete(0, 'end')
            self.control_panel.architecture_entry.insert(0, arch_str)

            activation_funcs_raw = self.control_panel.get_activation_functions()

            # Expand activation functions for all layers
            # activation_funcs_raw = [hidden_activation, output_activation]
            # We need one activation per layer: L layers = len(architecture) - 1
            hidden_activation = activation_funcs_raw[0]  # e.g., "relu"
            output_activation = activation_funcs_raw[1]  # e.g., "softmax"

            # Build activation list: all hidden layers use hidden_activation, last uses output_activation
            num_layers = len(architecture) - 1
            activation_funcs = [hidden_activation] * (num_layers - 1) + [output_activation]

            l2_lambda = self.control_panel.inputs.get_l2_lambda()
            use_momentum, momentum_factor = self.control_panel.inputs.get_momentum_config()

            model = MLP(
                layer_dims=architecture,
                activation_funcs=activation_funcs,
                learning_rate=learning_rate,
                l2_lambda=l2_lambda,
                task=task,
                use_momentum=use_momentum,
                momentum_factor=momentum_factor,
            )


        return model, batch_size
    
    def _run_training(self, model, X_train, y_train, epochs, batch_size, stopping_criteria='epochs', min_error=0.001, model_type='MLP'):
        """Run training loop with real-time visualization updates."""
        
        # Special handling for Autoencoder-based MLP
        if model_type == 'AutoencoderMLP':
            self._run_training_autoencoder(X_train, y_train, epochs, batch_size, stopping_criteria, min_error)
            return
        
        self.current_model = model
        self.stopping_criteria = stopping_criteria
        self.min_error = min_error
        
        if isinstance(model, (Perceptron, DeltaRule)):
            self.fit_generator = model.fit(X_train, y_train, epochs=epochs)
        else:
            self.fit_generator = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        self._train_next_epoch(X_train, y_train, model_type)
    
    def _train_next_epoch(self, X_train, y_train, model_type='MLP'):
        """Train one epoch and update visualizations (recursive async)."""
        if self.stop_requested:
            self.control_panel.set_status("Training interrupted by user.")
            self._finish_training()
            return

        try:
            epoch, loss, model = next(self.fit_generator)
            self.current_model = model
            
            self.current_model = model
            
            # Check stopping criteria (only using training loss)
            should_stop = False
            stop_reason = ""
            
            if self.stopping_criteria == 'error':
                if loss <= self.min_error:
                    should_stop = True
                    stop_reason = f"Training Error ({loss:.6f}) <= Min Error ({self.min_error:.6f})"
            
            # update UI
            if not should_stop:
                self.control_panel.set_status(f"Epoch {epoch} - Train Error: {loss:.4f}")
            else:
                self.control_panel.set_status(f"Training stopped! {stop_reason}")

            self.visualization_frame.update_loss_plot(epoch, loss)
            
            # update decision boundary every 5 epochs (manual 2D data only)
            if getattr(self, 'dataset_mode', 'manual') == 'manual':
                if epoch % 5 == 0 or epoch == 1:
                    self.visualization_frame.update_decision_boundary(
                        model, X_train, y_train, self.data_handler, tab_name='train', task=self.current_task
                    )
            
            # Stop if criteria met
            if should_stop:
                if getattr(self, 'dataset_mode', 'manual') == 'manual':
                    self.visualization_frame.update_decision_boundary(
                        self.current_model, X_train, y_train, self.data_handler, tab_name='train', task=self.current_task
                    )
                self._on_training_completed(self.current_model)
                return
            
            self.update_idletasks()
            self.after(50, lambda: self._train_next_epoch(X_train, y_train, model_type))

            
        except StopIteration:
            # training complete (max epochs reached)
            if getattr(self, 'dataset_mode', 'manual') == 'manual':
                self.visualization_frame.update_decision_boundary(
                    self.current_model, X_train, y_train, self.data_handler, tab_name='train', task=self.current_task
                )
            self._on_training_completed(self.current_model)
    
    def _on_training_completed(self, model):
        """Handle training completion and evaluate on test set."""
        self.is_training = False
        self.trained_model = model
        self.control_panel.enable_training(True)
        # Re-enable clicking after training
        self.visualization_frame.enable_clicking(True)
        
        # Check for stop request
        if self.stop_requested:
            self.control_panel.set_status("Training interrupted by user.")
            self._finish_training()
            return

        # Debug: Test model predictions
        if self.current_task == 'regression':
            test_points = [[0.0], [5.0], [10.0]]
            test_pred = model.predict(test_points)
            print(f"\n🔍 DEBUG - Model Predictions:")
            print(f"  Input X values: {[p[0] for p in test_points]}")
            print(f"  Predicted Y values: {test_pred}")
        
        # Evaluate on test set only (no validation)
        if hasattr(self, 'X_test') and len(self.X_test) > 0:
            # Only draw 2D decision boundary for manual dataset
            if getattr(self, 'dataset_mode', 'manual') == 'manual':
                self.visualization_frame.update_decision_boundary(
                    model, self.X_test, self.y_test, self.data_handler, tab_name='test', task=self.current_task
                )
            
            y_pred = model.predict(self.X_test)
            
            # Calculate metric
            if self.current_task == 'regression':
                # Calculate MSE
                mse = 0.0
                for i in range(len(y_pred)):
                    diff = y_pred[i] - self.y_test[i]
                    # handle possible list wrapping
                    if isinstance(diff, list):
                        diff = diff[0]
                    mse += diff * diff
                mse /= len(y_pred)
                
                # Print detailed results to console
                print(f"\n{'='*60}")
                print(f"TRAINING COMPLETE - Final Results")
                print(f"{'='*60}")
                print(f"Test MSE: {mse:.6f}")
                print(f"Test Samples: {len(self.y_test)}")
                print(f"{'='*60}\n")
                
                self.control_panel.set_status(f"Training complete! Test MSE: {mse:.4f}")
                messagebox.showinfo("Success", f"Training completed successfully!\nTest MSE: {mse:.4f}")
            else:
                # Calculate accuracy manually
                correct = 0
                for i in range(len(y_pred)):
                    if y_pred[i] == self.y_test[i]:
                        correct += 1
                accuracy = (correct / len(y_pred)) * 100
                
                # Print detailed results to console
                print(f"\n{'='*60}")
                print(f"TRAINING COMPLETE - Final Results")
                print(f"{'='*60}")
                print(f"Test Accuracy: {accuracy:.2f}%")
                print(f"Test Correct: {correct}/{len(y_pred)}")
                print(f"{'='*60}\n")
                
                self.control_panel.set_status(f"Training complete! Test Accuracy: {accuracy:.2f}%")
                messagebox.showinfo("Success", f"Training completed successfully!\nTest Accuracy: {accuracy:.2f}%")
            
            # For MNIST, stay on error graph (since 2D plots are not meaningful)
            # For manual 2D data, switch to test tab to see decision boundary
            if getattr(self, 'dataset_mode', 'manual') == 'manual':
                self.visualization_frame.switch_to_tab('test')
            else:
                # MNIST: stay on error graph
                self.visualization_frame.switch_to_tab('loss')
        else:
            self.control_panel.set_status("Training completed!")
            messagebox.showinfo("Success", "Training completed successfully!")
    
    def _update_class_radios(self):
        """Update class selection radio buttons."""
        classes = self.data_handler.classes
        colors = [self.data_handler.get_color(i) for i in range(len(classes))]
        self.control_panel.update_class_radios(classes, colors)
    
    def _run_training_autoencoder(self, X_train, y_train, epochs, batch_size, stopping_criteria, min_error):
        """Prepare and start Stage 1 (Autoencoder) training."""
        # Get autoencoder configuration
        encoder_dims = self.control_panel.inputs.get_encoder_architecture()
        ae_stop_mode, ae_epochs, ae_min_error_val = self.control_panel.inputs.get_ae_stopping_config()
        freeze_encoder = self.control_panel.inputs.get_freeze_encoder()
        recon_samples = self.control_panel.inputs.get_recon_samples()
        
        # Get MLP configuration (needed later for Stage 2)
        learning_rate = self.control_panel.inputs.get_learning_rate()
        use_momentum, momentum_factor = self.control_panel.inputs.get_momentum_config()
        
        # Encoder activations (all ReLU for encoder)
        num_encoder_layers = len(encoder_dims) - 1
        encoder_activations = ['relu'] * (num_encoder_layers - 1) + ['relu']
        
        # Decoder activations (last layer sigmoid for images)
        decoder_activations = ['relu'] * (num_encoder_layers - 1) + ['sigmoid']
        ae_activations = encoder_activations + decoder_activations
        
        self.control_panel.set_status("Stage 1/2: Training Autoencoder...")
        
        # Stage 1: Initialize Autoencoder
        autoencoder = Autoencoder(
            encoder_dims=encoder_dims,
            activation_funcs=ae_activations,
            learning_rate=learning_rate,
            use_momentum=use_momentum,
            momentum_factor=momentum_factor,
        )
        
        # Create generator
        self.ae_fit_generator = autoencoder.fit(X_train, epochs=ae_epochs, batch_size=batch_size)
        
        # Store context for Stage 2
        self.stage2_context = {
            'X_train': X_train, 'y_train': y_train,
            'epochs': epochs, 'batch_size': batch_size,
            'stopping_criteria': stopping_criteria, 'min_error': min_error,
            'encoder_dims': encoder_dims, 'freeze_encoder': freeze_encoder,
            'ae_stop_mode': ae_stop_mode, 'ae_min_error_val': ae_min_error_val,
            'recon_samples': recon_samples,
            'learning_rate': learning_rate, 'use_momentum': use_momentum, 'momentum_factor': momentum_factor,
        }
        
        # Start async loop
        self._train_autoencoder_next_epoch()

    def _train_autoencoder_next_epoch(self):
        """Train one epoch of Autoencoder (Stage 1)."""
        if self.stop_requested:
            self.control_panel.set_status("AE training interrupted.")
            self._finish_training()
            return

        try:
            epoch, loss, trained_ae = next(self.ae_fit_generator)
            self.current_model = trained_ae # Keep track of the latest trained AE
            
            # Check stopping criteria (Min Error)
            ctx = self.stage2_context
            if ctx['ae_stop_mode'] == 'error' and loss <= ctx['ae_min_error_val']:
                self.control_panel.set_status(f"AE Converged (Loss {loss:.4f})")
                self.visualization_frame.update_loss_plot(epoch, loss)
                # Proceed to Stage 2 immediately
                self._start_stage2_mlp(trained_ae)
                return

            self.control_panel.set_status(f"Stage 1/2: AE Epoch {epoch} - Recon Loss: {loss:.6f}")
            self.visualization_frame.update_loss_plot(epoch, loss)
            
            # Visualize reconstructions periodically
            if epoch % 5 == 0 or epoch == ctx['epochs']: # Assuming ctx['epochs'] is ae_epochs
                self.visualization_frame.visualize_reconstructions(trained_ae, ctx['X_train'], num_samples=ctx['recon_samples'])
                self.update_idletasks()
            
            # Schedule next epoch
            self.after(20, self._train_autoencoder_next_epoch)
            
        except StopIteration:
            # Stage 1 Complete (max epochs reached) -> Start Stage 2
            self.control_panel.set_status(f"Stage 1/2 Complete! Final Reconstruction Loss: {self.current_model.loss:.6f}. Starting Stage 2 (MLP)...")
            self.visualization_frame.clear_loss_history() # Clear for MLP training
            self.after(500, lambda: self._start_stage2_mlp(self.current_model)) # Pass the last trained AE

    def _start_stage2_mlp(self, trained_ae):
        """Initialize and start Stage 2 (MLP training)."""
        if self.stop_requested: # Double check
            self._finish_training()
            return
            
        self.trained_autoencoder = trained_ae # Save for reuse
        ctx = self.stage2_context
        
        self.control_panel.set_status("Stage 1 Complete. Starting Stage 2: MLP Training...")
        
        # Extract trained encoder weights
        encoder_weights = trained_ae.encoder_weights
        encoder_biases = trained_ae.encoder_biases
        
        # Stage 2: Initialize MLPWithEncoder
        # Current MLP architecture from UI
        mlp_layer_dims = self.control_panel.inputs.get_architecture() 
        
        # Ensure MLP input matches Encoder output
        encoder_output_dim = ctx['encoder_dims'][-1]
        if mlp_layer_dims[0] != encoder_output_dim:
            print(f"Adjusting MLP input dim from {mlp_layer_dims[0]} to {encoder_output_dim} to match encoder.")
            mlp_layer_dims[0] = encoder_output_dim
            
        activation_hidden = self.control_panel.inputs.get_hidden_activation()
        activation_output = self.control_panel.inputs.get_output_activation()
        
        # MLPWithEncoder activations
        mlp_activations = [activation_hidden] * (len(mlp_layer_dims) - 2) + [activation_output]
        
        # Create Loop/Model
        mlp_model = MLPWithEncoder(
            encoder_dims=ctx['encoder_dims'],
            mlp_layer_dims=mlp_layer_dims,
            mlp_activations=mlp_activations,
            learning_rate=ctx['learning_rate'],
            l2_lambda=self.control_panel.inputs.get_l2_lambda(),
            task='classification', # Autoencoder hybrid usually for classification
            use_momentum=ctx['use_momentum'],
            momentum_factor=ctx['momentum_factor'],
            freeze_encoder=ctx['freeze_encoder']
        )
        
        # Transfer trained weights
        mlp_model.set_encoder_weights(encoder_weights, encoder_biases)
        
        # Create fit generator for Stage 2
        self.fit_generator = mlp_model.fit(ctx['X_train'], ctx['y_train'], epochs=ctx['epochs'], batch_size=ctx['batch_size'])
        
        # Start async training loop for Stage 2
        # Note: model_type should be passed to handle logging correctly if needed, or just 'AutoencoderMLP'
        self._train_next_epoch(ctx['X_train'], ctx['y_train'], model_type='AutoencoderMLP')
    
    def _run_stage2_classifier(self, autoencoder, X_train, y_train, epochs, batch_size, 
                                freeze_encoder, stopping_criteria, min_error):
        """
        Stage 2: Train MLP classifier with pre-trained encoder.
        """
        encoder_dims = self.control_panel.inputs.get_encoder_architecture()
        latent_dim = encoder_dims[-1]
        
        # Get MLP architecture from control panel
        mlp_architecture = self.control_panel.inputs.get_architecture()
        mlp_architecture[0] = latent_dim  # Input is latent features
        mlp_architecture[-1] = 10  # Output is 10 classes (MNIST)
        
        # Update UI to show current architecture
        arch_str = ','.join(map(str, mlp_architecture))
        self.control_panel.architecture_entry.delete(0, 'end')
        self.control_panel.architecture_entry.insert(0, arch_str)
        
        # Get activation functions
        activation_funcs_raw = self.control_panel.get_activation_functions()
        hidden_activation = activation_funcs_raw[0]
        output_activation = activation_funcs_raw[1]
        
        # Build MLP activation functions
        num_mlp_layers = len(mlp_architecture) - 1
        mlp_activations = [hidden_activation] * (num_mlp_layers - 1) + [output_activation]
        
        # Encoder activations
        num_encoder_layers = len(encoder_dims) - 1
        encoder_activations = ['relu'] * num_encoder_layers
        
        # Get other hyperparameters
        learning_rate = self.control_panel.inputs.get_learning_rate()
        l2_lambda = self.control_panel.inputs.get_l2_lambda()
        use_momentum, momentum_factor = self.control_panel.inputs.get_momentum_config()
        
        # Create hybrid model
        encoder_params = autoencoder.get_encoder_weights()
        
        hybrid_model = MLPWithEncoder(
            encoder_params=encoder_params,
            encoder_dims=encoder_dims,
            encoder_activations=encoder_activations,
            mlp_layer_dims=mlp_architecture,
            mlp_activations=mlp_activations,
            learning_rate=learning_rate,
            l2_lambda=l2_lambda,
            freeze_encoder=freeze_encoder,
            use_momentum=use_momentum,
            momentum_factor=momentum_factor,
        )
        
        freeze_status = "Frozen" if freeze_encoder else "Trainable"
        self.control_panel.set_status(f"Stage 2/2: Training MLP Classifier (Encoder: {freeze_status})...")
        
        # Store for training loop
        self.current_model = hybrid_model
        self.stopping_criteria = stopping_criteria
        self.min_error = min_error
        
        # Start training classifier
        fit_generator = hybrid_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        self._train_next_epoch(fit_generator, X_train, y_train, 'AutoencoderMLP')


def main():
    """Application entry point."""
    app = NeuralNetworkVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()
