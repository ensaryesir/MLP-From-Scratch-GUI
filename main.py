"""
Neural Network Visualizer - Main Application
Interactive GUI for training and visualizing neural networks from scratch.
"""

import customtkinter as ctk
from tkinter import messagebox
import time

from utils.data_handler import DataHandler
from algorithms.single_layer import Perceptron, DeltaRule
from algorithms.mlp import MLP
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
        self.data_handler = DataHandler()
        self.data_handler.add_class("Class 0")
        self.data_handler.add_class("Class 1")
        
        # training state
        self.is_training = False
        self.trained_model = None
        
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
            on_task_changed_callback=self._on_task_changed
        )
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    
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
    
    def _on_start_training(self):
        """Initialize and start training process."""
        if self.is_training:
            messagebox.showinfo("Info", "Training already in progress.")
            return
        
        if self.data_handler.get_num_points() < 10:
            messagebox.showwarning("Warning", "Please add at least 10 data points.")
            return
        
        # prepare training
        self.is_training = True
        self.control_panel.enable_training(False)
        self.control_panel.set_status("Training starting...")
        self.visualization_frame.clear_loss_history()
        # Disable clicking during training
        self.visualization_frame.enable_clicking(False)
        
        # Update visualization task mode (for correct point rendering)
        task = self.control_panel.get_task_type()
        self.visualization_frame.current_task = task
        
        # get hyperparameters
        model_type = self.control_panel.get_model_type()
        learning_rate = self.control_panel.get_learning_rate()
        stopping_criteria, max_epochs, min_error = self.control_panel.get_stopping_criteria()
        epochs = max_epochs  # Use max_epochs for fit() call (will be limited by stopping criteria)
        batch_size = self.control_panel.get_batch_size()
        test_split = self.control_panel.get_test_split()
        task = self.control_panel.get_task_type()
        
        # Store stopping criteria for use in training loop
        self.stopping_criteria = stopping_criteria
        self.min_error = min_error
        
        # train/test split (pass task for proper data formatting)
        X_train, X_test, y_train, y_test = self.data_handler.get_train_test_split(
            test_ratio=test_split, 
            task=task
        )
        self.X_test = X_test
        self.y_test = y_test
        
        # create model
        n_classes = self.data_handler.get_num_classes()
        task = self.control_panel.get_task_type()
        self.current_task = task
        
        # Adjust n_classes for regression (1 output node)
        if task == 'regression':
            n_output_nodes = 1
        else:
            n_output_nodes = n_classes
        
        if model_type == "Perceptron":
            model = Perceptron(learning_rate=learning_rate, n_classes=n_output_nodes, task=task)
            batch_size = 1
        elif model_type == "DeltaRule":
            model = DeltaRule(learning_rate=learning_rate, n_classes=n_output_nodes, task=task)
        else:  # MLP
            architecture = self.control_panel.get_architecture()
            # Set input size based on task
            if task == 'regression':
                architecture[0] = 1  # Single feature (X coordinate)
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
            
            l2_lambda = self.control_panel.get_l2_lambda()
            use_momentum, momentum_factor = self.control_panel.get_momentum_config()
            
            model = MLP(
                layer_dims=architecture,
                activation_funcs=activation_funcs,
                learning_rate=learning_rate,
                l2_lambda=l2_lambda,
                task=task,
                use_momentum=use_momentum,
                momentum_factor=momentum_factor
            )

        
        # start training async
        self.after(100, lambda: self._run_training(model, X_train, y_train, epochs, batch_size, stopping_criteria, min_error))
    
    def _run_training(self, model, X_train, y_train, epochs, batch_size, stopping_criteria='epochs', min_error=0.001):
        """Run training loop with real-time visualization updates."""
        self.current_model = model
        self.stopping_criteria = stopping_criteria
        self.min_error = min_error
        
        if isinstance(model, (Perceptron, DeltaRule)):
            fit_generator = model.fit(X_train, y_train, epochs=epochs)
        else:
            fit_generator = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        self._train_next_epoch(fit_generator, X_train, y_train)
    
    def _train_next_epoch(self, fit_generator, X_train, y_train):
        """Train one epoch and update visualizations (recursive async)."""
        try:
            epoch, loss, model = next(fit_generator)
            self.current_model = model
            
            # Check stopping criteria
            should_stop = False
            if self.stopping_criteria == 'error':
                # Stop if loss is below min_error
                if loss <= self.min_error:
                    should_stop = True
                    self.control_panel.set_status(f"Training stopped! Error ({loss:.6f}) <= Min Error ({self.min_error:.6f})")
            
            # update UI
            if not should_stop:
                self.control_panel.set_status(f"Epoch {epoch} - Error: {loss:.4f}")
            self.visualization_frame.update_loss_plot(epoch, loss)
            
            # update decision boundary every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                self.visualization_frame.update_decision_boundary(
                    model, X_train, y_train, self.data_handler, tab_name='train', task=self.current_task
                )
            
            # Stop if criteria met
            if should_stop:
                self.visualization_frame.update_decision_boundary(
                    self.current_model, X_train, y_train, self.data_handler, tab_name='train', task=self.current_task
                )
                self._on_training_completed(self.current_model)
                return
            
            self.update_idletasks()
            self.after(50, lambda: self._train_next_epoch(fit_generator, X_train, y_train))
            
        except StopIteration:
            # training complete (max epochs reached)
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
        
        # Debug: Test model predictions
        if self.current_task == 'regression':
            test_points = [[0.0], [5.0], [10.0]]
            test_pred = model.predict(test_points)
            print(f"\n🔍 DEBUG - Model Predictions:")
            print(f"  Input X values: {[p[0] for p in test_points]}")
            print(f"  Predicted Y values: {test_pred}")
        
        # evaluate on test set
        if len(self.X_test) > 0:
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
                
                self.control_panel.set_status(f"Training complete! Test MSE: {mse:.4f}")
                messagebox.showinfo("Success", f"Training completed successfully!\nTest MSE: {mse:.4f}")
            else:
                # Calculate accuracy manually
                correct = 0
                for i in range(len(y_pred)):
                    if y_pred[i] == self.y_test[i]:
                        correct += 1
                accuracy = (correct / len(y_pred)) * 100
                
                self.control_panel.set_status(f"Training complete! Test Accuracy: {accuracy:.2f}%")
                messagebox.showinfo("Success", f"Training completed successfully!\nTest Accuracy: {accuracy:.2f}%")
            
            self.visualization_frame.switch_to_tab('test')
        else:
            self.control_panel.set_status("Training completed!")
            messagebox.showinfo("Success", "Training completed successfully!")
    
    def _update_class_radios(self):
        """Update class selection radio buttons."""
        classes = self.data_handler.classes
        colors = [self.data_handler.get_color(i) for i in range(len(classes))]
        self.control_panel.update_class_radios(classes, colors)


def main():
    """Application entry point."""
    app = NeuralNetworkVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()
