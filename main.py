"""
Neural Network Visualizer - Main Application
Interactive GUI for training and visualizing neural networks from scratch.

Author: Ensar Yesir
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
            on_start_training=self._on_start_training
        )
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    
    # Event Handlers
    
    def _on_point_added(self, x, y):
        """Add data point on mouse click."""
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
        
        # get hyperparameters
        model_type = self.control_panel.get_model_type()
        learning_rate = self.control_panel.get_learning_rate()
        epochs = self.control_panel.get_epochs()
        batch_size = self.control_panel.get_batch_size()
        test_split = self.control_panel.get_test_split()
        
        # train/test split
        X_train, X_test, y_train, y_test = self.data_handler.get_train_test_split(test_ratio=test_split)
        self.X_test = X_test
        self.y_test = y_test
        
        # create model
        n_classes = self.data_handler.get_num_classes()
        
        if model_type == "Perceptron":
            model = Perceptron(learning_rate=learning_rate, n_classes=n_classes)
            batch_size = 1
        elif model_type == "DeltaRule":
            model = DeltaRule(learning_rate=learning_rate, n_classes=n_classes)
        else:  # MLP
            architecture = self.control_panel.get_architecture()
            architecture[0] = 2
            architecture[-1] = n_classes
            activation_funcs_raw = self.control_panel.get_activation_functions()
            
            # Expand activation functions for all layers
            # activation_funcs_raw = [hidden_activation, output_activation]
            # We need one activation per layer: L layers = len(architecture) - 1
            hidden_activation = activation_funcs_raw[0]  # e.g., "relu"
            output_activation = activation_funcs_raw[1]  # e.g., "softmax"
            
            # Build activation list: all hidden layers use hidden_activation, last uses output_activation
            num_layers = len(architecture) - 1
            activation_funcs = [hidden_activation] * (num_layers - 1) + [output_activation]
            # Example: for [2,5,3] → ["relu", "softmax"]
            # Example: for [2,10,10,3] → ["relu", "relu", "softmax"]
            
            l2_lambda = self.control_panel.get_l2_lambda()
            model = MLP(
                layer_dims=architecture,
                activation_funcs=activation_funcs,
                learning_rate=learning_rate,
                l2_lambda=l2_lambda
            )
        
        # start training async
        self.after(100, lambda: self._run_training(model, X_train, y_train, epochs, batch_size))
    
    def _run_training(self, model, X_train, y_train, epochs, batch_size):
        """Run training loop with real-time visualization updates."""
        self.current_model = model
        
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
            
            # update UI
            self.control_panel.set_status(f"Epoch {epoch} - Error: {loss:.4f}")
            self.visualization_frame.update_loss_plot(epoch, loss)
            
            # update decision boundary every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                self.visualization_frame.update_decision_boundary(
                    model, X_train, y_train, self.data_handler, tab_name='train'
                )
            
            self.update_idletasks()
            self.after(50, lambda: self._train_next_epoch(fit_generator, X_train, y_train))
            
        except StopIteration:
            # training complete
            self.visualization_frame.update_decision_boundary(
                self.current_model, X_train, y_train, self.data_handler, tab_name='train'
            )
            self._on_training_completed(self.current_model)
    
    def _on_training_completed(self, model):
        """Handle training completion and evaluate on test set."""
        self.is_training = False
        self.trained_model = model
        self.control_panel.enable_training(True)
        
        # evaluate on test set
        if len(self.X_test) > 0:
            self.visualization_frame.update_decision_boundary(
                model, self.X_test, self.y_test, self.data_handler, tab_name='test'
            )
            
            y_pred = model.predict(self.X_test)
            
            # Calculate accuracy manually
            correct = 0
            for i in range(len(y_pred)):
                if y_pred[i] == self.y_test[i]:
                    correct += 1
            accuracy = (correct / len(y_pred)) * 100
            
            self.control_panel.set_status(f"Training complete! Test Accuracy: {accuracy:.2f}%")
            self.visualization_frame.switch_to_tab('test')
            messagebox.showinfo("Success", f"Training completed successfully!\nTest Accuracy: {accuracy:.2f}%")
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
