import customtkinter as ctk
from tkinter import messagebox, filedialog
import queue
import numpy as np
import pickle
import sys
import os

from utils.data_handler import DataHandler
from utils.load_mnist import load_mnist_dataset
from gui.control_panel import ControlPanel
from gui.visualization_frames import VisualizationFrame
from gui.training_manager import TrainingManager
from gui.handwriting_tester import HandwritingTester


def get_app_dir():
    """Get application directory - works for both script and exe"""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))


# Create weights directory if it doesn't exist
WEIGHTS_DIR = os.path.join(get_app_dir(), 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)


class NeuralNetworkVisualizer(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🧠 Neural Network Visualizer - MLP From Scratch")
        self.geometry("1400x800")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.dataset_mode = 'manual'
        self.data_handler = DataHandler()
        self.data_handler.add_class("Class 0")
        self.data_handler.add_class("Class 1")
        
        self.is_training = False
        self.stop_requested = False
        self.trained_model = None
        self.trained_autoencoder = None
        self.handwriting_window = None
        
        self.training_manager = TrainingManager(
            update_callback=None,
            completion_callback=None,
            error_callback=None
        )
        
        self._setup_ui()
        self._update_class_radios()
    
    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.visualization_frame = VisualizationFrame(
            self,
            on_point_added_callback=self._on_point_added
        )
        self.visualization_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        self.control_panel = ControlPanel(
            self,
            on_add_class=self._on_add_class,
            on_remove_class=self._on_remove_class,
            on_clear_data=self._on_clear_data,
            on_start_training=self._on_start_training,
            on_stop_training=self._on_stop_training,
            on_save_model=self._on_save_model,
            on_load_model=self._on_load_model,
            on_test_handwriting=self._on_test_handwriting,
            on_save_encoder=self._on_save_encoder,
            on_load_encoder=self._on_load_encoder,
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
        
        self.control_panel.on_model_changed_mnist_callback = self._on_model_changed_mnist
    
    def _on_point_added(self, x, y):
        if self.is_training:
            return
        
        class_id = self.control_panel.get_selected_class()
        self.data_handler.add_point(x, y, class_id)
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_add_class(self):
        if self.data_handler.get_num_classes() >= 6:
            messagebox.showwarning("Warning", "Maximum 6 classes allowed.")
            return
        
        class_name = f"Class {self.data_handler.get_num_classes()}"
        self.data_handler.add_class(class_name)
        self._update_class_radios()
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_remove_class(self):
        if self.data_handler.get_num_classes() <= 2:
            messagebox.showwarning("Warning", "Minimum 2 classes required.")
            return
        
        self.data_handler.remove_class()
        self._update_class_radios()
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_clear_data(self):
        response = messagebox.askyesno("Confirm", "Are you sure you want to clear all data points?")
        if response:
            self.data_handler.clear_data()
            self.visualization_frame.update_train_view(self.data_handler)
            self.visualization_frame.clear_test_view()
            self.visualization_frame.clear_loss_history()
            self.control_panel.set_status("Data cleared")
    
    def _on_task_changed(self, task_choice):
        task = 'regression' if task_choice == 'Regression' else 'classification'
        self.visualization_frame.current_task = task
        self.visualization_frame.update_train_view(self.data_handler)
    
    def _on_dataset_changed(self, dataset_choice):
        mode = 'mnist' if dataset_choice == 'MNIST' else 'manual'
        self.dataset_mode = mode

        if mode == 'mnist':
            self.visualization_frame.enable_clicking(False)
            self.control_panel.set_status("MNIST mode: using dataset/MNIST, click disabled.")
            model_type = self.control_panel.get_model_type()
            self.visualization_frame.configure_for_dataset_mode('mnist', model_type)
            self.control_panel.apply_mnist_mode()
        else:
            self.visualization_frame.enable_clicking(True)
            self.control_panel.set_status("Manual mode: click to add training points.")
            self.visualization_frame.configure_for_dataset_mode('manual')
            self.control_panel.apply_manual_mode()
    
    def _on_model_changed_mnist(self):
        if self.dataset_mode == 'mnist':
            model_type = self.control_panel.get_model_type()
            self.visualization_frame.configure_for_dataset_mode('mnist', model_type)
        self.visualization_frame.update_train_view(self.data_handler)

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

    def _on_preset_generated(self):
        self.control_panel.update_class_radios(
            self.data_handler.classes, 
            self.data_handler.colors
        )
        self.visualization_frame.update_train_view(self.data_handler)

    def _on_start_training(self):
        if self.is_training:
            messagebox.showinfo("Info", "Training already in progress.")
            return
        
        self.is_training = True
        self.stop_requested = False
        self.control_panel.enable_training(False)
        self.control_panel.set_status("Initializing training...")
        
        model_type = self.control_panel.get_model_type()
        if model_type == 'AutoencoderMLP':
            self.visualization_frame.clear_loss_history('Autoencoder')
            self.visualization_frame.clear_loss_history('AutoencoderMLP')
        else:
            self.visualization_frame.clear_loss_history()
            
        self.visualization_frame.enable_clicking(False)
        
        if getattr(self, 'dataset_mode', 'manual') == 'mnist':
            if model_type == 'AutoencoderMLP':
                self.visualization_frame.switch_to_tab('ae_loss')
            else:
                self.visualization_frame.switch_to_tab('loss')
                
        learning_rate = self.control_panel.inputs.get_learning_rate()
        stopping_criteria, max_epochs, min_error = self.control_panel.inputs.get_stopping_criteria()
        batch_size = self.control_panel.inputs.get_batch_size()
        test_split = self.control_panel.inputs.get_test_split()
        task = self.control_panel.get_task_type()
        epochs = max_epochs

        if getattr(self, 'dataset_mode', 'manual') == 'mnist':
            (X_train, y_train), (X_test, y_test) = load_mnist_dataset(
                per_class_train=100,
                per_class_test=10
            )
            task = 'classification'
        else:
            X_train, X_test, y_train, y_test = self.data_handler.get_train_test_split(
                test_ratio=test_split,
                task=task,
            )
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.current_task = task
        self.visualization_frame.current_task = task

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
            
        if model_type != "AutoencoderMLP":
            if task == 'regression':
                input_size = 1
            else:
                if getattr(self, 'dataset_mode', 'manual') == 'mnist':
                    input_size = 784
                else:
                    input_size = 2
                    
            if model_type == 'MLP':
                architecture = self.control_panel.inputs.get_architecture()
                architecture[0] = input_size
                architecture[-1] = n_output_nodes
                arch_str = ','.join(map(str, architecture))
                self.control_panel.architecture_entry.delete(0, 'end')
                self.control_panel.architecture_entry.insert(0, arch_str)

        model, batch_size = self.training_manager.build_model(
            model_type, self.control_panel.inputs, task, n_output_nodes
        )
        
        if model_type == 'AutoencoderMLP':
            self._start_autoencoder_pipeline(X_train, y_train, epochs, stopping_criteria, min_error)
        else:
            self.after(100, lambda: self._run_training_standard(
                model, X_train, y_train, epochs, batch_size, stopping_criteria, min_error
            ))

    def _run_training_standard(self, model, X_train, y_train, epochs, batch_size, stopping_criteria, min_error):
        self.training_manager.start_training_standard(model, X_train, y_train, epochs, batch_size, stopping_criteria, min_error)
        self.after(100, self._poll_training_results)

    def _start_autoencoder_pipeline(self, X_train, y_train, epochs, stopping_criteria, min_error):
        encoder_dims = self.control_panel.inputs.get_encoder_architecture()
        ae_stop_mode, ae_epochs, ae_min_error_val = self.control_panel.inputs.get_ae_stopping_config()
        freeze_encoder = self.control_panel.inputs.get_freeze_encoder()
        recon_samples = self.control_panel.inputs.get_recon_samples()
        
        learning_rate = self.control_panel.inputs.get_learning_rate()
        use_momentum, momentum_factor = self.control_panel.inputs.get_momentum_config()
        batch_size = self.control_panel.inputs.get_batch_size()
        
        num_encoder_layers = len(encoder_dims) - 1
        encoder_activations = ['relu'] * (num_encoder_layers - 1) + ['relu']
        decoder_activations = ['relu'] * (num_encoder_layers - 1) + ['sigmoid']
        ae_activations = encoder_activations + decoder_activations
        
        self.control_panel.set_status("Stage 1/2: Training Autoencoder...")
        self.update()
        
        params = {
            'encoder_dims': encoder_dims,
            'activations': ae_activations,
            'learning_rate': learning_rate,
            'use_momentum': use_momentum,
            'momentum_factor': momentum_factor,
            'ae_epochs': ae_epochs,
            'ae_stop_mode': ae_stop_mode,
            'ae_min_error': ae_min_error_val
        }
        
        self.stage2_context = {
            'X_train': X_train, 'y_train': y_train,
            'epochs': epochs, 'batch_size': batch_size,
            'stopping_criteria': stopping_criteria, 'min_error': min_error,
            'encoder_dims': encoder_dims, 'freeze_encoder': freeze_encoder,
            'recon_samples': recon_samples,
            'learning_rate': learning_rate, 'use_momentum': use_momentum, 'momentum_factor': momentum_factor,
            'l2_lambda': self.control_panel.inputs.get_l2_lambda(),
            'mlp_architecture': self.control_panel.inputs.get_architecture(),
            'mlp_activations': self.control_panel.inputs.get_activation_functions(),
        }
        
        # Check if encoder is already loaded
        if hasattr(self, 'trained_ae') and self.trained_ae:
            self.control_panel.set_status("Using pre-loaded encoder. Skipping Stage 1...")
            self.update()
            self.after(100, lambda: self._start_stage2_mlp(self.trained_ae))
        else:
            self.after(100, lambda: self._run_training_autoencoder_deferred(params, X_train, y_train, batch_size))

    def _run_training_autoencoder_deferred(self, params, X_train, y_train, batch_size):
        self.training_manager.start_training_autoencoder(params, X_train, y_train, batch_size)
        self.after(100, self._poll_training_results)

    def _on_stop_training(self):
        print("DEBUG: Stop request received in main.py")
        if self.is_training:
            self.stop_requested = True
            self.training_manager.request_stop()
            self.control_panel.set_status("Stop requested. Finishing current epoch...")
        else:
            messagebox.showinfo("Info", "No training in progress to stop.")

    def _finish_training(self):
        self.is_training = False
        self.stop_requested = False
        self.training_manager.is_training = False
        self.control_panel.enable_training(True)
        self.visualization_frame.enable_clicking(True)
        self.control_panel.set_status("Training stopped.")
        messagebox.showinfo("Info", "Training stopped.")

    def _poll_training_results(self):
        try:
            queue_item = self.training_manager.training_queue.get_nowait()
            result_type, data1, data2, data3 = queue_item
            
            model = data3
            is_autoencoder = model.__class__.__name__ == 'Autoencoder'
            
            if result_type == 'epoch':
                epoch, loss = data1, data2
                
                if is_autoencoder:
                    self._handle_epoch_result(epoch, loss, model)
                else:
                    model_type = self.control_panel.get_model_type()
                    self.control_panel.set_status(f"Epoch {epoch} - Train Error: {loss:.4f}")
                    self.visualization_frame.update_loss_plot(epoch, loss, model_type=model_type)
                    
                    if getattr(self, 'dataset_mode', 'manual') == 'manual':
                        if epoch % 50 == 0 or epoch == 1:
                            self.visualization_frame.update_decision_boundary(
                                model, self.X_train, self.y_train, self.data_handler, tab_name='train', task=self.current_task
                            )
                
                self.after(20, self._poll_training_results)
                
            elif result_type == 'converged':
                epoch, loss = data1, data2
                if is_autoencoder:
                    self.control_panel.set_status(f"Stage 1 Complete! AE Converged (Loss {loss:.4f})")
                    self.visualization_frame.update_loss_plot(epoch, loss, model_type='Autoencoder')
                    # Save encoder and finish Stage 1
                    self.trained_ae = model
                    self._finish_training()
                    messagebox.showinfo("Stage 1 Complete", 
                                      "Autoencoder training complete!\n\n"
                                      "You can now:\n"
                                      "1. Save the encoder using 'Save Encoder'\n"
                                      "2. Press 'START TRAINING' to begin MLP training (Stage 2)")
                else:
                    self.control_panel.set_status(f"Converged! Error {loss:.6f}")
                    self.visualization_frame.update_loss_plot(epoch, loss, model_type=self.control_panel.get_model_type())
                    self._on_training_completed(model)
                
            elif result_type == 'complete':
                if is_autoencoder:
                    final_loss = getattr(self, 'last_ae_loss', 0.0)
                    self.control_panel.set_status(f"Stage 1 Complete! Final Loss: {final_loss:.6f}")
                    # Save encoder and finish Stage 1
                    self.trained_ae = model
                    self._finish_training()
                    messagebox.showinfo("Stage 1 Complete", 
                                      "Autoencoder training complete!\n\n"
                                      "You can now:\n"
                                      "1. Save the encoder using 'Save Encoder'\n"
                                      "2. Press 'START TRAINING' to begin MLP training (Stage 2)")
                else:
                    self._on_training_completed(model)
                
            elif result_type == 'stopped':
                self.control_panel.set_status("Training interrupted.")
                self._finish_training()
                
            elif result_type == 'error':
                self.control_panel.set_status(f"Training error: {data1}")
                self._finish_training()
        
        except queue.Empty:
            if self.is_training:
                self.after(20, self._poll_training_results)

    def _handle_epoch_result(self, epoch, loss, trained_ae):
        self.last_ae_loss = loss
        self.control_panel.set_status(f"Stage 1/2: AE Epoch {epoch} - Recon Loss: {loss:.6f}")
        self.visualization_frame.update_loss_plot(epoch, loss, model_type='Autoencoder')
        self.update_idletasks()
        
        ctx = self.stage2_context
        if epoch % 10 == 0:
            try:
                if 'X_train' in ctx:
                    # Always show 10 samples (one per digit 0-9)
                    num_samples = 10
                    sample_indices = []
                    y_train = ctx['y_train']
                    X_train = ctx['X_train']
                    
                    # Try to get one sample per class (0-9)
                    for target_class in range(num_samples):
                        for idx, label in enumerate(y_train):
                            if label == target_class and idx not in sample_indices:
                                sample_indices.append(idx)
                                break
                    
                    # Fill remaining slots if needed
                    if len(sample_indices) < num_samples:
                        remaining = num_samples - len(sample_indices)
                        for i in range(len(X_train)):
                            if i not in sample_indices and remaining > 0:
                                sample_indices.append(i)
                                remaining -= 1
                    
                    X_sample = [X_train[i] for i in sample_indices]
                    X_reconstructed = trained_ae.reconstruct(X_sample)
                    
                    X_sample_np = np.array(X_sample)
                    X_recon_np = np.array(X_reconstructed)
                    mse_per_sample = np.mean((X_sample_np - X_recon_np)**2, axis=1)
                    
                    self.visualization_frame.update_reconstruction(X_sample, X_reconstructed, mse_per_sample)
                    self.update_idletasks()
            except Exception as e:
                print(f"Reconstruction update error: {e}")
                import traceback
                traceback.print_exc()

    def _start_stage2_mlp(self, trained_ae):
        if self.stop_requested:
            self._finish_training()
            return
            
        self.trained_ae = trained_ae
        ctx = self.stage2_context
        
        self.visualization_frame.switch_to_tab('mlp_loss')
        self.control_panel.set_status("Stage 1 Complete. Starting Stage 2: MLP Training...")
        
        encoder_dims = ctx['encoder_dims']
        latent_dim = encoder_dims[-1]
        
        self.training_manager.start_training_stage2(trained_ae, ctx, latent_dim)
        self.after(100, self._poll_training_results)

    def _on_training_completed(self, model):
        self.is_training = False
        self.trained_model = model
        self.training_manager.is_training = False
        self.control_panel.enable_training(True)
        self.visualization_frame.enable_clicking(True)
        
        if self.stop_requested:
            self.control_panel.set_status("Training interrupted by user.")
            self._finish_training()
            return

        if hasattr(self, 'X_test') and len(self.X_test) > 0:
            if getattr(self, 'dataset_mode', 'manual') == 'manual':
                self.visualization_frame.update_decision_boundary(
                    model, self.X_test, self.y_test, self.data_handler, tab_name='test', task=self.current_task
                )
            
            y_pred = model.predict(self.X_test)
            
            if self.current_task == 'regression':
                mse = np.mean((np.array(y_pred).flatten() - np.array(self.y_test).flatten())**2)
                self.control_panel.set_status(f"Training complete! Test MSE: {mse:.4f}")
                messagebox.showinfo("Success", f"Training completed successfully!\nTest MSE: {mse:.4f}")
            else:
                y_pred = np.array(y_pred)
                y_test = np.array(self.y_test)
                accuracy = np.mean(y_pred == y_test) * 100
                self.control_panel.set_status(f"Training complete! Test Accuracy: {accuracy:.2f}%")
                messagebox.showinfo("Success", f"Training completed successfully!\nTest Accuracy: {accuracy:.2f}%")
            
            if getattr(self, 'dataset_mode', 'manual') == 'manual':
                self.visualization_frame.switch_to_tab('test')
            else:
                model_type = self.control_panel.get_model_type()
                if model_type == 'AutoencoderMLP':
                    self.visualization_frame.switch_to_tab('mlp_loss')
                else:
                    self.visualization_frame.switch_to_tab('loss')
        else:
            self.control_panel.set_status("Training completed!")
            messagebox.showinfo("Success", "Training completed successfully!")

    def _on_save_model(self):
        if not self.trained_model:
            messagebox.showwarning("Warning", "No trained model to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Save the entire model object (simplest approach)
                # Or dictionary of parameters if we want to be strict, but object is fine for now
                with open(file_path, 'wb') as f:
                    pickle.dump(self.trained_model, f)
                messagebox.showinfo("Success", "Model saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")

    def _on_load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                
                self.trained_model = model
                self.control_panel.set_status("Model loaded.")
                messagebox.showinfo("Success", "Model loaded successfully!")
                
                # If we are in handwriting mode, update the existing window if open
                if self.handwriting_window and self.handwriting_window.winfo_exists():
                    self.handwriting_window.model = model
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")

    def _on_test_handwriting(self):
        if not self.trained_model:
            messagebox.showwarning("Warning", "Please training or load a model first!")
            return
            
        if self.handwriting_window is None or not self.handwriting_window.winfo_exists():
            self.handwriting_window = HandwritingTester(self, self.trained_model)
            self.handwriting_window.focus()
        else:
            self.handwriting_window.focus()
    
    def _on_save_encoder(self):
        if not hasattr(self, 'trained_ae') or not self.trained_ae:
            messagebox.showwarning("Warning", "No trained Autoencoder to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")],
            initialdir=WEIGHTS_DIR,
            title="Save Encoder"
        )
        
        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.trained_ae, f)
                messagebox.showinfo("Success", "Encoder saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save encoder: {e}")
    
    def _on_load_encoder(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")],
            initialdir=WEIGHTS_DIR,
            title="Load Encoder"
        )
        
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    encoder = pickle.load(f)
                
                self.trained_ae = encoder
                self.control_panel.set_status("Encoder loaded.")
                messagebox.showinfo("Success", "Encoder loaded successfully!\nYou can now train MLP with this encoder.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load encoder: {e}")

    def _update_class_radios(self):
        classes = self.data_handler.classes
        colors = [self.data_handler.get_color(i) for i in range(len(classes))]
        self.control_panel.update_class_radios(classes, colors)


def main():
    app = NeuralNetworkVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()
