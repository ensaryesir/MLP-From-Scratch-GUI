import threading
import queue
import time
import numpy as np

from algorithms.single_layer import Perceptron, DeltaRule
from algorithms.mlp import MLP
from algorithms.autoencoder import Autoencoder
from algorithms.mlp_with_encoder import MLPWithEncoder

class TrainingManager:
    def __init__(self, update_callback, completion_callback, error_callback):
        self.training_queue = queue.Queue()
        self.update_callback = update_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback
        self.stop_requested = False
        self.is_training = False
        self.current_model = None
        self.training_thread = None

    def request_stop(self):
        print("DEBUG: TrainingManager.request_stop called")
        if self.is_training:
            self.stop_requested = True

    def build_model(self, model_type, inputs, task, n_output_nodes):
        if model_type == "Perceptron":
            model = Perceptron(
                learning_rate=inputs.get_learning_rate(),
                n_classes=n_output_nodes,
                task=task,
            )
            return model, 1
        elif model_type == "DeltaRule":
            model = DeltaRule(
                learning_rate=inputs.get_learning_rate(),
                n_classes=n_output_nodes,
                task=task,
            )
            return model, 1 # Batch size ignored
        elif model_type == "AutoencoderMLP":
            return None, inputs.get_batch_size()
        else: # MLP
            architecture = inputs.get_architecture()
            # Input size must be set by caller before passing inputs, or inputs handle it.
            # Assuming inputs.get_architecture() returns the list we modified in main.py
            
            architecture[-1] = n_output_nodes
            
            activation_funcs_raw = inputs.get_activation_functions()
            hidden_activation = activation_funcs_raw[0]
            output_activation = activation_funcs_raw[1]
            
            num_layers = len(architecture) - 1
            activation_funcs = [hidden_activation] * (num_layers - 1) + [output_activation]
            
            l2_lambda = inputs.get_l2_lambda()
            use_momentum, momentum_factor = inputs.get_momentum_config()
            
            model = MLP(
                layer_dims=architecture,
                activation_funcs=activation_funcs,
                learning_rate=inputs.get_learning_rate(),
                l2_lambda=l2_lambda,
                task=task,
                use_momentum=use_momentum,
                momentum_factor=momentum_factor,
            )
            return model, inputs.get_batch_size()

    def start_training_standard(self, model, X_train, y_train, epochs, batch_size, stopping_criteria, min_error):
        self.is_training = True
        self.stop_requested = False
        self.current_model = model
        
        # Clear queue
        with self.training_queue.mutex:
            self.training_queue.queue.clear()
        
        def stop_check():
            if self.stop_requested:
                print(f"DEBUG: Stop callback Triggered!")
                return True
            return False
            
        # Perceptron and DeltaRule don't support batch_size in fit
        if isinstance(model, (Perceptron, DeltaRule)):
            fit_generator = model.fit(X_train, y_train, epochs=epochs, stop_callback=stop_check)
        else:
            fit_generator = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, stop_callback=stop_check)

        
        self.training_thread = threading.Thread(
            target=self._worker_standard,
            args=(fit_generator, stopping_criteria, min_error),
            daemon=True
        )
        self.training_thread.start()

    def start_training_autoencoder(self, params, X_train, y_train, batch_size):
        self.is_training = True
        self.stop_requested = False
        
        encoder_dims = params['encoder_dims']
        activation_funcs = params['activations']
        
        autoencoder = Autoencoder(
            encoder_dims=encoder_dims,
            activation_funcs=activation_funcs,
            learning_rate=params['learning_rate'],
            use_momentum=params['use_momentum'],
            momentum_factor=params['momentum_factor'],
        )
        self.current_model = autoencoder
        
        with self.training_queue.mutex:
            self.training_queue.queue.clear()

        stop_check = lambda: self.stop_requested
            
        fit_generator = autoencoder.fit(X_train, epochs=params['ae_epochs'], batch_size=batch_size, stop_callback=stop_check)
        
        self.training_thread = threading.Thread(
            target=self._worker_autoencoder,
            args=(fit_generator, params['ae_stop_mode'], params['ae_min_error']),
            daemon=True
        )
        self.training_thread.start()
        
        return autoencoder

    def start_training_stage2(self, trained_ae, ctx, input_dim):
        self.is_training = True
        self.stop_requested = False
        
        encoder_params = trained_ae.get_encoder_weights()
        
        mlp_layer_dims = ctx['mlp_architecture']
        mlp_layer_dims[0] = input_dim
        
        activation_hidden = ctx['mlp_activations'][0]
        activation_output = ctx['mlp_activations'][1]
        mlp_activations = [activation_hidden] * (len(mlp_layer_dims) - 2) + [activation_output]
        
        num_encoder_layers = len(ctx['encoder_dims']) - 1
        encoder_activations = ['relu'] * num_encoder_layers
        
        mlp_model = MLPWithEncoder(
            encoder_params=encoder_params,
            encoder_dims=ctx['encoder_dims'],
            encoder_activations=encoder_activations,
            mlp_layer_dims=mlp_layer_dims,
            mlp_activations=mlp_activations,
            learning_rate=ctx['learning_rate'],
            l2_lambda=ctx['l2_lambda'],
            freeze_encoder=ctx['freeze_encoder'],
            use_momentum=ctx['use_momentum'],
            momentum_factor=ctx['momentum_factor']
        )
        self.current_model = mlp_model
        
        with self.training_queue.mutex:
            self.training_queue.queue.clear()
            
        stop_check = lambda: self.stop_requested

        fit_generator = mlp_model.fit(ctx['X_train'], ctx['y_train'], epochs=ctx['epochs'], batch_size=ctx['batch_size'], stop_callback=stop_check)
        
        self.training_thread = threading.Thread(
            target=self._worker_standard,
            args=(fit_generator, ctx['stopping_criteria'], ctx['min_error']),
            daemon=True
        )
        self.training_thread.start()

    def _worker_standard(self, fit_generator, stopping_criteria, min_error):
        last_update_time = 0
        update_interval = 0.05
        
        try:
            for epoch, loss, model in fit_generator:
                if self.stop_requested:
                    self.training_queue.put(('stopped', None, None, None))
                    return
                
                if stopping_criteria == 'error' and loss <= min_error:
                    self.training_queue.put(('converged', epoch, loss, model))
                    return
                
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    self.training_queue.put(('epoch', epoch, loss, model))
                    last_update_time = current_time
            
            self.training_queue.put(('epoch', epoch, loss, model))
            self.training_queue.put(('complete', None, None, self.current_model))
            
        except Exception as e:
            self.training_queue.put(('error', str(e), None, None))

    def _worker_autoencoder(self, fit_generator, ae_stop_mode, ae_min_error_val):
        last_update_time = 0
        update_interval = 0.05
        
        try:
            for epoch, loss, trained_ae in fit_generator:
                if self.stop_requested:
                    self.training_queue.put(('stopped', None, None, None))
                    return
                
                if ae_stop_mode == 'error' and loss <= ae_min_error_val:
                    self.training_queue.put(('converged', epoch, loss, trained_ae))
                    return
                
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    self.training_queue.put(('epoch', epoch, loss, trained_ae))
                    last_update_time = current_time
            
            self.training_queue.put(('epoch', epoch, loss, trained_ae))
            self.training_queue.put(('complete', None, None, self.current_model))
            
        except Exception as e:
            self.training_queue.put(('error', str(e), None, None))
