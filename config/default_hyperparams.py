# ============================================================================
# SAFETY DEFAULTS (Fallback values if UI input is invalid)
# ============================================================================
UI_SAFETY_DEFAULTS = {
    'learning_rate': 0.001,
    'epochs': 100,
    'min_error': 0.001,
    'batch_size': 32,
    'l2_lambda': 0.0,
    'momentum_factor': 0.9,
    'test_split': 20.0,
    'architecture': [784, 128, 10], 
    'encoder_architecture': [784, 128, 32],
    'ae_epochs': 50,
    'recon_samples': 10,
    'ae_stopping_criteria': 'epochs',
    'ae_min_error': 0.001
}

# ============================================================================
# PERCEPTRON DEFAULTS
# ============================================================================

PERCEPTRON_DEFAULTS = {
    'classification': {
        'learning_rate': 0.01,
        'epochs': 1000,
        'min_error': 0.01,
    },
    'regression': {
        'learning_rate': 0.01,
        'epochs': 1000,
        'min_error': 0.01,
    }
}

# ============================================================================
# DELTA RULE DEFAULTS
# ============================================================================

DELTA_RULE_DEFAULTS = {
    'classification': {
        'learning_rate': 0.01,
        'epochs': 1000,
        'min_error': 0.01,
    },
    'regression': {
        'learning_rate': 0.01,
        'epochs': 1000,
        'min_error': 0.01,
    }
}

# ============================================================================
# MLP (MULTI-LAYER PERCEPTRON) DEFAULTS
# ============================================================================

MLP_DEFAULTS = {
    'mnist': {
        # MNIST dataset-specific presets
        'architecture': '784,256,10',
        'activation_hidden': 'relu', 
        'activation_output': 'softmax',
        'learning_rate': 0.01, 
        'batch_size': 64,      
        'l2_lambda': 0.0001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 30,         
        'min_error': 0.01,
        'test_split': 20,
    },
    'manual_classification': {
        'architecture': '2,16,16,2',
        'activation_hidden': 'relu',
        'activation_output': 'softmax',
        'learning_rate': 0.01, 
        'batch_size': 16,        
        'l2_lambda': 0.0001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 2000,
        'min_error': 0.001,
    },
    'manual_regression': {
        'architecture': '1,16,16,1', 
        'activation_hidden': 'tanh',
        'activation_output': 'linear',
        'learning_rate': 0.01, 
        'batch_size': 8,
        'l2_lambda': 0.0001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 2000,
        'min_error': 0.0001,
    }
}

# ============================================================================
# AUTOENCODER-MLP DEFAULTS
# ============================================================================

AUTOENCODER_MLP_DEFAULTS = {
    'mnist': {
        # --- STAGE 1: ENCODER (Feature Extractor) ---
        # We compress the 784-pixel data into a 128-dimensional "summary" vector.
        # 64 might be too narrow; 128 captures richer features.
        'encoder_architecture': '784,128', 
        'ae_epochs': 20,                # Sufficient time for the Encoder to learn patterns.
        'ae_min_error': 0.005,          # Reconstruction error should be low.
        'freeze_encoder': True,         # IMPORTANT: Encoder weights are frozen after training.
        'recon_samples': 10,
        
        # --- STAGE 2: MLP (Classifier) ---
        # NOTE: Input layer (128) MUST match Encoder output.
        'architecture': '128,64,10',    
        'activation_hidden': 'relu',    # ReLU is standard.
        'activation_output': 'softmax',
        'learning_rate': 0.01,          # MLP learns quickly while Encoder is frozen.
        'batch_size': 128,              # Ideal for speed and stability.
        'l2_lambda': 0.0001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 30,
        'min_error': 0.01,
    },

    'manual': {
        # --- Manual Data (XOR / Spiral) ---
        # Compressing 2D data (2->1) causes data loss.
        # Instead, we use "Feature Expansion" (2->16) to map data to a higher 
        # dimension, making separation easier (like Kernel Trick).
        'encoder_architecture': '2,16', 
        'ae_epochs': 500,               # AE training is fast on manual data, can be kept high.
        'ae_min_error': 0.001,
        'freeze_encoder': True,         
        'recon_samples': 10,

        # --- MLP ---
        # Input (16) must match Encoder output.
        'architecture': '16,8,2',       # Take 16 features, reduce to 8, classify into 2 classes.
        'activation_hidden': 'relu',
        'activation_output': 'softmax',
        'learning_rate': 0.01,
        'batch_size': 16,               # 16 is more stable than 4 for manual data.
        'l2_lambda': 0.0001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 2000,
        'min_error': 0.001,
    }
}

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_defaults(model_type, task=None, dataset_mode=None):
    if model_type == 'Perceptron':
        return PERCEPTRON_DEFAULTS.get(task, PERCEPTRON_DEFAULTS['classification'])
    
    elif model_type == 'DeltaRule':
        return DELTA_RULE_DEFAULTS.get(task, DELTA_RULE_DEFAULTS['classification'])
    
    elif model_type == 'AutoencoderMLP':
        return AUTOENCODER_MLP_DEFAULTS.get(dataset_mode, AUTOENCODER_MLP_DEFAULTS['mnist'])
    
    else:  # MLP
        if dataset_mode == 'mnist':
            return MLP_DEFAULTS['mnist']
        elif task == 'classification':
            return MLP_DEFAULTS['manual_classification']
        else:  # regression
            return MLP_DEFAULTS['manual_regression']
