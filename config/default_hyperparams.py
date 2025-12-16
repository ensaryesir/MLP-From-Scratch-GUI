"""
Default Hyperparameter Configuration
====================================

This file contains all default hyperparameter values for different models and tasks.
Users can easily modify these values without touching the source code.

Structure:
- Each model type has its own section
- Within each model, there are presets for different scenarios (task type, dataset mode)
- Values are organized as nested dictionaries for easy access
"""

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
    'architecture': [784, 128, 10],  # Generic fallback
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
        'epochs': 100,
        'min_error': 0.1,
    },
    'regression': {
        'learning_rate': 0.01,
        'epochs': 200,
        'min_error': 0.1,
    }
}

# ============================================================================
# DELTA RULE DEFAULTS
# ============================================================================

DELTA_RULE_DEFAULTS = {
    'classification': {
        'learning_rate': 0.01,
        'epochs': 100,
        'min_error': 0.1,
    },
    'regression': {
        'learning_rate': 0.01,
        'epochs': 200,
        'min_error': 0.1,
    }
}

# ============================================================================
# MLP (MULTI-LAYER PERCEPTRON) DEFAULTS
# ============================================================================

MLP_DEFAULTS = {
    'mnist': {
        # MNIST dataset-specific presets
        'architecture': '784,128,64,10',
        'activation_hidden': 'relu',
        'activation_output': 'softmax',
        'learning_rate': 0.1,
        'batch_size': 32,
        'l2_lambda': 0.0001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 100,
        'min_error': 0.1,
        'test_split': 20,
    },
    'manual_classification': {
        # Manual 2D data classification
        'architecture': '2,10,2',
        'activation_hidden': 'tanh',
        'activation_output': 'softmax',
        'learning_rate': 0.01,
        'batch_size': 16,
        'l2_lambda': 0.001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 500,
        'min_error': 0.002,
    },
    'manual_regression': {
        # Manual 2D data regression
        'architecture': '1,10,1',
        'activation_hidden': 'tanh',
        'activation_output': 'linear',
        'learning_rate': 0.01,
        'batch_size': 16,
        'l2_lambda': 0.001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 500,
        'min_error': 0.002,
    }
}

# ============================================================================
# AUTOENCODER-MLP DEFAULTS
# ============================================================================

AUTOENCODER_MLP_DEFAULTS = {
    'mnist': {
        # MNIST dataset with autoencoder feature extraction
        'architecture': '32,64,10',  # MLP classifier part (input is encoder output)
        'encoder_architecture': '784,128,32',  # Encoder layers
        'activation_hidden': 'relu',
        'activation_output': 'softmax',
        'learning_rate': 0.1,
        'batch_size': 32,
        'l2_lambda': 0.0001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 100,  # Classifier training epochs
        'min_error': 0.1,
        # Autoencoder-specific parameters
        'ae_epochs': 50,  # Autoencoder pre-training epochs
        'ae_stopping_criteria': 'epochs',
        'ae_min_error': 0.001,
        'freeze_encoder': True,  # Freeze encoder weights during classifier training
        'recon_samples': 10,  # Number of reconstruction samples to visualize
    },
    'manual': {
        # Manual mode (rarely used with autoencoder)
        'architecture': '2,4,2',
        'encoder_architecture': '2,16,2',
        'activation_hidden': 'relu',
        'activation_output': 'softmax',
        'learning_rate': 0.01,
        'batch_size': 16,
        'l2_lambda': 0.001,
        'use_momentum': True,
        'momentum_factor': 0.9,
        'epochs': 500,
        'min_error': 0.002,
        # Autoencoder-specific (uses same defaults as MNIST)
        'ae_epochs': 50,
        'ae_stopping_criteria': 'epochs',
        'ae_min_error': 0.001,
        'freeze_encoder': True,
        'recon_samples': 10,
    }
}

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_defaults(model_type, task=None, dataset_mode=None):
    """
    Get default hyperparameters for a specific configuration.
    
    Parameters
    ----------
    model_type : str
        One of: 'Perceptron', 'DeltaRule', 'MLP', 'AutoencoderMLP'
    task : str, optional
        One of: 'classification', 'regression' (not used for AutoencoderMLP)
    dataset_mode : str, optional
        One of: 'manual', 'mnist' (used for MLP and AutoencoderMLP)
    
    Returns
    -------
    dict
        Dictionary of default hyperparameter values
    """
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
