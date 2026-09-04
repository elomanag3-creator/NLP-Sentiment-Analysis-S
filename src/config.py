"""
Configuration Module
Centralized configuration for the NLP sentiment analysis project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, '..', 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, '..', 'results')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, '..', 'notebooks')

# Data configuration
DATA_CONFIG = {
    'input_file': os.path.join(DATA_DIR, 'reviews.csv'),
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'max_features': 100,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8,
    'sublinear_tf': True,
    'lowercase': True,
    'remove_stopwords': True,
    'remove_special_chars': True,
    'min_token_length': 2
}

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'class_weight_balance': True,
    'cv_folds': 5,
    
    # Logistic Regression
    'lr_config': {
        'class_weight': 'balanced',
        'max_iter': 1000,
        'solver': 'lbfgs',
        'random_state': 42
    },
    
    # XGBoost
    'xgb_config': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42,
        'scale_pos_weight': 1,
        'eval_metric': 'logloss'
    },
    
    # Ensemble
    'ensemble_config': {
        'voting': 'soft',
        'random_state': 42
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'confusion_matrix': True,
    'classification_report': True,
    'roc_curve': True
}

# File paths for serialized objects
MODEL_PATHS = {
    'ensemble_model': os.path.join(MODELS_DIR, 'sentiment_model.pkl'),
    'preprocessor': os.path.join(DATA_DIR, 'preprocessor.pkl'),
    'training_history': os.path.join(MODELS_DIR, 'training_history.pkl')
}

DATA_PATHS = {
    'X_train': os.path.join(DATA_DIR, 'X_train.pkl'),
    'X_test': os.path.join(DATA_DIR, 'X_test.pkl'),
    'y_train': os.path.join(DATA_DIR, 'y_train.pkl'),
    'y_test': os.path.join(DATA_DIR, 'y_test.pkl')
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'style': 'default',
    'figure_dpi': 300,
    'figure_format': 'png',
    'figure_size': (10, 6),
    'colors': {
        'positive': '#2ca02c',
        'negative': '#d62728',
        'neutral': '#1f77b4'
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(RESULTS_DIR, 'project.log')
}

# API/Serving configuration (for future use)
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'workers': 4
}


def get_config(key, default=None):
    """
    Get configuration value by key
    
    Parameters:
    - key: configuration key (e.g., 'DATA_CONFIG', 'MODEL_CONFIG')
    - default: default value if key not found
    
    Returns:
    - Configuration value or default
    """
    return globals().get(key, default)


def print_config():
    """Print all configuration settings"""
    print("="*70)
    print("PROJECT CONFIGURATION")
    print("="*70)
    
    print("\nDIRECTORIES:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Models Directory: {MODELS_DIR}")
    print(f"  Results Directory: {RESULTS_DIR}")
    
    print("\nDATA CONFIGURATION:")
    for key, value in DATA_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nPREPROCESSING CONFIGURATION:")
    for key, value in PREPROCESSING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nMODEL CONFIGURATION:")
    for key, value in MODEL_CONFIG.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\nEVALUATION CONFIGURATION:")
    for key, value in EVALUATION_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print_config()
