"""
Main Entry Point
Complete NLP Sentiment Analysis Pipeline
"""

import sys
import os
import sys, io, os
from datetime import datetime


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


os.makedirs("results", exist_ok=True)
_log_file = open("results/run_log.txt", "w", encoding="utf-8")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess_pipeline
from train import train_pipeline
from predict import SentimentPredictor, run_prediction_demo
from analysis import generate_all_visualizations


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)


def main():
    """Run complete pipeline"""
    
    print_header("NLP SENTIMENT ANALYSIS PROJECT")
    print("Predicting product review sentiment using ML ensemble\n")
    
    # Step 1: Preprocessing
    print_header("STEP 1: DATA PREPROCESSING")
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        input_path='data/reviews.csv',
        output_dir='data',
        test_size=0.2,
        max_features=100
    )
    
    # Step 2: Model Training
    print_header("STEP 2: MODEL TRAINING")
    
    model, _, _, _, _ = train_pipeline(
        data_dir='data',
        models_dir='models'
    )
    
    # Step 3: Predictions
    print_header("STEP 3: PREDICTION & INFERENCE")
    
    predictor = SentimentPredictor(
        model_path='models/sentiment_model.pkl',
        preprocessor_path='data/preprocessor.pkl'
    )
    
    # Run prediction demo
    run_prediction_demo()
    
    generate_all_visualizations(
        model.model,
        model.history,
        model.history['y_test'],
        model.history['y_pred'],
        model.history['y_pred_proba'],
        preprocessor,
    )

    # Step 4: Summary
    print_header("PIPELINE COMPLETE")
    
    print(f"""
✓ Project completed successfully!

Results:
  - Model F1-Score: {model.history['test_f1']:.4f}
  - Accuracy: {model.history['test_accuracy']:.4f}
  - Precision: {model.history['test_precision']:.4f}
  - Recall: {model.history['test_recall']:.4f}
  - AUC-ROC: {model.history['test_auc']:.4f}

Files Generated:
  - data/reviews.csv - Original dataset
  - data/X_train.pkl, X_test.pkl - Feature matrices
  - data/y_train.pkl, y_test.pkl - Labels
  - data/preprocessor.pkl - Fitted preprocessor
  - models/sentiment_model.pkl - Trained model

Next Steps:
  1. Review results in results/ directory
  2. Use predict.py for inference on new data
  3. Check CV_TEMPLATE.txt for CV description
  
For more info, see README.md
""")
    
    return model, predictor


if __name__ == "__main__":
    main()
