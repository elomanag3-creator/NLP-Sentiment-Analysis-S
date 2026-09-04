"""
Model Training Module
Trains Logistic Regression and XGBoost ensemble
"""

import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt


class SentimentModel:
    """Ensemble sentiment classification model"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.history = {}
    
    def build_ensemble(self, class_weight_balance=True):
        """
        Build ensemble classifier
        
        Parameters:
        - class_weight_balance: whether to balance classes
        """
        print("Building ensemble model...")
        
        # Calculate class weights for imbalanced data
        class_weight = 'balanced' if class_weight_balance else None
        
        # Model 1: Logistic Regression
        lr_model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        # Model 2: XGBoost
        # For imbalanced data, scale_pos_weight helps
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state,
            scale_pos_weight=1,  # Adjust if needed for class imbalance
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Voting Ensemble
        self.model = VotingClassifier(
            estimators=[
                ('lr', lr_model),
                ('xgb', xgb_model)
            ],
            voting='soft'  # Use probability predictions
        )
        
        print("✓ Ensemble model created:")
        print("  - Logistic Regression")
        print("  - XGBoost")
        print("  - Voting strategy: soft (probability)")
        
        return self.model
    
    def train(self, X_train, y_train):
        """Train the ensemble model"""
        print("\nTraining ensemble...")
        self.model.fit(X_train, y_train)
        print("✓ Training complete!")
    
    def cross_validate(self, X_train, y_train, cv=5):
        """
        Perform k-fold cross-validation
        
        Parameters:
        - cv: number of folds (default 5)
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_f1 = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='f1')
        cv_acc = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_prec = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='precision')
        cv_recall = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='recall')
        
        print(f"\nCross-Validation Results ({cv}-fold):")
        print(f"  F1-Score:  {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
        print(f"  Accuracy:  {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
        print(f"  Precision: {cv_prec.mean():.4f} (+/- {cv_prec.std():.4f})")
        print(f"  Recall:    {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
        
        self.history['cv_f1'] = cv_f1
        self.history['cv_acc'] = cv_acc
        self.history['cv_prec'] = cv_prec
        self.history['cv_recall'] = cv_recall
        
        return {
            'f1': cv_f1,
            'accuracy': cv_acc,
            'precision': cv_prec,
            'recall': cv_recall
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print("\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc_score:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Negative', 'Positive'],
            digits=4
        ))
        
        # Store results
        self.history['test_accuracy'] = accuracy
        self.history['test_precision'] = precision
        self.history['test_recall'] = recall
        self.history['test_f1'] = f1
        self.history['test_auc'] = auc_score
        self.history['confusion_matrix'] = cm
        self.history['y_pred'] = y_pred
        self.history['y_pred_proba'] = y_pred_proba
        self.history['y_test'] = y_test
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': cm
        }
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")


def train_pipeline(data_dir='data', models_dir='models'):
    """
    Complete training pipeline
    """
    
    print("="*70)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*70)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    with open(os.path.join(data_dir, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(data_dir, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(data_dir, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(data_dir, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    
    print(f"✓ Loaded training set: {X_train.shape}")
    print(f"✓ Loaded test set: {X_test.shape}")
    
    # Class distribution
    print(f"\nClass Distribution (Training):")
    print(f"  Positive: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
    print(f"  Negative: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
    
    # Initialize model
    model = SentimentModel()
    
    # Build ensemble
    model.build_ensemble(class_weight_balance=True)
    
    # Train
    model.train(X_train, y_train)
    
    # Cross-validate
    cv_results = model.cross_validate(X_train, y_train, cv=5)
    
    # Evaluate
    test_results = model.evaluate(X_test, y_test)
    
    # Save model
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, 'sentiment_model.pkl'))
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n✓ Model saved to {models_dir}/sentiment_model.pkl")
    print(f"✓ Best F1-Score: {model.history['test_f1']:.4f}")
    
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train_pipeline()
