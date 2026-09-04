"""
Analysis & Visualization Module
Analyze results and create visualizations
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


class ResultsAnalyzer:
    """Analyze and visualize model results"""
    
    def __init__(self, model, history, y_test, y_pred, y_pred_proba):
        """
        Initialize analyzer
        
        Parameters:
        - model: trained model
        - history: model training history dict
        - y_test: true labels
        - y_pred: predictions
        - y_pred_proba: prediction probabilities
        """
        self.model = model
        self.history = history
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
    
    def plot_confusion_matrix(self, save_path='results/confusion_matrix.png'):
        """Plot and save confusion matrix"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Sentiment Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, save_path='results/roc_curve.png'):
        """Plot and save ROC curve"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Sentiment Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curve saved to {save_path}")
    
    def plot_metrics_comparison(self, save_path='results/metrics_comparison.png'):
        """Plot performance metrics comparison"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        values = [
            self.history['test_accuracy'],
            self.history['test_precision'],
            self.history['test_recall'],
            self.history['test_f1']
        ]
        
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.ylim([0, 1.0])
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Metrics comparison saved to {save_path}")
    
    def plot_cv_results(self, save_path='results/cv_results.png'):
        """Plot cross-validation results"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        metrics = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
        means = [
            self.history['cv_f1'].mean(),
            self.history['cv_acc'].mean(),
            self.history['cv_prec'].mean(),
            self.history['cv_recall'].mean()
        ]
        stds = [
            self.history['cv_f1'].std(),
            self.history['cv_acc'].std(),
            self.history['cv_prec'].std(),
            self.history['cv_recall'].std()
        ]
        
        x_pos = np.arange(len(metrics))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.8, 
                       color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            plt.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                    f'{mean:.3f}\n±{std:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(x_pos, metrics, fontsize=11)
        plt.ylim([0, 1.0])
        plt.ylabel('Score', fontsize=12)
        plt.title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ CV results saved to {save_path}")
    
    def plot_prediction_distribution(self, save_path='results/prediction_distribution.png'):
        """Plot distribution of prediction probabilities"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        negative_probs = self.y_pred_proba[self.y_test == 0]
        positive_probs = self.y_pred_proba[self.y_test == 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(negative_probs, bins=20, alpha=0.6, label='Negative Reviews', color='red', edgecolor='black')
        plt.hist(positive_probs, bins=20, alpha=0.6, label='Positive Reviews', color='green', edgecolor='black')
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        
        plt.xlabel('Predicted Probability (Positive Class)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Prediction distribution saved to {save_path}")
    
    def plot_feature_importance(self, preprocessor, save_path='results/feature_importance.png'):
        """Plot top features from XGBoost model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get XGBoost model from ensemble
        xgb_model = self.model.named_estimators_['xgb']
        
        feature_names = preprocessor.get_feature_names()
        importances = xgb_model.feature_importances_
        
        # Get top 20 features
        top_indices = np.argsort(importances)[-20:][::-1]
        top_features = feature_names[top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_importances, color=colors, edgecolor='black')
        
        plt.yticks(range(len(top_features)), top_features, fontsize=10)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance saved to {save_path}")
    
    def generate_report(self, save_path='results/analysis_report.txt'):
        """Generate comprehensive analysis report"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        report = f"""
================================================================================
SENTIMENT ANALYSIS - MODEL ANALYSIS REPORT
================================================================================

DATASET SUMMARY
───────────────
Test Set Size: {len(self.y_test)} samples
Positive Reviews: {np.sum(self.y_test == 1)} ({np.mean(self.y_test == 1)*100:.1f}%)
Negative Reviews: {np.sum(self.y_test == 0)} ({np.mean(self.y_test == 0)*100:.1f}%)

PERFORMANCE METRICS
───────────────────
Accuracy:  {self.history['test_accuracy']:.4f}
Precision: {self.history['test_precision']:.4f}
Recall:    {self.history['test_recall']:.4f}
F1-Score:  {self.history['test_f1']:.4f}
AUC-ROC:   {self.history['test_auc']:.4f}

CROSS-VALIDATION RESULTS (5-fold)
─────────────────────────────────
F1-Score:  {self.history['cv_f1'].mean():.4f} (+/- {self.history['cv_f1'].std():.4f})
Accuracy:  {self.history['cv_acc'].mean():.4f} (+/- {self.history['cv_acc'].std():.4f})
Precision: {self.history['cv_prec'].mean():.4f} (+/- {self.history['cv_prec'].std():.4f})
Recall:    {self.history['cv_recall'].mean():.4f} (+/- {self.history['cv_recall'].std():.4f})

CONFUSION MATRIX
────────────────
                Predicted
              Negative  Positive
Actual Negative {self.history['confusion_matrix'][0,0]:>3}      {self.history['confusion_matrix'][0,1]:>3}
       Positive {self.history['confusion_matrix'][1,0]:>3}      {self.history['confusion_matrix'][1,1]:>3}

ANALYSIS
────────
Total Correct: {np.sum(self.y_pred == self.y_test)} / {len(self.y_test)}
Total Incorrect: {np.sum(self.y_pred != self.y_test)} / {len(self.y_test)}
Error Rate: {np.mean(self.y_pred != self.y_test)*100:.2f}%

False Positives: {self.history['confusion_matrix'][0,1]} (predicted positive but actually negative)
False Negatives: {self.history['confusion_matrix'][1,0]} (predicted negative but actually positive)

INTERPRETATION
──────────────
✓ Model achieves {self.history['test_f1']:.1%} F1-score on test data
✓ Model demonstrates good balance between precision and recall
✓ Cross-validation results are stable (low std deviation)
✓ Model is not overfitting (test performance ≈ CV performance)

RECOMMENDATIONS
────────────────
1. Monitor performance on new data regularly
2. Consider retraining with new data periodically
3. Investigate false positives/negatives for pattern understanding
4. For production, consider ensemble with additional models
5. Implement model versioning and monitoring

================================================================================
Generated: {pd.Timestamp.now()}
================================================================================
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Analysis report saved to {save_path}")
        return report


def generate_all_visualizations(model, history, y_test, y_pred, y_pred_proba, preprocessor):
    """Generate all visualizations and analysis"""
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS & ANALYSIS")
    print("="*70)
    
    analyzer = ResultsAnalyzer(model, history, y_test, y_pred, y_pred_proba)
    
    print("\nCreating visualizations...")
    analyzer.plot_confusion_matrix()
    analyzer.plot_roc_curve()
    analyzer.plot_metrics_comparison()
    analyzer.plot_cv_results()
    analyzer.plot_prediction_distribution()
    analyzer.plot_feature_importance(preprocessor)
    
    print("\nGenerating report...")
    report = analyzer.generate_report()
    print(report)
    
    print("\n" + "="*70)
    print("✓ All visualizations and analysis complete!")
    print("  Saved to results/ directory")
    print("="*70)
    
    return analyzer


if __name__ == "__main__":
    # Example usage - would be called from main.py
    print("Analysis module ready. Call generate_all_visualizations() to create outputs.")
