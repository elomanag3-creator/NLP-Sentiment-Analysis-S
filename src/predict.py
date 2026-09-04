"""
Prediction Module
Make sentiment predictions on new reviews
"""

import pickle
import os
import pandas as pd
from preprocess import TextPreprocessor


class SentimentPredictor:
    """Makes predictions on new reviews"""
    
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize predictor with trained model and preprocessor
        
        Parameters:
        - model_path: path to trained model
        - preprocessor_path: path to fitted preprocessor
        """
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Loading preprocessor from {preprocessor_path}...")
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        print("✓ Model and preprocessor loaded successfully!")
    
    def predict_single(self, review_text):
        """
        Predict sentiment for single review
        
        Returns:
        - sentiment: 'Positive' or 'Negative'
        - confidence: probability score (0-1)
        """
        # Clean text
        clean_text = self.preprocessor.clean_text(review_text)
        
        # Vectorize
        X = self.preprocessor.transform([clean_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        sentiment = "Positive ✓" if prediction == 1 else "Negative ✗"
        confidence = max(proba)
        
        return {
            'review': review_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probability': {
                'negative': proba[0],
                'positive': proba[1]
            }
        }
    
    def predict_batch(self, reviews_list):
        """
        Predict sentiment for multiple reviews
        
        Parameters:
        - reviews_list: list of review texts
        
        Returns:
        - list of predictions
        """
        results = []
        for review in reviews_list:
            result = self.predict_single(review)
            results.append(result)
        return results
    
    def predict_from_file(self, filepath):
        """
        Predict sentiment from CSV file
        
        Expected format: CSV with 'review_text' column
        """
        print(f"\nLoading reviews from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Processing {len(df)} reviews...")
        
        predictions = []
        sentiments = []
        confidences = []
        
        for idx, row in df.iterrows():
            result = self.predict_single(row['review_text'])
            predictions.append(result)
            sentiments.append(1 if "Positive" in result['sentiment'] else 0)
            confidences.append(result['confidence'])
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'review_text': [p['review'] for p in predictions],
            'sentiment': [p['sentiment'] for p in predictions],
            'confidence': [p['confidence'] for p in predictions],
            'prob_negative': [p['probability']['negative'] for p in predictions],
            'prob_positive': [p['probability']['positive'] for p in predictions]
        })
        
        return results_df
    
    def analyze_predictions(self, y_true, y_pred):
        """Analyze prediction results"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


def run_prediction_demo():
    """Demo: Run predictions on sample reviews"""
    
    print("="*70)
    print("SENTIMENT PREDICTION DEMO")
    print("="*70)
    
    # Initialize predictor
    predictor = SentimentPredictor(
        model_path='models/sentiment_model.pkl',
        preprocessor_path='data/preprocessor.pkl'
    )
    
    # Sample reviews for testing
    test_reviews = [
        "This product is absolutely fantastic! Best quality ever!",
        "Terrible quality, completely broken, worst purchase ever.",
        "Great value for money, very satisfied with this purchase.",
        "Horrible and disappointing, do not recommend at all.",
        "Amazing product! Exceeded all my expectations greatly.",
        "Poor quality materials, fell apart after one day.",
        "Excellent customer service and fast shipping!",
        "Defective and unusable, total waste of money.",
        "Love it! This is exactly what I was looking for.",
        "Awful product, regret buying this immediately."
    ]
    
    print("\n" + "="*70)
    print("SINGLE REVIEW PREDICTIONS")
    print("="*70)
    
    for idx, review in enumerate(test_reviews[:5], 1):
        print(f"\n[{idx}] Review: '{review}'")
        
        result = predictor.predict_single(review)
        
        print(f"    Sentiment: {result['sentiment']}")
        print(f"    Confidence: {result['confidence']:.2%}")
        print(f"    Probabilities:")
        print(f"      - Negative: {result['probability']['negative']:.4f}")
        print(f"      - Positive: {result['probability']['positive']:.4f}")
    
    print("\n" + "="*70)
    print("BATCH PREDICTION")
    print("="*70)
    
    print(f"\nPredicting sentiment for {len(test_reviews)} reviews...")
    batch_results = predictor.predict_batch(test_reviews)
    
    # Summary
    positive_count = sum(1 for r in batch_results if "Positive" in r['sentiment'])
    negative_count = len(batch_results) - positive_count
    avg_confidence = sum(r['confidence'] for r in batch_results) / len(batch_results)
    
    print(f"\nResults Summary:")
    print(f"  Total reviews: {len(batch_results)}")
    print(f"  Positive: {positive_count} ({positive_count/len(batch_results)*100:.1f}%)")
    print(f"  Negative: {negative_count} ({negative_count/len(batch_results)*100:.1f}%)")
    print(f"  Average confidence: {avg_confidence:.2%}")
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {
            'review': r['review'][:50] + '...' if len(r['review']) > 50 else r['review'],
            'sentiment': r['sentiment'],
            'confidence': f"{r['confidence']:.2%}"
        }
        for r in batch_results
    ])
    
    print("\nDetailed Results:")
    print(results_df.to_string(index=False))
    
    return predictor


if __name__ == "__main__":
    predictor = run_prediction_demo()
