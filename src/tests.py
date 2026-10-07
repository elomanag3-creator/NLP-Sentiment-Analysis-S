"""
Unit Tests
Tests for NLP sentiment analysis project
"""

import unittest
import numpy as np
import pandas as pd
import os
import pickle
import tempfile
from unittest.mock import MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocess import TextPreprocessor, load_data
from train import SentimentModel


class TestTextPreprocessor(unittest.TestCase):
    """Tests for TextPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text_lowercase(self):
        """Test that text is converted to lowercase"""
        text = "Hello World"
        cleaned = self.preprocessor.clean_text(text)
        self.assertTrue(cleaned.islower() or len(cleaned) == 0)
    
    def test_clean_text_removes_special_chars(self):
        """Test that special characters are removed"""
        text = "Hello! How are you? @#$%"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("#", cleaned)
    
    def test_clean_text_empty_string(self):
        """Test handling of empty string"""
        text = ""
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "")
    
    def test_clean_text_none_input(self):
        """Test handling of None input"""
        text = None
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "")
    
    def test_clean_text_numbers_removed(self):
        """Test that numbers are removed"""
        text = "Product 123 is great"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("123", cleaned)
    
    def test_clean_text_returns_string(self):
        """Test that output is a string"""
        text = "This is a test"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIsInstance(cleaned, str)
    
    def test_vectorizer_fit(self):
        """Test vectorizer fitting"""
        texts = [
            "This is positive",
            "This is negative",
            "Great product",
            "Bad quality"
        ]
        vectors = self.preprocessor.fit_vectorizer(texts, max_features=50)
        
        # Check shape
        self.assertEqual(vectors.shape[0], len(texts))
        self.assertLessEqual(vectors.shape[1], 50)
    
    def test_vectorizer_transform(self):
        """Test vectorizer transform"""
        texts = [
            "This is positive",
            "This is negative",
            "Great product",
            "Bad quality"
        ]
        self.preprocessor.fit_vectorizer(texts, max_features=50)
        
        new_texts = ["Great stuff"]
        vectors = self.preprocessor.transform(new_texts)
        
        self.assertEqual(vectors.shape[0], 1)
        self.assertGreater(vectors.shape[1], 0)
    
    def test_vectorizer_not_fitted(self):
        """Test error when transform called before fit"""
        texts = ["Test text"]
        
        with self.assertRaises(ValueError):
            self.preprocessor.transform(texts)


class TestSentimentModel(unittest.TestCase):
    """Tests for SentimentModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SentimentModel()
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model)
        self.assertIsNone(self.model.model)
        self.assertIsInstance(self.model.history, dict)
    
    def test_build_ensemble(self):
        """Test ensemble building"""
        self.model.build_ensemble()
        
        self.assertIsNotNone(self.model.model)
        self.assertEqual(len(self.model.model.named_estimators), 2)
    
    def test_model_training(self):
        """Test model training"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create dummy data
        texts = [
            "good great excellent",
            "bad terrible awful",
            "amazing wonderful",
            "horrible bad"
        ]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        y = np.array([1, 0, 1, 0])
        
        self.model.build_ensemble()
        self.model.train(X, y)
        
        # Model should be trained
        self.assertIsNotNone(self.model.model)
    
    def test_model_prediction(self):
        """Test model prediction"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create dummy data
        texts = [
            "good great excellent",
            "bad terrible awful",
            "amazing wonderful",
            "horrible bad"
        ]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        y = np.array([1, 0, 1, 0])
        
        self.model.build_ensemble()
        self.model.train(X, y)
        
        predictions = self.model.predict(X)
        
        # Should return predictions
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_model_predict_proba(self):
        """Test probability predictions"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create dummy data
        texts = [
            "good great excellent",
            "bad terrible awful",
            "amazing wonderful",
            "horrible bad"
        ]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        y = np.array([1, 0, 1, 0])
        
        self.model.build_ensemble()
        self.model.train(X, y)
        
        proba = self.model.predict_proba(X)
        
        # Should return probabilities
        self.assertEqual(proba.shape[0], len(y))
        self.assertEqual(proba.shape[1], 2)  # Binary classification
        
        # Check probabilities are between 0 and 1
        self.assertTrue(np.all(proba >= 0))
        self.assertTrue(np.all(proba <= 1))
        
        # Check probabilities sum to 1
        self.assertTrue(np.allclose(proba.sum(axis=1), 1))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create and train model
        texts = [
            "good great excellent",
            "bad terrible awful",
            "amazing wonderful",
            "horrible bad"
        ]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        y = np.array([1, 0, 1, 0])
        
        self.model.build_ensemble()
        self.model.train(X, y)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            self.model.save(model_path)
            
            # Check file exists
            self.assertTrue(os.path.exists(model_path))
            
            # Load and check
            loaded_model = SentimentModel()
            loaded_model.load(model_path)
            
            self.assertIsNotNone(loaded_model.model)


class TestDataLoading(unittest.TestCase):
    """Tests for data loading functions"""
    
    def test_load_data(self):
        """Test data loading from CSV"""
        # Create temporary CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_data.csv')
            
            # Create sample data
            df = pd.DataFrame({
                'review_text': ['Good product', 'Bad product'],
                'sentiment': [1, 0]
            })
            df.to_csv(csv_path, index=False)
            
            # Load data
            loaded_df = load_data(csv_path)
            
            self.assertEqual(len(loaded_df), 2)
            self.assertIn('review_text', loaded_df.columns)
            self.assertIn('sentiment', loaded_df.columns)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def test_preprocessing_to_training_pipeline(self):
        """Test complete preprocessing and training pipeline"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create sample data
        texts = [
            "This product is absolutely amazing great excellent",
            "This product is terrible horrible bad awful",
            "Fantastic wonderful excellent quality",
            "Worst purchase ever completely terrible",
            "Love this product so much",
            "Terrible quality does not work"
        ]
        y = np.array([1, 0, 1, 0, 1, 0])
        
        # Preprocess
        preprocessor = TextPreprocessor()
        cleaned_texts = [preprocessor.clean_text(t) for t in texts]
        X = preprocessor.fit_vectorizer(cleaned_texts, max_features=50)
        
        # Train
        model = SentimentModel()
        model.build_ensemble()
        model.train(X, y)
        
        # Predict
        predictions = model.predict(X)
        
        # Check results
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(all(p in [0, 1] for p in predictions))


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
