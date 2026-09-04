"""
Data Preprocessing Module
Handles text cleaning, tokenization, and vectorization
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Handles all text preprocessing operations"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning:
        1. Lowercase
        2. Remove special characters and numbers
        3. Tokenize
        4. Remove stopwords
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Join back to string
        return ' '.join(tokens)
    
    def fit_vectorizer(self, texts, max_features=100):
        """
        Fit TF-IDF vectorizer on the texts
        
        Parameters:
        - texts: list of text documents
        - max_features: maximum number of features to keep
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,            # Minimum document frequency
            max_df=0.8,          # Maximum document frequency
            sublinear_tf=True    # Apply sublinear TF scaling
        )
        
        vectors = self.vectorizer.fit_transform(texts)
        return vectors
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        return self.vectorizer.get_feature_names_out()


def load_data(filepath):
    """Load CSV data"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} reviews")
    return df


def analyze_data(df):
    """Analyze dataset statistics"""
    print("\n" + "="*70)
    print("DATA ANALYSIS")
    print("="*70)
    
    print(f"Total reviews: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Class distribution
    print("\nClass Distribution:")
    class_dist = df['sentiment'].value_counts()
    for class_label, count in class_dist.items():
        percentage = (count / len(df)) * 100
        sentiment_name = "POSITIVE" if class_label == 1 else "NEGATIVE"
        print(f"  {sentiment_name}: {count} reviews ({percentage:.1f}%)")
    
    # Text statistics
    df['text_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()
    
    print(f"\nText Statistics:")
    print(f"  Average review length: {df['text_length'].mean():.0f} characters")
    print(f"  Average word count: {df['word_count'].mean():.1f} words")
    print(f"  Min/Max length: {df['text_length'].min()}/{df['text_length'].max()} chars")
    
    # Sample reviews
    print("\nSample Reviews:")
    for idx, row in df.head(3).iterrows():
        sentiment_name = "POSITIVE ✓" if row['sentiment'] == 1 else "NEGATIVE ✗"
        print(f"\n  {sentiment_name}: {row['review_text'][:80]}...")


def preprocess_pipeline(input_path, output_dir='data', test_size=0.2, max_features=100):
    """
    Complete preprocessing pipeline
    
    Returns:
    - X_train, X_test, y_train, y_test: feature matrices and labels
    - preprocessor: fitted TextPreprocessor object
    """
    
    # Load data
    df = load_data(input_path)
    
    # Analyze data
    analyze_data(df)
    
    # Initialize preprocessor
    print("\n" + "="*70)
    print("TEXT PREPROCESSING")
    print("="*70)
    
    preprocessor = TextPreprocessor()
    
    # Clean texts
    print("Cleaning text data...")
    df['review_clean'] = df['review_text'].apply(preprocessor.clean_text)
    print("✓ Text cleaning complete")
    
    # Show sample
    print("\nSample (before → after):")
    print(f"  Before: {df['review_text'].iloc[0][:70]}...")
    print(f"  After:  {df['review_clean'].iloc[0][:70]}...")
    
    # Vectorize texts
    print("\n" + "="*70)
    print("FEATURE ENGINEERING (TF-IDF)")
    print("="*70)
    
    print(f"Vectorizing with {max_features} features (unigrams + bigrams)...")
    X = preprocessor.fit_vectorizer(df['review_clean'], max_features=max_features)
    y = df['sentiment'].values
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"  ({X.shape[0]} documents, {X.shape[1]} features)")
    
    # Show top features
    feature_names = preprocessor.get_feature_names()
    print(f"\nSample features (words/bigrams): {', '.join(feature_names[:15])}")
    
    # Split data
    print("\n" + "="*70)
    print("TRAIN-TEST SPLIT")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    # Save processed data
    print("\n" + "="*70)
    print("SAVING PROCESSED DATA")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle for later use
    with open(os.path.join(output_dir, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(output_dir, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(output_dir, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(output_dir, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    with open(os.path.join(output_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("✓ Saved X_train, X_test, y_train, y_test, preprocessor")
    print(f"  Location: {output_dir}/")
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Run preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        input_path='data/reviews.csv',
        output_dir='data',
        test_size=0.2,
        max_features=100
    )
