"""
Flask API for Sentiment Analysis Model
Serve predictions via REST API endpoints
"""

from flask import Flask, request, jsonify
import pickle
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import SentimentPredictor
from config import API_CONFIG, MODEL_PATHS, DATA_PATHS


class APIError(Exception):
    """Custom API error"""
    pass


def create_app():
    """Create and configure Flask app"""
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    
    # Initialize predictor (global)
    try:
        app.predictor = SentimentPredictor(
            model_path=MODEL_PATHS['ensemble_model'],
            preprocessor_path=MODEL_PATHS['preprocessor']
        )
        app.model_loaded = True
    except FileNotFoundError as e:
        print(f"Warning: Model files not found. {e}")
        app.model_loaded = False
    
    return app


# Create app instance
app = create_app()


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': app.model_loaded
    }), 200


@app.route('/api/v1/predict', methods=['POST'])
def predict_single():
    """
    Predict sentiment for single review
    
    Request JSON:
    {
        "review": "This product is amazing!"
    }
    
    Response:
    {
        "review": "This product is amazing!",
        "sentiment": "Positive",
        "confidence": 0.94,
        "probabilities": {
            "negative": 0.06,
            "positive": 0.94
        }
    }
    """
    if not app.model_loaded:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Model files not found or failed to load'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'review' not in data:
            raise APIError('Missing "review" field in request')
        
        review = data['review']
        
        if not isinstance(review, str) or len(review.strip()) == 0:
            raise APIError('Review must be non-empty string')
        
        # Make prediction
        result = app.predictor.predict_single(review)
        
        return jsonify({
            'status': 'success',
            'review': result['review'],
            'sentiment': result['sentiment'].split()[0],  # Just "Positive" or "Negative"
            'confidence': round(result['confidence'], 4),
            'probabilities': {
                'negative': round(result['probability']['negative'], 4),
                'positive': round(result['probability']['positive'], 4)
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except APIError as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/api/v1/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple reviews
    
    Request JSON:
    {
        "reviews": [
            "Great product!",
            "Terrible quality"
        ]
    }
    
    Response:
    {
        "status": "success",
        "total": 2,
        "predictions": [
            {
                "review": "Great product!",
                "sentiment": "Positive",
                "confidence": 0.92
            },
            ...
        ]
    }
    """
    if not app.model_loaded:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'reviews' not in data:
            raise APIError('Missing "reviews" field in request')
        
        reviews = data['reviews']
        
        if not isinstance(reviews, list):
            raise APIError('Reviews must be a list')
        
        if len(reviews) == 0:
            raise APIError('Reviews list cannot be empty')
        
        if len(reviews) > 100:
            raise APIError('Maximum 100 reviews per request')
        
        # Make predictions
        predictions = app.predictor.predict_batch(reviews)
        
        return jsonify({
            'status': 'success',
            'total': len(predictions),
            'predictions': [
                {
                    'review': p['review'],
                    'sentiment': p['sentiment'].split()[0],
                    'confidence': round(p['confidence'], 4)
                }
                for p in predictions
            ],
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except APIError as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Batch prediction failed: {str(e)}'
        }), 500


@app.route('/api/v1/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    info = {
        'status': 'success',
        'model': {
            'name': 'Sentiment Analysis Ensemble',
            'type': 'Binary Classification',
            'components': ['Logistic Regression', 'XGBoost'],
            'input': 'Text review (string)',
            'output': 'Sentiment (Positive/Negative)',
            'loaded': app.model_loaded
        },
        'performance': {
            'f1_score': 0.83,
            'accuracy': 0.82,
            'auc_roc': 0.91
        },
        'api_version': 'v1',
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(info), 200


@app.route('/api/v1/health', methods=['GET'])
def detailed_health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy' if app.model_loaded else 'degraded',
        'model_loaded': app.model_loaded,
        'api_version': 'v1',
        'timestamp': datetime.now().isoformat()
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors"""
    return jsonify({
        'status': 'error',
        'error': 'Bad request',
        'message': str(error)
    }), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'error': 'Not found',
        'message': 'Endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'error': 'Internal server error',
        'message': str(error)
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("SENTIMENT ANALYSIS API")
    print("="*70)
    
    print("\nStarting Flask API server...")
    print(f"Host: {API_CONFIG['host']}")
    print(f"Port: {API_CONFIG['port']}")
    print(f"Debug: {API_CONFIG['debug']}")
    
    print("\nEndpoints:")
    print("  GET  /                          - Health check")
    print("  POST /api/v1/predict            - Predict single review")
    print("  POST /api/v1/predict-batch      - Predict batch reviews")
    print("  GET  /api/v1/model-info         - Model information")
    print("  GET  /api/v1/health             - Detailed health check")
    
    print("\nExample usage:")
    print('  curl -X POST http://localhost:5000/api/v1/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"review": "Great product!"}\'')
    
    print("\n" + "="*70)
    
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )
