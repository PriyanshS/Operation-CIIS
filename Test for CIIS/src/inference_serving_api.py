import os
import json
import pickle
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
import numpy as np

# FastAPI and Pydantic imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ML imports (adjust based on your actual model type)
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    logging.warning(f"Some ML libraries not available: {e}")

# Import storage from Step 1
from src.twitter_storage_pipeline import TwitterDataStorage, PipelineIntegration


class TweetInferenceEngine:
    """
    Main inference engine for classifying tweets as anti-India narratives
    """
    
    def __init__(self, model_path: str, model_type: str = "sklearn"):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model file
            model_type: Type of model ('sklearn', 'transformers', 'custom')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.label_encoder = None
        self.confidence_threshold = 0.7
        
        self.setup_logging()
        self.load_model()
    
    def setup_logging(self):
        """Setup logging for inference engine"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("InferenceEngine")
    
    def load_model(self):
        """Load the trained model and associated components"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
            if self.model_type == "sklearn":
                self._load_sklearn_model()
            elif self.model_type == "transformers":
                self._load_transformers_model()
            elif self.model_type == "custom":
                self._load_custom_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.logger.info(f"Model loaded successfully: {self.model_type}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _load_sklearn_model(self):
        """Load scikit-learn model pipeline"""
        # Load complete pipeline (includes vectorizer + classifier)
        if self.model_path.endswith('.pkl'):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = joblib.load(self.model_path)
        
        # If separate components, load them
        model_dir = os.path.dirname(self.model_path)
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
    
    def _load_transformers_model(self):
        """Load Hugging Face transformers model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            self.logger.error(f"Error loading transformers model: {e}")
            raise
    
    def _load_custom_model(self):
        """Load custom model format"""
        # Implement based on your specific model format
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.vectorizer = model_data.get('vectorizer')
            self.label_encoder = model_data.get('label_encoder')
    
    def classify_tweet(self, text: str) -> Dict[str, Any]:
        """
        Classify a single tweet
        
        Args:
            text: Tweet text to classify
            
        Returns:
            Classification result with confidence scores
        """
        try:
            if not text or not text.strip():
                return {
                    'label': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {},
                    'error': 'Empty text provided'
                }
            
            if self.model_type == "sklearn":
                return self._classify_sklearn(text)
            elif self.model_type == "transformers":
                return self._classify_transformers(text)
            elif self.model_type == "custom":
                return self._classify_custom(text)
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return {
                'label': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'error': str(e)
            }
    
    def _classify_sklearn(self, text: str) -> Dict[str, Any]:
        """Classify using scikit-learn model"""
        # Get prediction and probabilities
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        
        # Get class labels
        classes = self.model.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        
        max_confidence = float(np.max(probabilities))
        
        return {
            'label': str(prediction),
            'confidence': max_confidence,
            'probabilities': prob_dict,
            'is_anti_india': prediction == 'anti-india',
            'needs_review': max_confidence < self.confidence_threshold
        }
    
    def _classify_transformers(self, text: str) -> Dict[str, Any]:
        """Classify using transformers model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Assuming binary classification: [not_anti_india, anti_india]
        confidence = float(torch.max(predictions))
        predicted_class = int(torch.argmax(predictions))
        
        labels = ['neutral', 'anti-india']
        probabilities = {labels[i]: float(predictions[0][i]) for i in range(len(labels))}
        
        return {
            'label': labels[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities,
            'is_anti_india': predicted_class == 1,
            'needs_review': confidence < self.confidence_threshold
        }
    
    def _classify_custom(self, text: str) -> Dict[str, Any]:
        """Classify using custom model"""
        # Vectorize text if needed
        if self.vectorizer:
            text_vector = self.vectorizer.transform([text])
        else:
            text_vector = [text]
        
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0] if hasattr(self.model, 'predict_proba') else [0.5, 0.5]
        
        max_confidence = float(np.max(probabilities))
        
        return {
            'label': str(prediction),
            'confidence': max_confidence,
            'probabilities': {f'class_{i}': float(prob) for i, prob in enumerate(probabilities)},
            'is_anti_india': prediction == 1 or prediction == 'anti-india',
            'needs_review': max_confidence < self.confidence_threshold
        }
    
    def batch_classify(self, tweets: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple tweets efficiently
        
        Args:
            tweets: List of tweet texts
            
        Returns:
            List of classification results
        """
        results = []
        for tweet in tweets:
            result = self.classify_tweet(tweet)
            results.append(result)
        
        self.logger.info(f"Batch classified {len(tweets)} tweets")
        return results


# Pydantic models for API
class TweetInput(BaseModel):
    text: str = Field(..., description="Tweet text to classify")
    tweet_id: Optional[str] = Field(None, description="Optional tweet ID for tracking")
    store_result: bool = Field(False, description="Whether to store classification result")

class BatchTweetInput(BaseModel):
    tweets: List[TweetInput] = Field(..., description="List of tweets to classify")

class ClassificationResult(BaseModel):
    tweet_id: Optional[str]
    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    is_anti_india: bool
    needs_review: bool
    error: Optional[str] = None
    timestamp: str

class BatchClassificationResult(BaseModel):
    results: List[ClassificationResult]
    summary: Dict[str, Any]


class TwitterInferenceAPI:
    """
    FastAPI application for serving tweet classification
    """
    
    def __init__(self, model_path: str, model_type: str = "sklearn", 
                 storage_db_path: str = "twitter_anti_india_data.db"):
        """
        Initialize the API
        
        Args:
            model_path: Path to trained model
            model_type: Type of model
            storage_db_path: Path to storage database
        """
        self.inference_engine = TweetInferenceEngine(model_path, model_type)
        self.storage = TwitterDataStorage(storage_db_path)
        self.app = FastAPI(
            title="Twitter Anti-India Narrative Detection API",
            description="API for classifying tweets as anti-India narratives",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.setup_routes()
        self.logger = logging.getLogger("InferenceAPI")
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Health check endpoint"""
            return {
                "status": "operational",
                "service": "Twitter Anti-India Narrative Detection API",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/classify", response_model=ClassificationResult)
        async def classify_tweet(tweet_input: TweetInput, background_tasks: BackgroundTasks):
            """
            Classify a single tweet
            """
            try:
                # Perform classification
                result = self.inference_engine.classify_tweet(tweet_input.text)
                
                # Create response
                classification_result = ClassificationResult(
                    tweet_id=tweet_input.tweet_id,
                    text=tweet_input.text,
                    label=result['label'],
                    confidence=result['confidence'],
                    probabilities=result['probabilities'],
                    is_anti_india=result['is_anti_india'],
                    needs_review=result['needs_review'],
                    error=result.get('error'),
                    timestamp=datetime.now().isoformat()
                )
                
                # Store result if requested
                if tweet_input.store_result:
                    background_tasks.add_task(
                        self._store_classification_result,
                        tweet_input, result
                    )
                
                return classification_result
                
            except Exception as e:
                self.logger.error(f"Classification API error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/classify/batch", response_model=BatchClassificationResult)
        async def classify_tweets_batch(batch_input: BatchTweetInput, background_tasks: BackgroundTasks):
            """
            Classify multiple tweets in batch
            """
            try:
                results = []
                summary = {
                    'total_tweets': len(batch_input.tweets),
                    'anti_india_count': 0,
                    'needs_review_count': 0,
                    'avg_confidence': 0.0,
                    'processing_time': datetime.now().isoformat()
                }
                
                total_confidence = 0.0
                
                for tweet_input in batch_input.tweets:
                    result = self.inference_engine.classify_tweet(tweet_input.text)
                    
                    classification_result = ClassificationResult(
                        tweet_id=tweet_input.tweet_id,
                        text=tweet_input.text,
                        label=result['label'],
                        confidence=result['confidence'],
                        probabilities=result['probabilities'],
                        is_anti_india=result['is_anti_india'],
                        needs_review=result['needs_review'],
                        error=result.get('error'),
                        timestamp=datetime.now().isoformat()
                    )
                    
                    results.append(classification_result)
                    
                    # Update summary
                    if result['is_anti_india']:
                        summary['anti_india_count'] += 1
                    if result['needs_review']:
                        summary['needs_review_count'] += 1
                    total_confidence += result['confidence']
                    
                    # Store result if requested
                    if tweet_input.store_result:
                        background_tasks.add_task(
                            self._store_classification_result,
                            tweet_input, result
                        )
                
                summary['avg_confidence'] = round(total_confidence / len(batch_input.tweets), 3)
                summary['anti_india_percentage'] = round((summary['anti_india_count'] / summary['total_tweets']) * 100, 2)
                
                return BatchClassificationResult(results=results, summary=summary)
                
            except Exception as e:
                self.logger.error(f"Batch classification API error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/classify/stored/{tweet_id}")
        async def classify_stored_tweet(tweet_id: str):
            """
            Classify a tweet already stored in the database
            """
            try:
                # Load tweet from storage
                tweets = self.storage.load_tweets(filters={'tweet_id': tweet_id}, limit=1)
                
                if not tweets:
                    raise HTTPException(status_code=404, detail="Tweet not found in database")
                
                tweet = tweets[0]
                text_to_classify = tweet.get('cleaned_text') or tweet.get('text')
                
                if not text_to_classify:
                    raise HTTPException(status_code=400, detail="No text available for classification")
                
                # Perform classification
                result = self.inference_engine.classify_tweet(text_to_classify)
                
                # Update database with classification result
                await self._store_classification_in_db(tweet_id, result)
                
                return ClassificationResult(
                    tweet_id=tweet_id,
                    text=text_to_classify,
                    label=result['label'],
                    confidence=result['confidence'],
                    probabilities=result['probabilities'],
                    is_anti_india=result['is_anti_india'],
                    needs_review=result['needs_review'],
                    timestamp=datetime.now().isoformat()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Stored tweet classification error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/dashboard")
        async def get_analytics_dashboard():
            """
            Get analytics dashboard data
            """
            try:
                # Get storage stats
                storage_stats = self.storage.get_dataset_stats()
                
                # Get classification stats from database
                with sqlite3.connect(self.storage.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Classification distribution
                    cursor.execute('''
                        SELECT classification_label, COUNT(*) as count
                        FROM tweet_classifications
                        GROUP BY classification_label
                    ''')
                    classification_dist = dict(cursor.fetchall())
                    
                    # Recent classifications
                    cursor.execute('''
                        SELECT COUNT(*) FROM tweet_classifications
                        WHERE created_at >= datetime('now', '-24 hours')
                    ''')
                    recent_classifications = cursor.fetchone()[0]
                    
                    # High confidence anti-India tweets
                    cursor.execute('''
                        SELECT COUNT(*) FROM tweet_classifications
                        WHERE classification_label = 'anti-india' AND confidence > 0.8
                    ''')
                    high_conf_anti_india = cursor.fetchone()[0]
                
                return {
                    'storage_stats': storage_stats,
                    'classification_distribution': classification_dist,
                    'recent_24h_classifications': recent_classifications,
                    'high_confidence_anti_india': high_conf_anti_india,
                    'model_info': {
                        'type': self.inference_engine.model_type,
                        'confidence_threshold': self.inference_engine.confidence_threshold
                    },
                    'last_updated': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Analytics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analyze/bulk")
        async def analyze_bulk_stored_data(background_tasks: BackgroundTasks, 
                                         limit: int = 1000, 
                                         cleaned_only: bool = True):
            """
            Analyze bulk stored data in background
            """
            try:
                background_tasks.add_task(
                    self._bulk_analyze_stored_tweets, 
                    limit, 
                    cleaned_only
                )
                
                return {
                    'status': 'started',
                    'message': f'Bulk analysis of up to {limit} tweets started in background',
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Bulk analysis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _store_classification_result(self, tweet_input: TweetInput, result: Dict):
        """Store classification result in database"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                
                # Create classifications table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tweet_classifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id TEXT,
                        original_text TEXT,
                        classification_label TEXT,
                        confidence REAL,
                        probabilities TEXT,
                        is_anti_india BOOLEAN,
                        needs_review BOOLEAN,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    INSERT INTO tweet_classifications (
                        tweet_id, original_text, classification_label, confidence,
                        probabilities, is_anti_india, needs_review
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tweet_input.tweet_id,
                    tweet_input.text,
                    result['label'],
                    result['confidence'],
                    json.dumps(result['probabilities']),
                    result['is_anti_india'],
                    result['needs_review']
                ))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Error storing classification: {e}")
    
    async def _store_classification_in_db(self, tweet_id: str, result: Dict):
        """Store classification result for existing tweet"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                
                # Update tweets table with classification
                cursor.execute('''
                    UPDATE tweets 
                    SET 
                        classification_label = ?,
                        classification_confidence = ?,
                        classification_probabilities = ?,
                        is_anti_india = ?,
                        needs_review = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE tweet_id = ?
                ''', (
                    result['label'],
                    result['confidence'],
                    json.dumps(result['probabilities']),
                    result['is_anti_india'],
                    result['needs_review'],
                    tweet_id
                ))
                
                # Also add classification columns to tweets table if they don't exist
                try:
                    cursor.execute('ALTER TABLE tweets ADD COLUMN classification_label TEXT')
                    cursor.execute('ALTER TABLE tweets ADD COLUMN classification_confidence REAL')
                    cursor.execute('ALTER TABLE tweets ADD COLUMN classification_probabilities TEXT')
                    cursor.execute('ALTER TABLE tweets ADD COLUMN is_anti_india BOOLEAN')
                    cursor.execute('ALTER TABLE tweets ADD COLUMN needs_review BOOLEAN')
                except sqlite3.OperationalError:
                    # Columns already exist
                    pass
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Error updating tweet classification: {e}")
    
    async def _bulk_analyze_stored_tweets(self, limit: int, cleaned_only: bool):
        """Background task for bulk analysis of stored tweets"""
        try:
            self.logger.info(f"Starting bulk analysis of {limit} tweets")
            
            # Load tweets that haven't been classified yet
            filters = {}
            if cleaned_only:
                filters['has_cleaned_text'] = True
            
            tweets = self.storage.load_tweets(filters=filters, limit=limit)
            
            classified_count = 0
            anti_india_count = 0
            
            for tweet in tweets:
                text_to_classify = tweet.get('cleaned_text') or tweet.get('text')
                if text_to_classify:
                    result = self.inference_engine.classify_tweet(text_to_classify)
                    await self._store_classification_in_db(tweet['tweet_id'], result)
                    
                    classified_count += 1
                    if result['is_anti_india']:
                        anti_india_count += 1
                    
                    # Log progress every 100 tweets
                    if classified_count % 100 == 0:
                        self.logger.info(f"Bulk analysis progress: {classified_count}/{len(tweets)}")
            
            self.logger.info(f"Bulk analysis completed: {classified_count} tweets classified, {anti_india_count} flagged as anti-India")
            
        except Exception as e:
            self.logger.error(f"Bulk analysis error: {e}")


def create_app(model_path: str, model_type: str = "sklearn", 
               storage_db_path: str = "twitter_anti_india_data.db") -> FastAPI:
    """
    Factory function to create FastAPI app
    
    Args:
        model_path: Path to trained model
        model_type: Type of model ('sklearn', 'transformers', 'custom')
        storage_db_path: Path to storage database
        
    Returns:
        Configured FastAPI application
    """
    api = TwitterInferenceAPI(model_path, model_type, storage_db_path)
    return api.app


# Configuration and deployment
class InferenceConfig:
    """Configuration for inference API"""
    
    MODEL_PATH = os.getenv("MODEL_PATH", "models/anti_india_classifier.pkl")
    MODEL_TYPE = os.getenv("MODEL_TYPE", "sklearn")
    STORAGE_DB_PATH = os.getenv("STORAGE_DB_PATH", "twitter_anti_india_data.db")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def run_server():
    """Run the inference API server"""
    config = InferenceConfig()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
    
    # Create app
    app = create_app(
        model_path=config.MODEL_PATH,
        model_type=config.MODEL_TYPE,
        storage_db_path=config.STORAGE_DB_PATH
    )
    
    # Run server
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower()
    )


# CLI interface for testing and debugging
def test_inference_engine():
    """Test function for inference engine"""
    print("Testing Inference Engine...")
    
    # This assumes you have a trained model
    # Replace with actual model path
    model_path = "models/anti_india_classifier.pkl"
    
    try:
        # Initialize engine
        engine = TweetInferenceEngine(model_path, "sklearn")
        
        # Test tweets
        test_tweets = [
            "India is a great country with rich culture",
            "Anti-India propaganda spreading on social media",
            "Kashmir issue needs international attention #FreKashmir",
            "Normal tweet about cricket match today"
        ]
        
        print("\nClassification Results:")
        print("-" * 50)
        
        for tweet in test_tweets:
            result = engine.classify_tweet(tweet)
            print(f"Tweet: {tweet[:50]}...")
            print(f"Label: {result['label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Anti-India: {result['is_anti_india']}")
            print(f"Needs Review: {result['needs_review']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: This test requires a trained model file. Create a dummy model for testing.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_inference_engine()
        elif sys.argv[1] == "server":
            run_server()
        else:
            print("Usage: python inference.py [test|server]")
    else:
        # Default: run server
        run_server()


# Example usage and integration script
"""
USAGE EXAMPLES:

1. Start the API server:
   python inference.py server

2. Test inference engine:
   python inference.py test

3. Programmatic usage:
   
   # Initialize
   engine = TweetInferenceEngine("models/classifier.pkl", "sklearn")
   storage = TwitterDataStorage("twitter_data.db")
   
   # Classify single tweet
   result = engine.classify_tweet("Sample tweet text")
   
   # Classify stored tweets
   app = create_app("models/classifier.pkl")
   
   # Deploy with: uvicorn inference:app --host 0.0.0.0 --port 8000

API ENDPOINTS:
- GET /: Health check
- POST /classify: Classify single tweet
- POST /classify/batch: Classify multiple tweets
- GET /classify/stored/{tweet_id}: Classify stored tweet
- GET /analytics/dashboard: Get analytics data
- POST /analyze/bulk: Start bulk analysis of stored data

ENVIRONMENT VARIABLES:
- MODEL_PATH: Path to trained model file
- MODEL_TYPE: sklearn|transformers|custom
- STORAGE_DB_PATH: Path to SQLite database
- API_HOST: Server host (default: 0.0.0.0)
- API_PORT: Server port (default: 8000)
- LOG_LEVEL: Logging level (default: INFO)
"""