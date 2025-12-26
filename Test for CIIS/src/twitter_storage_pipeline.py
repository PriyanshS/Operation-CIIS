import sqlite3
import json
import csv
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
import os
from pathlib import Path

class TwitterDataStorage:
    """
    Storage pipeline for Twitter scraping project
    Handles SQLite database operations and data export for NLP analysis
    """
    
    def __init__(self, db_path: str = "twitter_data.db"):
        """
        Initialize the storage pipeline
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.setup_logging()
        self.init_database()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('storage_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Main tweets table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tweets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id TEXT UNIQUE NOT NULL,
                        text TEXT NOT NULL,
                        cleaned_text TEXT,
                        timestamp DATETIME,
                        user_id TEXT,
                        username TEXT,
                        retweet_count INTEGER DEFAULT 0,
                        like_count INTEGER DEFAULT 0,
                        reply_count INTEGER DEFAULT 0,
                        quote_count INTEGER DEFAULT 0,
                        keyword_matched TEXT,
                        hashtags TEXT,
                        urls TEXT,
                        is_retweet BOOLEAN DEFAULT 0,
                        language TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Keywords tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS keywords (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        keyword TEXT UNIQUE NOT NULL,
                        category TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Scraping sessions table for tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS scraping_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        start_time DATETIME,
                        end_time DATETIME,
                        total_tweets INTEGER DEFAULT 0,
                        keywords_used TEXT,
                        status TEXT DEFAULT 'running'
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweet_id ON tweets(tweet_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON tweets(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_keyword ON tweets(keyword_matched)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON tweets(user_id)')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
    
    def save_tweets(self, tweets_data: List[Dict]) -> int:
        """
        Save tweets to database with duplicate handling
        
        Args:
            tweets_data: List of tweet dictionaries
            
        Returns:
            Number of tweets successfully saved
        """
        saved_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for tweet in tweets_data:
                    try:
                        # Check for duplicates
                        cursor.execute('SELECT id FROM tweets WHERE tweet_id = ?', (tweet.get('tweet_id'),))
                        if cursor.fetchone():
                            self.logger.debug(f"Duplicate tweet found: {tweet.get('tweet_id')}")
                            continue
                        
                        # Insert new tweet
                        cursor.execute('''
                            INSERT INTO tweets (
                                tweet_id, text, cleaned_text, timestamp, user_id, username,
                                retweet_count, like_count, reply_count, quote_count,
                                keyword_matched, hashtags, urls, is_retweet, language
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            tweet.get('tweet_id'),
                            tweet.get('text'),
                            tweet.get('cleaned_text'),
                            tweet.get('timestamp'),
                            tweet.get('user_id'),
                            tweet.get('username'),
                            tweet.get('retweet_count', 0),
                            tweet.get('like_count', 0),
                            tweet.get('reply_count', 0),
                            tweet.get('quote_count', 0),
                            tweet.get('keyword_matched'),
                            json.dumps(tweet.get('hashtags', [])),
                            json.dumps(tweet.get('urls', [])),
                            tweet.get('is_retweet', False),
                            tweet.get('language', 'en')
                        ))
                        saved_count += 1
                        
                    except sqlite3.Error as e:
                        self.logger.error(f"Error saving tweet {tweet.get('tweet_id')}: {e}")
                        continue
                
                conn.commit()
                self.logger.info(f"Successfully saved {saved_count} tweets to database")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        
        return saved_count
    
    def update_cleaned_text(self, tweet_id: str, cleaned_text: str) -> bool:
        """
        Update cleaned text for a specific tweet (for Tanay's cleaning module)
        
        Args:
            tweet_id: Twitter post ID
            cleaned_text: Processed/cleaned text
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE tweets 
                    SET cleaned_text = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE tweet_id = ?
                ''', (cleaned_text, tweet_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.debug(f"Updated cleaned text for tweet: {tweet_id}")
                    return True
                else:
                    self.logger.warning(f"Tweet not found for update: {tweet_id}")
                    return False
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error updating cleaned text: {e}")
            return False
    
    def load_tweets(self, filters: Optional[Dict] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Load tweets from database with optional filters
        
        Args:
            filters: Dictionary of filters (keyword, date_range, user_id, etc.)
            limit: Maximum number of tweets to return
            
        Returns:
            List of tweet dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                # Build query based on filters
                query = "SELECT * FROM tweets WHERE 1=1"
                params = []
                
                if filters:
                    if 'keyword' in filters:
                        query += " AND keyword_matched LIKE ?"
                        params.append(f"%{filters['keyword']}%")
                    
                    if 'date_from' in filters:
                        query += " AND timestamp >= ?"
                        params.append(filters['date_from'])
                    
                    if 'date_to' in filters:
                        query += " AND timestamp <= ?"
                        params.append(filters['date_to'])
                    
                    if 'user_id' in filters:
                        query += " AND user_id = ?"
                        params.append(filters['user_id'])
                    
                    if 'has_cleaned_text' in filters and filters['has_cleaned_text']:
                        query += " AND cleaned_text IS NOT NULL AND cleaned_text != ''"
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                tweets = []
                for row in rows:
                    tweet_dict = dict(row)
                    # Parse JSON fields
                    if tweet_dict['hashtags']:
                        tweet_dict['hashtags'] = json.loads(tweet_dict['hashtags'])
                    if tweet_dict['urls']:
                        tweet_dict['urls'] = json.loads(tweet_dict['urls'])
                    tweets.append(tweet_dict)
                
                self.logger.info(f"Loaded {len(tweets)} tweets from database")
                return tweets
                
        except sqlite3.Error as e:
            self.logger.error(f"Error loading tweets: {e}")
            return []
    
    def export_for_nlp(self, output_format: str = 'csv', output_path: str = None, 
                      cleaned_only: bool = True) -> str:
        """
        Export data specifically formatted for NLP analysis
        
        Args:
            output_format: 'csv', 'json', or 'parquet'
            output_path: Custom output path (optional)
            cleaned_only: Only export tweets with cleaned text
            
        Returns:
            Path to exported file
        """
        # Set up filters for NLP-ready data
        filters = {}
        if cleaned_only:
            filters['has_cleaned_text'] = True
        
        tweets = self.load_tweets(filters=filters)
        
        if not tweets:
            self.logger.warning("No tweets found for NLP export")
            return None
        
        # Prepare data for NLP analysis
        nlp_data = []
        for tweet in tweets:
            nlp_record = {
                'tweet_id': tweet['tweet_id'],
                'original_text': tweet['text'],
                'cleaned_text': tweet['cleaned_text'],
                'timestamp': tweet['timestamp'],
                'user_id': tweet['user_id'],
                'keyword_matched': tweet['keyword_matched'],
                'retweet_count': tweet['retweet_count'],
                'like_count': tweet['like_count'],
                'is_retweet': tweet['is_retweet'],
                'language': tweet['language']
            }
            nlp_data.append(nlp_record)
        
        # Generate output filename if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"nlp_ready_data_{timestamp}.{output_format}"
        
        # Export based on format
        try:
            if output_format.lower() == 'csv':
                df = pd.DataFrame(nlp_data)
                df.to_csv(output_path, index=False, encoding='utf-8')
                
            elif output_format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(nlp_data, f, indent=2, ensure_ascii=False)
                    
            elif output_format.lower() == 'parquet':
                df = pd.DataFrame(nlp_data)
                df.to_parquet(output_path, index=False)
                
            else:
                raise ValueError(f"Unsupported format: {output_format}")
            
            self.logger.info(f"Exported {len(nlp_data)} tweets to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            raise
    
    def get_dataset_stats(self) -> Dict:
        """Get comprehensive statistics about the stored dataset"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Basic counts
                cursor.execute('SELECT COUNT(*) FROM tweets')
                stats['total_tweets'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM tweets WHERE cleaned_text IS NOT NULL AND cleaned_text != ""')
                stats['cleaned_tweets'] = cursor.fetchone()[0]
                
                # Keyword distribution
                cursor.execute('''
                    SELECT keyword_matched, COUNT(*) as count 
                    FROM tweets 
                    GROUP BY keyword_matched 
                    ORDER BY count DESC
                ''')
                stats['keyword_distribution'] = dict(cursor.fetchall())
                
                # Date range
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM tweets')
                date_range = cursor.fetchone()
                stats['date_range'] = {
                    'earliest': date_range[0],
                    'latest': date_range[1]
                }
                
                # Language distribution
                cursor.execute('''
                    SELECT language, COUNT(*) as count 
                    FROM tweets 
                    GROUP BY language 
                    ORDER BY count DESC
                ''')
                stats['language_distribution'] = dict(cursor.fetchall())
                
                # Engagement stats
                cursor.execute('''
                    SELECT 
                        AVG(retweet_count) as avg_retweets,
                        AVG(like_count) as avg_likes,
                        MAX(retweet_count) as max_retweets,
                        MAX(like_count) as max_likes
                    FROM tweets
                ''')
                engagement = cursor.fetchone()
                stats['engagement'] = {
                    'avg_retweets': round(engagement[0] or 0, 2),
                    'avg_likes': round(engagement[1] or 0, 2),
                    'max_retweets': engagement[2] or 0,
                    'max_likes': engagement[3] or 0
                }
                
                return stats
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting dataset stats: {e}")
            return {}
    
    def save_keywords(self, keywords: List[Dict]):
        """
        Save keyword list from Srishti's curation work
        
        Args:
            keywords: List of keyword dictionaries with 'keyword' and 'category'
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for kw in keywords:
                    cursor.execute('''
                        INSERT OR REPLACE INTO keywords (keyword, category)
                        VALUES (?, ?)
                    ''', (kw['keyword'], kw.get('category', 'uncategorized')))
                
                conn.commit()
                self.logger.info(f"Saved {len(keywords)} keywords to database")
                
        except sqlite3.Error as e:
            self.logger.error(f"Error saving keywords: {e}")
            raise
    
    def create_analysis_dataset(self, output_dir: str = "analysis_ready") -> Dict[str, str]:
        """
        Create analysis-ready datasets for the NLP model (Step 2)
        Exports multiple formats and creates subsets for different analysis needs
        
        Args:
            output_dir: Directory to save analysis files
            
        Returns:
            Dictionary mapping dataset type to file path
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Full cleaned dataset
            full_data_path = os.path.join(output_dir, f"full_dataset_{timestamp}.csv")
            tweets = self.load_tweets(filters={'has_cleaned_text': True})
            
            if tweets:
                df = pd.DataFrame(tweets)
                # Select columns relevant for NLP
                nlp_columns = [
                    'tweet_id', 'cleaned_text', 'timestamp', 'keyword_matched',
                    'retweet_count', 'like_count', 'is_retweet', 'language', 'user_id'
                ]
                df[nlp_columns].to_csv(full_data_path, index=False, encoding='utf-8')
                exported_files['full_dataset'] = full_data_path
                
                # 2. High-engagement subset (for priority analysis)
                high_engagement = df[
                    (df['retweet_count'] > df['retweet_count'].quantile(0.75)) |
                    (df['like_count'] > df['like_count'].quantile(0.75))
                ]
                if not high_engagement.empty:
                    high_eng_path = os.path.join(output_dir, f"high_engagement_{timestamp}.csv")
                    high_engagement[nlp_columns].to_csv(high_eng_path, index=False, encoding='utf-8')
                    exported_files['high_engagement'] = high_eng_path
                
                # 3. Category-based subsets
                categories = df['keyword_matched'].value_counts()
                for category in categories.index[:5]:  # Top 5 categories
                    category_data = df[df['keyword_matched'] == category]
                    if len(category_data) >= 10:  # Only if sufficient data
                        safe_category = "".join(c for c in category if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        category_path = os.path.join(output_dir, f"category_{safe_category}_{timestamp}.csv")
                        category_data[nlp_columns].to_csv(category_path, index=False, encoding='utf-8')
                        exported_files[f'category_{safe_category}'] = category_path
                
                # 4. JSON format for NLP libraries that prefer it
                json_path = os.path.join(output_dir, f"nlp_dataset_{timestamp}.json")
                nlp_json_data = df[nlp_columns].to_dict('records')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(nlp_json_data, f, indent=2, ensure_ascii=False, default=str)
                exported_files['json_format'] = json_path
            
            # 5. Create metadata file
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'total_tweets': len(tweets),
                'files_created': exported_files,
                'dataset_stats': self.get_dataset_stats()
            }
            
            metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            exported_files['metadata'] = metadata_path
            
            self.logger.info(f"Created analysis datasets: {list(exported_files.keys())}")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error creating analysis dataset: {e}")
            raise
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create a backup of the database"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"twitter_data_backup_{timestamp}.db"
        
        try:
            # Simple file copy for SQLite
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            raise
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old tweets to manage database size
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of tweets deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute('''
                    DELETE FROM tweets 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                # Vacuum to reclaim space
                cursor.execute('VACUUM')
                
                self.logger.info(f"Cleaned up {deleted_count} old tweets")
                return deleted_count
                
        except sqlite3.Error as e:
            self.logger.error(f"Cleanup error: {e}")
            return 0


class PipelineIntegration:
    """
    Integration script that connects all team components
    """
    
    def __init__(self, storage: TwitterDataStorage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
    
    def process_scraped_data(self, raw_tweets: List[Dict], session_id: str) -> Dict:
        """
        Process raw tweets from Jayendra's scraper through the pipeline
        
        Args:
            raw_tweets: Raw tweet data from scraper
            session_id: Unique session identifier
            
        Returns:
            Processing summary
        """
        self.logger.info(f"Starting pipeline processing for session: {session_id}")
        
        # Save raw tweets
        saved_count = self.storage.save_tweets(raw_tweets)
        
        # Log session
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO scraping_sessions (session_id, start_time, total_tweets, status)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, datetime.now(), saved_count, 'completed'))
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Error logging session: {e}")
        
        return {
            'session_id': session_id,
            'tweets_processed': len(raw_tweets),
            'tweets_saved': saved_count,
            'duplicates_filtered': len(raw_tweets) - saved_count,
            'status': 'success'
        }
    
    def prepare_for_cleaning(self, batch_size: int = 1000) -> List[Dict]:
        """
        Prepare data for Tanay's cleaning module
        Returns tweets that need cleaning in batches
        
        Args:
            batch_size: Number of tweets per batch
            
        Returns:
            List of tweets needing cleaning
        """
        filters = {'has_cleaned_text': False}  # Only uncleaned tweets
        tweets = self.storage.load_tweets(filters=filters, limit=batch_size)
        
        self.logger.info(f"Prepared {len(tweets)} tweets for cleaning")
        return tweets
    
    def generate_pipeline_report(self) -> Dict:
        """Generate comprehensive pipeline status report"""
        stats = self.storage.get_dataset_stats()
        
        report = {
            'pipeline_status': 'operational',
            'last_updated': datetime.now().isoformat(),
            'data_summary': stats,
            'cleaning_progress': {
                'total_tweets': stats.get('total_tweets', 0),
                'cleaned_tweets': stats.get('cleaned_tweets', 0),
                'cleaning_percentage': round(
                    (stats.get('cleaned_tweets', 0) / max(stats.get('total_tweets', 1), 1)) * 100, 2
                )
            },
            'ready_for_nlp': stats.get('cleaned_tweets', 0) > 0
        }
        
        return report


# Example usage and testing functions
def main():
    """Main function demonstrating the storage pipeline"""
    
    # Initialize storage
    storage = TwitterDataStorage("twitter_anti_india_data.db")
    pipeline = PipelineIntegration(storage)
    
    # Example: Save sample keywords (normally from Srishti)
    sample_keywords = [
        {'keyword': '#anti-india', 'category': 'political'},
        {'keyword': 'india propaganda', 'category': 'propaganda'},
        {'keyword': '#kashmir', 'category': 'political'},
        {'keyword': 'indian bot', 'category': 'bot-like'}
    ]
    storage.save_keywords(sample_keywords)
    
    # Example: Process sample tweet data (normally from Jayendra)
    sample_tweets = [
        {
            'tweet_id': '1234567890',
            'text': 'Sample anti-India tweet content #anti-india',
            'timestamp': '2024-01-15 10:30:00',
            'user_id': 'user123',
            'username': 'testuser',
            'retweet_count': 25,
            'like_count': 45,
            'keyword_matched': '#anti-india',
            'hashtags': ['anti-india'],
            'urls': [],
            'is_retweet': False,
            'language': 'en'
        }
    ]
    
    # Process through pipeline
    result = pipeline.process_scraped_data(sample_tweets, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print("Processing result:", result)
    
    # Generate report
    report = pipeline.generate_pipeline_report()
    print("Pipeline report:", json.dumps(report, indent=2, default=str))
    
    # Prepare data for NLP (Step 2)
    # This would be called after Tanay's cleaning is complete
    # analysis_files = storage.create_analysis_dataset()
    # print("Analysis files created:", analysis_files)


if __name__ == "__main__":
    main()