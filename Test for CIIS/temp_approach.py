import pandas as pd
import json
import os
import sys
from src.twitter_storage_pipeline import TwitterDataStorage

def load_csv_to_db(csv_file_path: str, db_path: str = "twitter_anti_india_data.db"):
    """
    Reads a CSV file containing tweet data and loads it into the SQLite database.
    """
    print(f"Attempting to load data from {csv_file_path} into {db_path}...")
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read {len(df)} rows from CSV.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    storage = TwitterDataStorage(db_path)
    tweets_to_save = []

    # Assuming CSV columns match the expected tweet dictionary keys
    # Adjust column mapping if your CSV has different headers
    for index, row in df.iterrows():
        try:
            tweet_data = {
                "tweet_id": str(row.get("tweet_id")), # Ensure it's a string
                "text": row.get("text"),
                "cleaned_text": row.get("cleaned_text"),
                "timestamp": row.get("timestamp"),
                "user_id": str(row.get("user_id")), # Ensure it's a string
                "username": row.get("username"),
                "retweet_count": int(row.get("retweet_count", 0)),
                "like_count": int(row.get("like_count", 0)),
                "reply_count": int(row.get("reply_count", 0)),
                "quote_count": int(row.get("quote_count", 0)),
                "keyword_matched": row.get("keyword_matched"),
                "hashtags": json.dumps(row.get("hashtags", [])), # Store as JSON string
                "urls": json.dumps(row.get("urls", [])),         # Store as JSON string
                "is_retweet": bool(row.get("is_retweet", False)),
                "language": row.get("language", "en")
            }
            tweets_to_save.append(tweet_data)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            print(f"Row data: {row.to_dict()}")
            continue

    try:
        saved_count = storage.save_tweets(tweets_to_save)
        print(f"Successfully saved {saved_count} tweets to the database.")
    except Exception as e:
        print(f"Error saving tweets to database: {e}")

if __name__ == "__main__":
    # Get the absolute path of the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assuming script is in project_root/temp_approach.py

    # Define the absolute path to your CSV file
    csv_file = os.path.join(project_root, "data", "scraped_tweets_new.csv")
    # Define the absolute path to your database file
    database_file = os.path.join(project_root, "twitter_anti_india_data.db") 
    
    load_csv_to_db(csv_file, database_file)
