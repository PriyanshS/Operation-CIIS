import sqlite3
import os

# --- Configuration ---
SOURCE_DB = "twitter_anti_india_data.db"
TOP_LIKED_DB = "top_liked.db"
TOP_INFLUENCERS_DB = "top_influencers.db"
LIKES_THRESHOLD = 500
FOLLOWERS_THRESHOLD = 500

def get_db_connection(db_name):
    """Creates a database connection."""
    return sqlite3.connect(db_name)

def create_tweets_table(cursor):
    """Creates the tweets table with the same schema."""
    # The schema is based on twitter_storage_pipeline.py and inference_serving_api.py
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
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            classification_label TEXT,
            classification_confidence REAL,
            classification_probabilities TEXT,
            is_anti_india BOOLEAN,
            needs_review BOOLEAN
        )
    ''')

def process_top_liked_tweets(source_conn):
    """
    Loads anti-India tweets, sorts by likes, and saves top ones to a new DB.
    """
    print("Processing top liked tweets...")
    source_cursor = source_conn.cursor()

    # Check if classification_label column exists
    source_cursor.execute("PRAGMA table_info(tweets)")
    columns = [info[1] for info in source_cursor.fetchall()]
    if 'classification_label' not in columns:
        print("Error: 'classification_label' column not found in the source database.")
        print("Please run the inference API to classify tweets first.")
        return

    # 1. Load classified "anti-India" tweets, sorted by likes
    query = """
        SELECT * FROM tweets
        WHERE classification_label = 'anti-india'
        ORDER BY like_count DESC
    """
    source_cursor.execute(query)
    anti_india_tweets = source_cursor.fetchall()
    
    print(f"Found {len(anti_india_tweets)} 'anti-india' tweets.")

    # 2. Create new DB for top liked tweets
    if os.path.exists(TOP_LIKED_DB):
        os.remove(TOP_LIKED_DB)
    
    liked_conn = get_db_connection(TOP_LIKED_DB)
    liked_cursor = liked_conn.cursor()
    create_tweets_table(liked_cursor)

    # 3. Filter and insert tweets with more than LIKES_THRESHOLD likes
    top_liked_count = 0
    # Get column names from the source cursor description
    col_names = [description[0] for description in source_cursor.description]
    
    for tweet_row in anti_india_tweets:
        tweet_dict = dict(zip(col_names, tweet_row))
        if tweet_dict.get('like_count', 0) > LIKES_THRESHOLD:
            # Construct the insert statement dynamically
            # to handle any schema differences gracefully
            placeholders = ', '.join(['?'] * len(tweet_dict))
            insert_cols = ', '.join(tweet_dict.keys())
            sql = f"INSERT INTO tweets ({insert_cols}) VALUES ({placeholders})"
            liked_cursor.execute(sql, list(tweet_dict.values()))
            top_liked_count += 1
            
    liked_conn.commit()
    liked_conn.close()
    print(f"Created '{TOP_LIKED_DB}' with {top_liked_count} tweets having more than {LIKES_THRESHOLD} likes.")


def process_top_influencers(source_conn):
    """
    Attempts to filter tweets by follower count and save them to a new DB.
    """
    print("\nProcessing top influencers...")
    source_cursor = source_conn.cursor()

    # Check for follower_count column
    source_cursor.execute("PRAGMA table_info(tweets)")
    columns = [info[1] for info in source_cursor.fetchall()]
    
    if 'follower_count' not in columns:
        print("Warning: 'follower_count' column not found in the database schema.")
        print("Cannot create the 'top_influencers.db' database as requested.")
        print("Please ensure follower counts are scraped and stored in the database.")
        return

    # This part of the code will only run if 'follower_count' exists
    print("Found 'follower_count' column. Proceeding with influencer analysis.")
    
    # 1. Load anti-India tweets from authors with enough followers, sorted
    query = """
        SELECT * FROM tweets
        WHERE classification_label = 'anti-india' AND follower_count > ?
        ORDER BY follower_count DESC
    """
    source_cursor.execute(query, (FOLLOWERS_THRESHOLD,))
    influencer_tweets = source_cursor.fetchall()
    
    print(f"Found {len(influencer_tweets)} tweets from authors with more than {FOLLOWERS_THRESHOLD} followers.")

    # 2. Create new DB for top influencers
    if os.path.exists(TOP_INFLUENCERS_DB):
        os.remove(TOP_INFLUENCERS_DB)
        
    influencer_conn = get_db_connection(TOP_INFLUENCERS_DB)
    influencer_cursor = influencer_conn.cursor()
    create_tweets_table(influencer_cursor) # You might want a different schema here

    # 3. Insert the filtered tweets
    col_names = [description[0] for description in source_cursor.description]
    for tweet_row in influencer_tweets:
        tweet_dict = dict(zip(col_names, tweet_row))
        placeholders = ', '.join(['?'] * len(tweet_dict))
        insert_cols = ', '.join(tweet_dict.keys())
        sql = f"INSERT INTO tweets ({insert_cols}) VALUES ({placeholders})"
        influencer_cursor.execute(sql, list(tweet_dict.values()))

    influencer_conn.commit()
    influencer_conn.close()
    print(f"Created '{TOP_INFLUENCERS_DB}' with {len(influencer_tweets)} tweets.")


def main():
    """Main function to run the analysis."""
    if not os.path.exists(SOURCE_DB):
        print(f"Error: Source database '{SOURCE_DB}' not found.")
        print("Please ensure the database exists and contains classified tweets.")
        return

    source_conn = get_db_connection(SOURCE_DB)
    
    process_top_liked_tweets(source_conn)
    process_top_influencers(source_conn)
    
    source_conn.close()
    print("\nScript finished.")

if __name__ == "__main__":
    main()
