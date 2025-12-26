
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

def connect_to_db():
    """Connects to the MongoDB database using the URI from the .env file."""
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not found in .env file")

    client = MongoClient(mongo_uri)
    return client

def engagement_analysis(client, csv_file_path):
    """
    Performs engagement analysis on the given CSV file.

    Args:
        client: The MongoDB client.
        csv_file_path: The path to the input CSV file.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    # Store the data in the 'anti-india' database
    db = client["anti-india"]
    collection = db["tweets"]
    collection.insert_many(df.to_dict("records"))
    print("Successfully stored tweets in the 'anti-india' database.")

    # Get tweets with more than 500 likes and store them in the 'Top_Liked' collection
    top_liked_tweets = list(collection.find({"likes": {"$gt": 500}}).sort("likes", -1))
    if top_liked_tweets:
        top_liked_collection = db["Top_Liked"]
        top_liked_collection.insert_many(top_liked_tweets)
        print("Successfully stored top liked tweets in the 'Top_Liked' collection.")

    # Get handles with more than 500 followers from the 'Top_Liked' collection and store them in the 'Top_Influencers' collection
    if top_liked_tweets:
        top_influencers = list(db["Top_Liked"].find({"followers": {"$gt": 500}}).sort("followers", -1))
        if top_influencers:
            top_influencers_collection = db["Top_Influencers"]
            top_influencers_collection.insert_many(top_influencers)
            print("Successfully stored top influencers in the 'Top_Influencers' collection.")

if __name__ == "__main__":
    # Assuming the input CSV file is 'data/cleaned_sentiment_data.csv'
    # Please change this if your file has a different name.
    csv_file = "data/cleaned_sentiment_data.csv"
    
    try:
        mongo_client = connect_to_db()
        engagement_analysis(mongo_client, csv_file)
        mongo_client.close()
    except ValueError as e:
        print(e)
