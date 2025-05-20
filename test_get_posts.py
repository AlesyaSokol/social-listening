import os
import dotenv
from datetime import datetime
import logging
from model.model_new import get_posts_for_day

# Set up logging
logging.basicConfig(level=logging.WARNING)

# Load environment variables
dotenv.load_dotenv('.env')

def test_get_posts():
    # Test date - using a specific date as requested
    target_date = datetime(2025, 5, 18)  # May 18, 2025
    
    # Get posts with limit 10000
    df, embeddings, last_post_date = get_posts_for_day(target_date, limit=10000)
    
    # Print results
    print(f"\nResults for {target_date.date()}:")
    print(f"Number of posts retrieved: {len(df)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Last post date: {last_post_date}")
    
    if len(df) > 0:
        print("\nFirst post:")
        print(f"Date: {df.iloc[0]['post_date']}")
        print(f"Text: {df.iloc[0]['post_text'][:100]}...")
        
        print("\nLast post:")
        print(f"Date: {df.iloc[-1]['post_date']}")
        print(f"Text: {df.iloc[-1]['post_text'][:100]}...")

if __name__ == "__main__":
    test_get_posts() 