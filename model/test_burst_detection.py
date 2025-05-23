import os
import logging
from datetime import datetime, timedelta
from burst_detection import analyze_trends_for_period

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Load environment variables
    # dotenv.load_dotenv('.env')
    
    # Set target date and lookback period
    target_date = datetime(2025, 5, 21)
    lookback_days = 9
    
    print(f"Analyzing trends for {target_date.date()} with {lookback_days} days lookback")
    
    try:
        # Get trends
        bursts = analyze_trends_for_period(target_date, lookback_days)
        
        # Print results
        if bursts:
            print(f"\nFound {len(bursts)} bursts:")
            # for burst in bursts:
            #     print(f"\nCluster {burst['cluster_id']}:")
            #     print(f"Number of posts: {burst['number_of_posts']}")
                # print("Posts:")
                # for post in burst['posts']:
                #     print(f"- {post['text'][:100]}...")
                #     print(f"  Link: {post['link']}")
        else:
            print("\nNo bursts found")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 