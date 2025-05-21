from datetime import datetime, timedelta
import dotenv
import os
from model_new import cluster_all_posts
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
dotenv.load_dotenv()

def run_clustering_for_date_range(start_date, end_date):
    """Run clustering for each day in the date range"""
    current_date = start_date
    while current_date <= end_date:
        logging.warning(f"\nProcessing date: {current_date.date()}")
        try:
            df_last_day, all_post_labels = cluster_all_posts(current_date, batch_size=10000)
            logging.warning(f"Successfully processed {len(df_last_day)} posts")
        except Exception as e:
            logging.error(f"Error processing date {current_date.date()}: {str(e)}")
            logging.error("Full traceback:")
            import traceback
            traceback.print_exc()
            raise  # Re-raise the exception to stop execution
        
        current_date += timedelta(days=1)

if __name__ == "__main__":
    # Set date range
    start_date = datetime(2025, 5, 12)
    end_date = datetime(2025, 5, 19)
    
    run_clustering_for_date_range(start_date, end_date) 