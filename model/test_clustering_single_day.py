from datetime import datetime
import dotenv
import os
from model.model_new import cluster_all_posts
import logging
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Set root logger to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
dotenv.load_dotenv()

def log_time(start_time, message):
    """Helper function to log elapsed time"""
    elapsed = time.time() - start_time
    logging.warning(f"{message} - Took {elapsed:.2f} seconds")
    return time.time()

def test_single_day_clustering():
    total_start = time.time()
    
    # Set test date
    test_date = datetime(2025, 5, 12)
    dates = [test_date]
    logging.warning(f"Testing clustering for single day: {test_date.date()}")
    
    try:
        # Run clustering with detailed timing
        logging.warning("\nStarting clustering process...")
        cluster_start = time.time()
        
        df_last_day, cluster_counts, all_post_labels = cluster_all_posts(dates, batch_size=10000)
        current_time = log_time(cluster_start, "Clustering completed")
        
        # Print detailed results
        logging.warning("\nClustering Results:")
        logging.warning(f"Total posts processed: {len(df_last_day)}")
        logging.warning(f"Number of clusters found: {len(cluster_counts)}")
        
        # Print cluster statistics
        if len(cluster_counts) > 0:
            logging.warning("\nCluster sizes:")
            sizes = []
            for cluster_id, daily_counts in cluster_counts.items():
                cluster_size = daily_counts[test_date]
                if cluster_size > 0:
                    sizes.append((cluster_id, cluster_size))
            
            # Sort clusters by size and print
            for cluster_id, size in sorted(sizes, key=lambda x: x[1], reverse=True):
                logging.warning(f"Cluster {cluster_id}: {size} posts")
            
            # Print size distribution statistics
            sizes = [s[1] for s in sizes]
            logging.warning(f"\nCluster size statistics:")
            logging.warning(f"Largest cluster: {max(sizes)} posts")
            logging.warning(f"Smallest cluster: {min(sizes)} posts")
            logging.warning(f"Average cluster size: {sum(sizes)/len(sizes):.1f} posts")
        
        # Print total time
        log_time(total_start, "\nTotal processing")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    test_single_day_clustering() 