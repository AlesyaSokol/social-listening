from datetime import datetime
import dotenv
import os
import logging
import sys
from qdrant_client import QdrantClient, models

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

def check_centroids(start_date, end_date):
    """
    Check centroids stored in Qdrant for the specified date range.
    """
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    # Get all centroids from the collection
    response = client.scroll(
        collection_name="cluster_centroids",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="start_date",
                    range=models.DatetimeRange(
                        gte=start_date.isoformat(),
                        lt=end_date.isoformat()
                    )
                )
            ]
        ),
        with_payload=True,
        with_vectors=True,
        limit = 1000
    )
    
    points = response[0]
    if not points:
        logging.warning("No centroids found for the specified period")
        return
    
    # Print information about each centroid
    logging.warning(f"\nFound {len(points)} centroids:")
    for point in points:
        payload = point.payload
        logging.warning(f"\nCluster ID: {payload['cluster_id']}")
        logging.warning(f"Post count: {payload['post_count']}")
        logging.warning(f"Start date: {payload['start_date']}")
        logging.warning(f"End date: {payload['end_date']}")
        logging.warning(f"Last updated: {payload['last_updated']}")
        logging.warning(f"Vector size: {len(point.vector)}")
        
        # Print first few dimensions of the vector
        vector_preview = point.vector[:5]
        logging.warning(f"Vector preview (first 5 dimensions): {vector_preview}")

def main():
    # Set date range (same as in calculate_centroids.py)
    start_date = datetime(2025, 5, 12)
    end_date = datetime(2025, 5, 17, 23, 59, 59, 999999)
    
    logging.warning(f"Checking centroids for period {start_date.date()} to {end_date.date()}")
    
    try:
        check_centroids(start_date, end_date)
    except Exception as e:
        logging.error(f"Error checking centroids: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 