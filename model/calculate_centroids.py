from datetime import datetime
import dotenv
import os
import logging
import sys
from model_new import calculate_cluster_centroids
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

def get_cluster_ids(start_date, end_date):
    """
    Get all unique cluster IDs from the date range using ID-based batching.
    Only retrieves cluster_ids from payload for efficiency.
    """
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    cluster_ids = set()
    batch_size = 1000
    offset_id = None
    
    while True:
        # Get next batch of points
        response = client.scroll(
            collection_name=os.getenv("QDRANT_COLLECTION"),
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="post_date",
                        range=models.DatetimeRange(
                            gte=start_date.isoformat(),
                            lt=end_date.isoformat()
                        )
                    )
                ]
            ),
            limit=batch_size,
            offset=offset_id,
            with_payload=["cluster_id"],  # Only get cluster_id from payload
        )
        
        points = response[0]
        if not points:
            break
            
        # Extract cluster_ids from this batch
        for point in points:
            if point.payload and 'cluster_id' in point.payload:
                cluster_id = point.payload['cluster_id']
                if cluster_id != -1 and cluster_id >= 2007:  # Skip unclustered posts and clusters before 1910
                    cluster_ids.add(cluster_id)
        
        # Update offset_id for next batch using the last point's ID
        offset_id = points[-1].id
        logging.warning(f"Processed batch, found {len(cluster_ids)} unique clusters so far")
        
        # If we got less than batch_size points, we've reached the end
        if len(points) < batch_size:
            break
    
    return list(cluster_ids)

def main():
    # Set date range
    start_date = datetime(2025, 5, 12)
    end_date = datetime(2025, 5, 17, 23, 59, 59, 999999)
    
    logging.warning(f"Starting centroid calculation for period {start_date.date()} to {end_date.date()}")
    
    try:
        # Get cluster IDs
        cluster_ids = get_cluster_ids(start_date, end_date)
        logging.warning(f"Found {len(cluster_ids)} clusters to process")
        
        # Calculate and save centroids
        centroids = calculate_cluster_centroids(start_date, end_date, cluster_ids)
        
        # Print summary
        logging.warning("\nCentroid calculation completed successfully")
        logging.warning(f"Total clusters processed: {len(centroids)}")
        
        # Print cluster sizes
        cluster_sizes = [(cluster_id, info['count']) for cluster_id, info in centroids.items()]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        logging.warning("\nTop 10 largest clusters:")
        for cluster_id, size in cluster_sizes[:10]:
            logging.warning(f"Cluster {cluster_id}: {size} posts")
        
    except Exception as e:
        logging.error(f"Error during centroid calculation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 