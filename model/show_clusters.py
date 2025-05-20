from datetime import datetime, timedelta
import dotenv
import os
from qdrant_client import QdrantClient, models
import logging
import sys
import time

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

def show_clusters_for_day(target_date):
    """Show unique cluster IDs for posts from a given day"""
    logging.warning(f"Reading clusters for date: {target_date.date()}")
    
    # Initialize Qdrant client
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    # Set up date range for the day
    start_date = datetime(target_date.year, target_date.month, target_date.day)
    end_date = start_date + timedelta(days=1) - timedelta(microseconds=1)
    
    # Get posts with their cluster assignments
    offset_id = None
    batch_size = 1000
    cluster_counts = {}
    
    while True:
        # Get next batch of points
        points = client.scroll(
            collection_name=os.getenv('QDRANT_COLLECTION'),
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
            with_payload=["cluster_id"],
            with_vectors=False
        )[0]
        
        if not points:
            break
            
        # Process points in this batch
        for point in points:
            if 'cluster_id' in point.payload:
                cluster_id = point.payload['cluster_id']
                if cluster_id not in cluster_counts:
                    cluster_counts[cluster_id] = 0
                cluster_counts[cluster_id] += 1
        
        # Update offset_id for next batch
        offset_id = points[-1].id
        
        # If we got less than batch_size points, we've reached the end
        if len(points) < batch_size:
            break
    
    # Print results
    logging.warning(f"\nFound {len(cluster_counts)} unique clusters:")
    for cluster_id, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
        logging.warning(f"Cluster {cluster_id}: {count} posts")

def get_clusters_by_date_range(target_date):
    """Get clusters that started before target_date and ended on target_date"""
    logging.warning(f"Getting clusters for date range ending on: {target_date.date()}")
    
    # Initialize Qdrant client
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    # Set up date range
    end_date = datetime(target_date.year, target_date.month, target_date.day)
    end_date_iso = end_date.isoformat()
    
    # Get centroids with date range filter
    points = client.scroll(
        collection_name="cluster_centroids",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="end_date",
                    match={"value": end_date_iso}
                ),
                models.FieldCondition(
                    key="start_date",
                    range=models.DatetimeRange(
                        lt=end_date_iso
                    )
                )
            ]
        ),
        with_payload=["cluster_id", "start_date", "end_date", "post_count"],
        with_vectors=False
    )[0]
    
    # Print results
    logging.warning(f"\nFound {len(points)} clusters ending on {target_date.date()}:")
    for point in points:
        payload = point.payload
        logging.warning(f"Cluster {payload['cluster_id']}:")
        logging.warning(f"  - Started: {payload['start_date']}")
        logging.warning(f"  - Ended: {payload['end_date']}")
        logging.warning(f"  - Total posts: {payload['post_count']}")

if __name__ == "__main__":
    # Set test date
    test_date = datetime(2025, 5, 18)
    # show_clusters_for_day(test_date)
    # print("\n" + "="*50 + "\n")
    get_clusters_by_date_range(test_date) 