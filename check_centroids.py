import os
import dotenv
from qdrant_client import QdrantClient, models

def check_cluster_centroids(cluster_id):
    """
    Check if there are multiple centroids for a specific cluster_id.
    
    Args:
        cluster_id (int): The cluster ID to check
    """
    dotenv.load_dotenv('.env')
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    # Get all centroids for this cluster_id
    points = client.scroll(
        collection_name="cluster_centroids",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="cluster_id",
                    match={"value": int(cluster_id)}
                )
            ]
        ),
        with_payload=True,
        with_vectors=False
    )[0]
    
    if not points:
        print(f"No centroids found for cluster {cluster_id}")
        return
    
    print(f"\nFound {len(points)} centroids for cluster {cluster_id}:")
    total_count = 0
    
    for i, point in enumerate(points, 1):
        payload = point.payload
        print(f"\nCentroid {i}:")
        print(f"  Point ID: {point.id}")
        print(f"  Post count: {payload['post_count']}")
        print(f"  Start date: {payload['start_date']}")
        print(f"  End date: {payload['end_date']}")
        print(f"  Last updated: {payload['last_updated']}")
        total_count += int(payload['post_count'])
    
    print(f"\nTotal post count across all centroids: {total_count}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_centroids.py <cluster_id>")
        exit(1)
    
    cluster_id = int(sys.argv[1])
    check_cluster_centroids(cluster_id) 