import os
import dotenv
from qdrant_client import QdrantClient, models

def count_posts_in_cluster(cluster_id):
    dotenv.load_dotenv('.env')
    QDRANT_ADDRESS = os.getenv('QDRANT_ADDRESS')
    QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION')
    client = QdrantClient(url=QDRANT_ADDRESS)
    count = 0
    offset = None
    batch_size = 1000

    while True:
        points = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="cluster_id",
                        match={"value": cluster_id}
                    )
                ]
            ),
            limit=batch_size,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )[0]
        if not points:
            break
        count += len(points)
        offset = points[-1].id
        if len(points) < batch_size:
            break
    return count

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python count_cluster_posts.py <cluster_id>")
        exit(1)
    cluster_id = int(sys.argv[1])
    count = count_posts_in_cluster(cluster_id)
    print(f"Cluster {cluster_id} contains {count} posts.") 