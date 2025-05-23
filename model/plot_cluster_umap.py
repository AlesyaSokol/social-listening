import os
import numpy as np
import matplotlib.pyplot as plt
from qdrant_client import QdrantClient, models
import umap
import dotenv

def get_vectors_for_cluster(cluster_id):
    dotenv.load_dotenv('.env')
    QDRANT_ADDRESS = os.getenv('QDRANT_ADDRESS')
    QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION')
    client = QdrantClient(url=QDRANT_ADDRESS)
    vectors = []
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
            with_vectors=True
        )[0]
        if not points:
            break
        vectors.extend([point.vector for point in points])
        offset = points[-1].id
        if len(points) < batch_size:
            break
    return np.array(vectors)

def plot_umap(vectors, cluster_id):
    print(len(vectors))
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(vectors)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.7)
    plt.title(f'UMAP projection of cluster {cluster_id}')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_cluster_umap.py <cluster_id>")
        exit(1)
    cluster_id = int(sys.argv[1])
    vectors = get_vectors_for_cluster(cluster_id)
    if vectors.shape[0] == 0:
        print(f"No vectors found for cluster {cluster_id}")
    else:
        plot_umap(vectors, cluster_id) 