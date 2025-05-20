import os
import dotenv
from qdrant_client import QdrantClient
import json

def analyze_centroids_collection():
    dotenv.load_dotenv('.env')
    QDRANT_ADDRESS = os.getenv('QDRANT_ADDRESS')
    client = QdrantClient(url=QDRANT_ADDRESS)
    
    # Get collection info
    collection_info = client.get_collection('centroids')
    print("\nCollection Info:")
    print(json.dumps(collection_info, indent=2))
    
    # Get first few centroids to examine their structure
    points = client.scroll(
        collection_name='centroids',
        limit=5,
        with_payload=True,
        with_vectors=False
    )[0]
    
    print("\nSample Centroid Structure:")
    for point in points:
        print(f"\nPoint ID: {point.id}")
        print("Payload:", json.dumps(point.payload, indent=2))

if __name__ == "__main__":
    analyze_centroids_collection() 