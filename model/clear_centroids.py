import dotenv
import os
import logging
import sys
from qdrant_client import QdrantClient

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

def clear_centroids_collection():
    """
    Delete and recreate the cluster_centroids collection.
    """
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    collection_name = "cluster_centroids"
    
    try:
        # Delete the collection
        client.delete_collection(collection_name=collection_name)
        logging.warning(f"Successfully deleted collection {collection_name}")
        
        # Recreate the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 1536,  # OpenAI embedding size
                "distance": "Cosine"
            }
        )
        logging.warning(f"Successfully recreated collection {collection_name}")
    except Exception as e:
        logging.error(f"Error managing collection: {str(e)}")
        raise

def main():
    logging.warning("Starting to clear cluster_centroids collection...")
    try:
        clear_centroids_collection()
    except Exception as e:
        logging.error(f"Error during collection clearing: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 