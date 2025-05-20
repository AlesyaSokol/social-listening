import os
import dotenv
from qdrant_client import QdrantClient, models
import pandas as pd

# Load environment variables
dotenv.load_dotenv('.env')

def get_posts_from_cluster(cluster_id):
    """
    Retrieve posts from a specific cluster from Qdrant
    """
    try:
        
        # Initialize Qdrant client
        client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'), timeout=30.0)
        
        # Create filter for the specific cluster
        must_conditions = [
            models.FieldCondition(
                key="cluster_id",
                match={"value": cluster_id}
            )
        ]
        
        # Query Qdrant
        response = client.scroll(
            collection_name=os.getenv('QDRANT_COLLECTION'),
            scroll_filter=models.Filter(must=must_conditions),
            limit=1000,  # Limit to 100 posts
            with_payload=True,
            with_vectors=True,
            order_by="post_date"
        )
        
        points = response[0]
        
        if not points:
            print(f"No posts found for cluster {cluster_id}")
            return pd.DataFrame()
        
        # Extract post data
        posts_data = []
        for point in points:
            post_data = point.payload
            post_data['id'] = point.id
            posts_data.append(post_data)
        
        # Create DataFrame
        df = pd.DataFrame(posts_data)
        return df
        
    except Exception as e:
        print(f"Error fetching posts: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Get posts from cluster
    cluster_id = 42
    df = get_posts_from_cluster(cluster_id)
    
    if not df.empty:
        print(f"\nFound {len(df)} posts in cluster {cluster_id}")
        print("\nPost texts:")
        for i, row in df.iterrows():
            print(f"\n--- Post {i+1} ---")
            print(f"Date: {row['post_date']}")
            print(f"Text: {row['post_text']}")
            print("-" * 80)
    else:
        print(f"No posts found in cluster {cluster_id}") 