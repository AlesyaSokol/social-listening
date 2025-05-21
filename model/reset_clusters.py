import os
import dotenv
from qdrant_client import QdrantClient, models
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv('.env')

def reset_all_cluster_ids(target_date):
    """
    Reset cluster_id to -1 for all posts in Qdrant for the given date
    """
    try:
        # Initialize Qdrant client
        client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'), timeout=30.0)
        collection_name = os.getenv('QDRANT_COLLECTION')
        
        # Get all points in batches
        offset = None
        batch_size = 1000
        total_updated = 0

        # Устанавливаем начало и конец дня
        start_date = datetime(target_date.year, target_date.month, target_date.day)
        end_date = start_date + timedelta(days=1) - timedelta(microseconds=1)
        
        # Преобразуем даты в ISO формат
        start_date_iso = start_date.isoformat()
        end_date_iso = end_date.isoformat()
        
        # Формируем запрос
        must_conditions = [
            models.FieldCondition(
                key="post_date",
                range=models.DatetimeRange(
                    gte=start_date_iso,
                    lt=end_date_iso
                )
            )
        ]
        
        while True:
            # Get batch of points
            response = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=must_conditions
                ),
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False  # We don't need vectors for this operation
            )
            
            points = response[0]
            if not points:
                break
                
            # Get IDs for this batch
            point_ids = [point.id for point in points]
            
            # Update cluster_id to -1 for all points in this batch
            client.set_payload(
                collection_name=collection_name,
                payload={"cluster_id": -1},
                points=point_ids
            )
            
            total_updated += len(point_ids)
            logger.info(f"Updated {total_updated} posts so far...")
            
            # Get offset for next batch
            offset = response[1]
            if offset is None:
                break
        
        logger.info(f"Successfully reset cluster_id to -1 for {total_updated} posts")
        return total_updated
        
    except Exception as e:
        logger.error(f"Error resetting cluster IDs: {str(e)}")
        return 0

if __name__ == "__main__":
    logger.info("Starting cluster ID reset process...")
    
    # Set date range
    start_date = datetime(2025, 5, 12)
    end_date = datetime(2025, 5, 17)
    
    # Process each day
    current_date = start_date
    while current_date <= end_date:
        logger.info(f"\nProcessing date: {current_date.date()}")
        total_updated = reset_all_cluster_ids(current_date)
        logger.info(f"Completed processing for {current_date.date()}")
        current_date += timedelta(days=1)
    
    logger.info("Process completed.") 