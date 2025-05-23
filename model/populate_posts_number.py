import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client import models

# Load environment variables
load_dotenv()

def get_post_count_for_date(date):
    """Get number of posts for a specific date from Qdrant"""
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    
    # Set date range for the current day
    start_datetime = datetime(date.year, date.month, date.day)
    end_datetime = start_datetime + timedelta(days=1) - timedelta(microseconds=1)
    
    try:
        total_count = 0
        offset_id = None
        batch_size = 10000
        
        while True:
            # Count points in Qdrant for this date
            response = client.scroll(
                collection_name=os.getenv('QDRANT_COLLECTION'),
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="post_date",
                            range=models.DatetimeRange(
                                gte=start_datetime.isoformat(),
                                lt=end_datetime.isoformat()
                            )
                        )
                    ]
                ),
                limit=batch_size,
                offset=offset_id,
                with_payload=False,
                with_vectors=False
            )
            
            points = response[0]
            batch_count = len(points)
            total_count += batch_count
            
            # If we got less than batch_size, we've reached the end
            if batch_count < batch_size:
                break
                
            # Get the ID of the last point in this batch
            offset_id = points[-1].id
        
        return total_count
        
    except Exception as e:
        print(f"Error getting count for {date}: {str(e)}")
        return 0

def main():
    # Connect to PostgreSQL

    print(os.getenv('POSTGRES_DB').strip('"'))
    print(os.getenv('POSTGRES_USER').strip('"'))
    print(os.getenv('POSTGRES_PASSWORD').strip('"'))
    print(os.getenv('POSTGRES_HOST').strip('"'))
    print(os.getenv('POSTGRES_PORT').strip('"'))
    
    conn = psycopg2.connect(
        dbname=os.getenv('POSTGRES_DB').strip('"'),
        user=os.getenv('POSTGRES_USER').strip('"'),
        password=os.getenv('POSTGRES_PASSWORD').strip('"'),
        host=os.getenv('POSTGRES_HOST').strip('"'),
        port=int(os.getenv('POSTGRES_PORT').strip('"'))
    )
    
    # Create cursor
    cur = conn.cursor()
    
    # Define date range
    start_date = datetime(2025, 5, 20)
    end_date = datetime(2025, 5, 21)
    
    # Generate list of dates
    current_date = start_date
    while current_date <= end_date:
        # Get post count for current date
        count = get_post_count_for_date(current_date)
        
        # Insert into database
        try:
            cur.execute(
                "INSERT INTO posts_number (date, count) VALUES (%s, %s)",
                (current_date.date(), count)
            )
            print(f"Inserted {count} posts for {current_date.date()}")
        except Exception as e:
            print(f"Error inserting data for {current_date.date()}: {str(e)}")
        
        current_date += timedelta(days=1)
    
    # Commit changes and close connection
    conn.commit()
    cur.close()
    conn.close()
    
    print("Database population completed!")

if __name__ == "__main__":
    main() 