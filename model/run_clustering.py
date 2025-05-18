import dotenv
from datetime import datetime, timedelta
from model_new import cluster_all_posts

# Load environment variables
dotenv.load_dotenv()

# Define date range
start_date = datetime(2025, 5, 12)
end_date = datetime(2025, 5, 17)

# Generate list of dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    current_date += timedelta(days=1)

# Run clustering
df_last_day, cluster_counts, all_post_labels = cluster_all_posts(dates, batch_size=10000) 