from datetime import datetime
from model.burst_detection import get_cluster_counts_from_qdrant, events_time_series

def main():
    # Define parameters
    cluster_ids = [1, 2, 3]
    start_date = datetime(2025, 5, 12)
    end_date = datetime(2025, 5, 13)
    
    # Get cluster counts from Qdrant
    print(f"Getting counts for clusters {cluster_ids} from {start_date.date()} to {end_date.date()}")
    cluster_counts = get_cluster_counts_from_qdrant(cluster_ids, start_date, end_date)
    
    # Generate dates list
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date = current_date.replace(day=current_date.day + 1)
    
    # Get time series
    print("\nGenerating time series...")
    events = events_time_series(cluster_counts, dates)
    
    # Print results
    print("\nResults:")
    for cluster_id in cluster_ids:
        print(f"\nCluster {cluster_id}:")
        for date, count in cluster_counts[cluster_id].items():
            print(f"  {date.date()}: {count} posts")
    
    print("\nTotal posts per day:")
    for i, date in enumerate(dates):
        print(f"  {date.date()}: {events['total'][i]} posts")

if __name__ == "__main__":
    main() 