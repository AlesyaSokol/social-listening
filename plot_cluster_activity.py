import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import dotenv
from model.burst_detection import get_cluster_counts_from_qdrant

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def plot_cluster_activity(cluster_id, start_date, end_date, save_path=None):
    """
    Plot the number of posts in a cluster over time.
    
    Args:
        cluster_id (int): ID of the cluster to plot
        start_date (datetime): Start date for the period
        end_date (datetime): End date for the period
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    # Load environment variables
    dotenv.load_dotenv('.env')
    
    # Get cluster counts
    cluster_counts = get_cluster_counts_from_qdrant([cluster_id], start_date, end_date)
    
    if not cluster_counts or cluster_id not in cluster_counts:
        logging.error(f"No data found for cluster {cluster_id}")
        return
    
    # Extract dates and counts
    dates = sorted(cluster_counts[cluster_id].keys())
    counts = [cluster_counts[cluster_id][date] for date in dates]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, counts, marker='o', linestyle='-', linewidth=2)
    
    # Customize the plot
    plt.title(f'Cluster {cluster_id} Activity Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    # Example usage
    cluster_id = 8998  # Example cluster ID
    end_date = datetime(2025, 5, 19)
    start_date = end_date - timedelta(days=7)
    
    print(f"Plotting activity for cluster {cluster_id} from {start_date.date()} to {end_date.date()}")
    
    try:
        plot_cluster_activity(cluster_id, start_date, end_date)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 