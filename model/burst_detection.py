import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import scipy.special as c
import os
import dotenv
from qdrant_client import QdrantClient, models

def get_cluster_counts_from_qdrant(cluster_ids, start_date, end_date):
    """
    Get daily post counts for specified clusters from Qdrant.
    
    Args:
        cluster_ids (list): List of cluster IDs to get counts for
        start_date (datetime): Start date for the period
        end_date (datetime): End date for the period
    
    Returns:
        dict: Dictionary with cluster counts in format {cluster_id: {date: count}}
    """
    # Load environment variables
    dotenv.load_dotenv('.env')
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    # Initialize cluster counts dictionary
    cluster_counts = {cluster_id: {} for cluster_id in cluster_ids}
    
    # Generate list of dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    # For each date, get counts for each cluster
    for date in dates:
        # Set date range for the current day
        start_datetime = datetime(date.year, date.month, date.day)
        end_datetime = start_datetime + timedelta(days=1) - timedelta(microseconds=1)
        
        # Get counts for each cluster
        for cluster_id in cluster_ids:
            try:
                # Count points in Qdrant for this cluster and date
                response = client.scroll(
                    collection_name=os.getenv('QDRANT_COLLECTION'),
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="cluster_id",
                                match={"value": int(cluster_id)}
                            ),
                            models.FieldCondition(
                                key="post_date",
                                range=models.DatetimeRange(
                                    gte=start_datetime.isoformat(),
                                    lt=end_datetime.isoformat()
                                )
                            )
                        ]
                    ),
                    limit=10000,  # We only need the count
                    with_payload=False,
                    with_vectors=False
                )
                
                # Store the count
                cluster_counts[cluster_id][date] = len(response[0])
                
            except Exception as e:
                logging.error(f"Error getting count for cluster {cluster_id} on {date}: {str(e)}")
                cluster_counts[cluster_id][date] = 0
                raise e
    
    return cluster_counts

def events_time_series(df, cluster_counts, dates):
    """Создание временных рядов для каждого события"""
    post_events = {}
    
    # Для каждого кластера создаем временной ряд из сохраненных счетчиков
    for cluster_id, daily_counts in cluster_counts.items():
        post_events[cluster_id] = [daily_counts[d] for d in dates]
    
    # Создаем временной ряд для общего количества постов
    total_counts = []
    for d in dates:
        total = sum(counts[d] for counts in cluster_counts.values())
        total_counts.append(total)
    post_events['total'] = total_counts

    return post_events

def tau(i1,i2,gamma,n):
    """Cost of switching states"""
    if i1>=i2:
        return 0
    else: 
        return (i2-i1) * gamma * np.log(n)

def fit(d,r,p):
    """Goodness of fit to the expected outputs of each state"""
    return -np.log(float(c.binomial(d,r)) * (p**r) * (1-p)**(d-r))

def burst_detection(r,d,n,s,gamma,smooth_win):
    """
    Burst detection for a two-state automaton
    """
    k = 2  # two states
    
    # smooth the data if the smoothing window is greater than 1
    if smooth_win > 1:
        temp_p = r/d
        temp_p = temp_p.rolling(window=smooth_win, center=True).mean()
        r = temp_p*d
        real_n = sum(~np.isnan(r))
    else: 
        real_n = n
          
    # calculate the expected proportions for states 0 and 1
    p = {}
    p[0] = np.nansum(r) / float(np.nansum(d))
    p[1] = p[0] * s
    if p[1] > 1:
        p[1] = 0.99999

    # initialize matrices
    cost = np.full([n,k],np.nan)
    q = np.full([n,1],np.nan)

    # Viterbi algorithm
    for t in range(int((smooth_win-1)/2),(int((smooth_win-1)/2))+real_n):
        for j in range(k): 
            if t==0:
                cost[t,j] = fit(d[t],r[t],p[j])
            else:
                cost[t,j] = tau(q[t-1],j,gamma,real_n) + fit(d[t],r[t],p[j])
        q[t] = np.argmin(cost[t, :])

    return q, d, r, p

def enumerate_bursts(q, label):
    """Enumerate the bursts from state sequence"""
    bursts = pd.DataFrame(columns=['label','begin','end','weight'])
    b = 0
    burst = False
    for t in range(1,len(q)):
        if (burst==False) & (q[t] > q[t-1]):
            bursts.loc[b,'begin'] = t
            burst = True
        if (burst==True) & (q[t] < q[t-1]):
            bursts.loc[b,'end'] = t
            burst = False
            b = b+1
    if burst == True:
        bursts.loc[b,'end']=t
    bursts.loc[:,'label'] = label
    return bursts

def burst_weights(bursts, r, d, p):
    """Calculate weights for each burst"""
    for b in range(len(bursts)):
        cost_diff_sum = 0
        for t in range(bursts.loc[b,'begin'], bursts.loc[b,'end']):
            cost_diff_sum = cost_diff_sum + (fit(d[t],r[t],p[0]) - fit(d[t],r[t],p[1]))
        bursts.loc[b,'weight'] = cost_diff_sum
    return bursts.sort_values(by='weight', ascending=False)

def find_bursts(post_events, dates, e):
    """Поиск всплесков активности"""
    q, d, r, p = burst_detection(post_events[e], post_events['total'], len(dates), s=2, gamma=1, smooth_win=1)
    bursts = enumerate_bursts(q, 'burstLabel')
    weighted_bursts = burst_weights(bursts, r, d, p)
    return weighted_bursts

def analyze_trends_for_period(start_date, end_date, df_last_day, all_post_labels):
    """
    Анализирует тренды за указанный период
    
    Args:
        start_date (datetime): начальная дата
        end_date (datetime): конечная дата
        df_last_day (DataFrame): DataFrame с постами за последний день
        all_post_labels (dict): словарь {(post_id, public_id): cluster_id} для всех постов
    
    Returns:
        dict: Словарь с трендами и их характеристиками, включая:
            - bursts: всплески активности
            - posts: примеры постов
            - total_posts: общее количество постов
            - post_labels: словарь {(post_id, public_id): cluster_id} для всех постов
    """
    # Получаем список дат для анализа
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    dates = sorted(dates)
    
    if len(df_last_day) == 0:
        return {"error": "No posts found for the specified period"}
    
    # Получаем временные ряды для каждого события
    events = events_time_series(df_last_day, all_post_labels, dates)
    
    # Анализируем всплески для каждого кластера
    trends = {}
    for event_id in events:
        if event_id != 'total':
            bursts = find_bursts(events, dates, event_id)
            if len(bursts) > 0:
                # Получаем тексты постов только из последнего дня для этого тренда
                trend_posts = df_last_day[df_last_day['cluster'] == event_id]['post_text'].tolist()
                trends[event_id] = {
                    'bursts': bursts,
                    'posts': trend_posts[:5] if trend_posts else [],  # Берем первые 5 постов как пример
                    'total_posts': sum(all_post_labels[event_id].values())  # Общее количество постов во всех днях
                }
    
    # Добавляем метки постов в возвращаемый результат
    trends['post_labels'] = all_post_labels
    
    return trends 