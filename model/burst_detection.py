import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import scipy.special as c
import os
import dotenv
from qdrant_client import QdrantClient, models
import psycopg2
import json
import random

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
    
    # For each date, get all posts in batches
    for date in dates:
        # Set date range for the current day
        start_datetime = datetime(date.year, date.month, date.day)
        end_datetime = start_datetime + timedelta(days=1) - timedelta(microseconds=1)
        
        try:
            # Initialize date counts
            date_counts = {cluster_id: 0 for cluster_id in cluster_ids}
            offset = None
            batch_size = 10000
            
            while True:
                # Get batch of posts for this date
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
                            ),
                            models.FieldCondition(
                                key="cluster_id",
                                match=models.MatchAny(
                                    any=[int(cluster_id) for cluster_id in cluster_ids]
                                )
                            )
                        ]
                    ),
                    limit=batch_size,
                    offset=offset,
                    with_payload=["cluster_id"],
                    with_vectors=False
                )
                
                if len(response[0]) < 2:
                    break
                
                # Count posts per cluster in this batch
                for point in response[0]:
                    if 'cluster_id' in point.payload:
                        cluster_id = int(point.payload['cluster_id'])
                        if cluster_id in cluster_ids:
                            date_counts[cluster_id] += 1
                
                # Update offset for next batch
                offset = response[0][-1].id
            
            # Store counts for each cluster
            for cluster_id in cluster_ids:
                cluster_counts[cluster_id][date] = date_counts[cluster_id]
                
        except Exception as e:
            logging.error(f"Error getting counts for date {date}: {str(e)}")
            for cluster_id in cluster_ids:
                cluster_counts[cluster_id][date] = 0
            raise e
    
    return cluster_counts

def events_time_series(cluster_counts, dates):
    """Создание временных рядов для каждого события"""
    post_events = {}
    
    # Для каждого кластера создаем временной ряд из сохраненных счетчиков
    for cluster_id, daily_counts in cluster_counts.items():
        post_events[cluster_id] = [daily_counts[d] for d in dates]
    
    # Создаем временной ряд для общего количества постов
    # total_counts = []
    # for d in dates:
    #     total = sum(counts[d] for counts in cluster_counts.values())
    #     total_counts.append(total)
    # post_events['total'] = total_counts

    return post_events

def tau(i1,i2,gamma,n):
    """Cost of switching states"""
    if i1>=i2:
        return 0
    else: 
        return (i2-i1) * gamma * np.log(n)

def fit(d, r, p):
    try:
        comb = float(c.comb(d, r))
        prob = (p**r) * ((1-p)**(d-r))
        val = comb * prob
        if val <= 0 or np.isnan(val):
            # Log the problematic values for debugging
            print(f"fit() warning: d={d}, r={r}, p={p}, comb={comb}, prob={prob}, val={val}")
            return 1000  # or a large value
        return -np.log(val)
    except Exception as e:
        print(f"fit() exception: d={d}, r={r}, p={p}, error={e}")
        return np.inf

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
        burst_point = bursts.loc[b,'begin']
        
        if burst_point == bursts.loc[b,'end']:  # Single-point burst
            # Compare burst point with previous point
            if burst_point > 0:  # Make sure we have a previous point
                # Calculate how much better the burst point fits the burst state
                burst_fit = fit(d[burst_point], r[burst_point], p[1])
                prev_fit = fit(d[burst_point-1], r[burst_point-1], p[0])
                cost_diff_sum = burst_fit - prev_fit
        else:  # Multi-point burst
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

def fill_post_regions(bursts):
    """
    For each post in each burst, fill the 'region' key using public_id.
    Minimizes DB calls by batching.
    """
    # 1. Collect all unique public_ids
    public_ids = set()
    for burst in bursts:
        for post in burst['posts']:
            public_id = post.get('public_id')
            if public_id is not None:
                public_ids.add(public_id)

    logging.warning(f"Found {len(public_ids)} unique public_ids")
    if not public_ids:
        return bursts

    # 2. Query the database for all public_ids at once
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB').strip('"'),
            user=os.getenv('POSTGRES_USER').strip('"'),
            password=os.getenv('POSTGRES_PASSWORD').strip('"'),
            host=os.getenv('POSTGRES_HOST').strip('"'),
            port=int(os.getenv('POSTGRES_PORT').strip('"'))
        )
        cur = conn.cursor()
        # Prepare the query for all public_ids
        format_strings = ','.join(['%s'] * len(public_ids))
        cur.execute(
            f"SELECT id, region_name FROM publics WHERE id IN ({format_strings})",
            tuple(public_ids)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error fetching region names: {e}")
        return bursts

    logging.warning(f"Fetched {len(rows)} region names")

    # 3. Build the mapping
    public_id_to_region = {str(row[0]): row[1] for row in rows}

    # 4. Fill the 'region' key for each post
    for burst in bursts:
        for post in burst['posts']:
            public_id = str(post.get('public_id'))
            post['region'] = public_id_to_region.get(public_id)

    return bursts

from collections import Counter

def get_majority_region(posts):
    """
    Given a list of posts (each with 'region'), returns the region name if more than 50% of posts
    belong to the same region. Otherwise, returns None.

    Args:
        posts (list): List of dicts, each with a 'region' field.

    Returns:
        str or None: Region name if majority, else None.
    """
    region_list = [post.get('region') for post in posts if post.get('region')]
    # logging.warning(f"Region list: {region_list}")
    if not region_list:
        return None

    region_counts = Counter(region_list)
    most_common_region, count = region_counts.most_common(1)[0]

    # logging.warning(f"Most common region: {most_common_region}, count: {count}, total posts: {len(posts)}")
    if count > len(posts) / 2:
        return most_common_region
    else:
        return None
    

 # GPT SUMMARY

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_APIKEY")
client_openai = OpenAI(api_key = OPENAI_API_KEY)

def ask_gpt(prompt, posts, model = 'gpt-4o-mini'):
    chat_completion = client_openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt + '\n\n'.join(posts),
            }
        ],
        model=model
    )
    return chat_completion.choices[0].message.content

prompt = "You're a newsroom helper. Summarize the posts in one caption (less than 30 words) containing the most common topic of the posts. Do NOT use any introductory words like 'The main topic is...', just formulate the topic itself. Answer in Russian. Posts:\n\n"
    

def analyze_trends_for_period(target_date, lookback_days):
    """
    Анализирует тренды для указанной даты, используя данные за предыдущие дни
    
    Args:
        target_date (datetime): дата для анализа
        lookback_days (int): количество дней до target_date для анализа
    
    Returns:
        list: Список словарей с информацией о всплесках, где каждый словарь содержит:
            - cluster_id: ID кластера
            - posts: список всех постов из кластера на target_date
            - number_of_posts: количество постов в кластере на target_date
    """
    logging.warning(f"Starting trend analysis for {target_date.date()} with {lookback_days} days lookback")
    
    # Calculate start date based on lookback
    start_date = target_date - timedelta(days=lookback_days)
    end_date = target_date
    logging.warning(f"Analysis period: from {start_date.date()} to {end_date.date()}")
    
    # Get list of dates for analysis
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    dates = sorted(dates)
    logging.warning(f"Generated {len(dates)} dates for analysis")
    
    # Get total post counts from PostgreSQL
    logging.warning("Fetching post counts from PostgreSQL...")
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB').strip('"'),
            user=os.getenv('POSTGRES_USER').strip('"'),
            password=os.getenv('POSTGRES_PASSWORD').strip('"'),
            host=os.getenv('POSTGRES_HOST').strip('"'),
            port=int(os.getenv('POSTGRES_PORT').strip('"'))
        )
        cur = conn.cursor()
        
        # Get post counts for each date
        post_counts = {}
        for date in dates:
            cur.execute(
                "SELECT count FROM posts_number WHERE date = %s",
                (date.date(),)
            )
            result = cur.fetchone()
            post_counts[date] = result[0] if result else 0
            logging.warning(f"Post count for {date.date()}: {post_counts[date]}")
        
        cur.close()
        conn.close()
        logging.warning("Successfully retrieved post counts from PostgreSQL")
        
    except Exception as e:
        logging.error(f"Error getting post counts from database: {str(e)}")
        raise e
    
    # Get clusters from Qdrant for target date only
    logging.warning("Connecting to Qdrant...")
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    # Set date range for target date
    start_datetime = datetime(target_date.year, target_date.month, target_date.day)
    end_datetime = start_datetime + timedelta(days=1) - timedelta(microseconds=1)
    logging.warning(f"Fetching clusters for target date: {target_date.date()}")
    
    # Get unique cluster IDs for target date using batching
    cluster_ids = set()
    offset = None
    batch_size = 10000
    batch_count = 0
    
    while True:
        batch_count += 1
        logging.warning(f"Fetching batch {batch_count} of cluster IDs...")
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
            offset=offset,
            with_payload=["cluster_id"],
            with_vectors=False
        )
        
        if len(response[0]) < 2:
            logging.warning("No more points to fetch")
            break
        
        # Extract cluster IDs from the batch
        batch_clusters = set()
        points_with_clusters = 0
        for point in response[0]:
            if 'cluster_id' in point.payload:
                batch_clusters.add(int(point.payload['cluster_id']))
                points_with_clusters += 1
        
        cluster_ids.update(batch_clusters)
        logging.warning(f"Found {len(response[0])} points, {points_with_clusters} with cluster_ids, {len(batch_clusters)} unique clusters in batch {batch_count}")
        
        # Only update offset if the batch is non-empty
        offset = response[0][-1].id
    
    if not cluster_ids:
        logging.warning("No clusters found for target date")
        raise Exception("No clusters found for target date")
    
    logging.warning(f"Total unique clusters found: {len(cluster_ids)}")
    
    # Get cluster counts for all dates
    logging.warning("Fetching cluster counts for all dates...")
    cluster_counts = get_cluster_counts_from_qdrant(list(cluster_ids), start_date, end_date)
    logging.warning(f"Retrieved counts for {len(cluster_counts)} clusters")
    
    # Get time series for events
    logging.warning("Generating time series for events...")
    events = events_time_series(cluster_counts, dates)
    logging.warning(f"Generated time series for {len(events)} events")
    events['total'] = [post_counts[d] for d in dates]
    
    # Analyze bursts for each cluster
    logging.warning("Analyzing bursts for each cluster...")
    bursts_list = []
    total_bursts = 0
    
    for event_id in events:
        if event_id != 'total':
            # logging.warning(f"Analyzing bursts for cluster {event_id}")
            # 1. Log number of posts for each analyzed date
            posts_per_date = [cluster_counts[int(event_id)][date] for date in dates]
            # logging.warning(f"Cluster {event_id} - posts per date: {posts_per_date}")
            # 2. Log the weighted bursts
            bursts = find_bursts(events, dates, event_id)
            if event_id == 8302:
                logging.warning(f"Cluster {event_id} - weighted bursts:\n{bursts}")

            # Filter bursts to only include those occurring on target_date
            target_date_bursts = bursts[(bursts['end'] == len(dates)-1) & (bursts['begin'] == len(dates)-1)]
            target_date_bursts = target_date_bursts[target_date_bursts['weight'] > 12]
            # logging.warning(f"Filtering bursts: end={bursts['end'].values}, len(dates)={len(dates)}")
            
            if len(target_date_bursts) > 0:

                if posts_per_date[-1] > 10:
                    
                    total_bursts += len(target_date_bursts)
                    # logging.warning(f"Found {len(target_date_bursts)} bursts on target date for cluster {event_id}")
                    
                    # Get all posts for this cluster from target date
                    all_posts = []
                    post_response = client.scroll(
                        collection_name=os.getenv('QDRANT_COLLECTION'),
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="post_date",
                                    range=models.DatetimeRange(
                                        gte=start_datetime.isoformat(),
                                        lt=end_datetime.isoformat()
                                    )
                                ),
                                models.FieldCondition(
                                    key="cluster_id",
                                    match={"value": int(event_id)}
                                )
                            ]
                        ),
                        limit=10000,  # Get all posts for this cluster on target date
                        with_payload=["post_text", "public_id", "post_id"],
                        with_vectors=False
                    )
                    
                    for point in post_response[0]:
                        all_posts.append({"text": point.payload.get('post_text', ''), 
                                          "public_id": point.payload.get('public_id', ''),
                                          "link": "https://vk.com/wall" + str(point.payload.get('public_id', '')) + "_" + str(point.payload.get('post_id', ''))})
                    
                    # Add each burst as a separate dictionary to the list
                    for _, burst in target_date_bursts.iterrows():
                        bursts_list.append({
                            'cluster_id': event_id,
                            'posts': all_posts,
                            'number_of_posts': len(all_posts)
                        })
                    
                    # logging.warning(f"Added {len(target_date_bursts)} bursts for cluster {event_id} with {len(all_posts)} posts")
            # else:
                # logging.warning(f"No bursts found on target date for cluster {event_id}")
    
    logging.warning(f"Analysis complete. Found {total_bursts} bursts on target date across {len(set(b['cluster_id'] for b in bursts_list))} clusters")

    logging.warning("Filling post regions...")
    bursts_list = fill_post_regions(bursts_list)

    logging.warning("Recording results to PostgreSQL...")
    # Record results to PostgreSQL
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB').strip('"'),
            user=os.getenv('POSTGRES_USER').strip('"'),
            password=os.getenv('POSTGRES_PASSWORD').strip('"'),
            host=os.getenv('POSTGRES_HOST').strip('"'),
            port=int(os.getenv('POSTGRES_PORT').strip('"'))
        )
        cur = conn.cursor()
        for burst in bursts_list:
            # Prepare related_posts as JSON or array
            related_posts = json.dumps(burst['posts'])
            cur.execute(
                """
                INSERT INTO model_output (title, cluster_id, related_posts, update_date, post_count, region)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (ask_gpt(prompt, random.sample([b['text'] for b in burst['posts']], min(20, len(burst['posts'])))), burst['cluster_id'], related_posts, target_date.date(), burst['number_of_posts'], get_majority_region(burst['posts']))
            )
        conn.commit()
        cur.close()
        conn.close()
        logging.warning(f"Inserted {len(bursts_list)} bursts into the database.")
    except Exception as e:
        logging.error(f"Error inserting bursts into database: {str(e)}")

    return bursts_list 