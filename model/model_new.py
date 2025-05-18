import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import copy
import pickle
import random
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, models
import os
import dotenv
import logging
import http.client

# Set root logger to WARNING
logging.getLogger().setLevel(logging.WARNING)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import time
import copy

print("Current working directory:", os.getcwd())
print("Loading .env from:", os.path.abspath('.env'))
dotenv.load_dotenv('.env')
print("Environment variables loaded")
print("OPENAI_APIKEY:", os.getenv("OPENAI_APIKEY"))
print("QDRANT_ADDRESS:", os.getenv("QDRANT_ADDRESS"))
print("QDRANT_COLLECTION:", os.getenv("QDRANT_COLLECTION"))

def get_posts_for_day(target_date, limit=None, last_post_date=None):
    """
    Получает посты из Qdrant за конкретный день
    
    Args:
        target_date (datetime): дата, за которую нужно получить посты
        limit (int): максимальное количество постов для получения
        last_post_date (str): ISO-формат даты последнего поста из предыдущего батча для пагинации
    
    Returns:
        tuple: (DataFrame с постами, список эмбеддингов, дата последнего поста)
    """
    try:
        client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'), timeout=30.0)
        
        # Устанавливаем начало и конец дня
        start_date = datetime(target_date.year, target_date.month, target_date.day)
        end_date = start_date + timedelta(days=1) - timedelta(microseconds=1)
        
        # Преобразуем даты в ISO формат
        start_date_iso = start_date.isoformat()
        end_date_iso = end_date.isoformat()
        
        logging.warning(f"Fetching posts with limit={limit}")
        logging.warning(f"Date range: from {start_date_iso} to {end_date_iso}")
        logging.warning(f"Last post date: {last_post_date}")
        
        # Формируем запрос
        must_conditions = [
            models.FieldCondition(
                key="post_date",
                range=models.DatetimeRange(
                    gte=last_post_date if last_post_date else start_date_iso,
                    lt=end_date_iso
                )
            )
        ]
        
        # Выполняем запрос к Qdrant
        response = client.scroll(
            collection_name=os.getenv('QDRANT_COLLECTION'),
            scroll_filter=models.Filter(
                must=must_conditions
            ),
            limit=limit if limit else 100,
            with_payload=True,
            with_vectors=True,
            order_by="post_date"
        )
        
        points = response[0]  # Первый элемент - список точек
        
        if not points:
            logging.warning(f"No posts found for date {target_date.date()}")
            return pd.DataFrame(), [], None
        
        # Извлекаем данные из результатов
        posts_data = []
        embeddings = []
        last_post_date = None
        first_post_date = None
        
        for point in points:
            post_data = point.payload
            post_data['id'] = point.id
            posts_data.append(post_data)
            embeddings.append(point.vector)
            if first_post_date is None:
                first_post_date = post_data['post_date']
            last_post_date = post_data['post_date']  # Сохраняем дату последнего поста для пагинации
        
        logging.warning(f"Retrieved {len(posts_data)} posts")
        logging.warning(f"First post date in batch: {first_post_date}")
        logging.warning(f"Last post date in batch: {last_post_date}")
        
        # Создаем DataFrame
        df = pd.DataFrame(posts_data)
        
        return df, embeddings, last_post_date
        
    except Exception as e:
        logging.warning(f"Error fetching posts for {target_date.date()}: {str(e)}")
        return pd.DataFrame(), [], None

def update_qdrant_cluster_labels(post_labels):
    """
    Обновляет метки кластеров в Qdrant для каждого поста
    
    Args:
        post_labels: dict, словарь {(post_id, public_id): cluster_id}
    """
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'), timeout=30.0)
    collection_name = os.getenv('QDRANT_COLLECTION')
    
    # Группируем обновления по cluster_id для эффективности
    updates_by_cluster = {}
    for (post_id, public_id), cluster_id in post_labels.items():
        if cluster_id not in updates_by_cluster:
            updates_by_cluster[cluster_id] = []
        updates_by_cluster[cluster_id].append((post_id, public_id))
    
    # Обновляем payload для каждого кластера
    for cluster_id, posts in updates_by_cluster.items():
        try:
            # Создаем условие для фильтрации постов по post_id и public_id
            points_selector = []
            
            # Группируем посты по батчам для эффективности
            batch_size = 1000
            for i in range(0, len(posts), batch_size):
                batch_posts = posts[i:i + batch_size]
                
                # Создаем условие для фильтрации батча постов
                should_conditions = []
                for post_id, public_id in batch_posts:
                    should_conditions.append(
                        models.Filter(
                            must=[
                                models.FieldCondition(key="post_id", match={"value": post_id}),
                                models.FieldCondition(key="public_id", match={"value": public_id})
                            ]
                        )
                    )
                
                # Получаем points_selector для всего батча
                response = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        should=should_conditions
                    ),
                    limit=batch_size
                )
                
                if response[0]:  # If we found any points
                    points_selector.extend(point.id for point in response[0])
            
            if points_selector:  # Only update if we found points
                # Обновляем payload для всех постов в кластере
                client.set_payload(
                    collection_name=collection_name,
                    payload={"cluster_id": cluster_id},
                    points=points_selector
                )
            
        except Exception as e:
            logging.error(f"Error updating cluster {cluster_id}: {str(e)}")
            continue

def cluster_all_posts(dates, batch_size=50000):
    """
    Кластеризация постов по датам и батчам
    
    Args:
        dates (list): список дат для анализа
        batch_size (int): размер батча для кластеризации
    
    Returns:
        tuple: (df_last_day, cluster_daily_counts, all_post_labels)
        где all_post_labels это словарь {(post_id, public_id): cluster_id}
    """
    start_time = time.time()
    logging.warning(f"Starting clustering process with batch size: {batch_size}")
    
    # Инициализируем DataFrame для текущего дня
    current_df = pd.DataFrame()
    
    # Словарь для хранения количества постов в кластерах по дням
    cluster_daily_counts = {}
    
    # Словарь для хранения меток всех постов
    all_post_labels = {}
    
    last_cl = 0
    av_embeddings = []
    av_embeddings_date = []
    
    clusters_dict = {}  # Хранит центроиды кластеров
    
    for d in dates:
        day_start = time.time()
        logging.warning(f"\nProcessing date: {d.date()}")
        
        # Получаем посты батчами
        last_post_date = None
        batch_number = 1
        has_more_posts = True
        batch_post_labels = {}  # Словарь для хранения меток текущего батча
        total_posts_for_day = 0
        
        while has_more_posts:
            batch_time = time.time()
            logging.warning(f"\nProcessing batch {batch_number} for {d.date()}")
            logging.warning(f"Total posts processed so far for this day: {total_posts_for_day}")
            
            # Получаем следующий батч постов
            df_batch, embeddings_batch, last_post_date = get_posts_for_day(d, limit=batch_size, last_post_date=last_post_date)
            
            if len(df_batch) == 0:
                logging.warning("Got empty batch, stopping pagination")
                break
                
            total_posts_for_day += len(df_batch)
            logging.warning(f"Got batch of {len(df_batch)} posts")
            logging.warning(f"Updated total posts for day: {total_posts_for_day}")
            
            # Очищаем av_embeddings_date для нового батча
            if len(av_embeddings_date) > 0:
                av_embeddings.extend(av_embeddings_date)
            av_embeddings_date = []
            
            # Кластеризуем текущий батч
            cluster_start = time.time()
            agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, metric='cosine', linkage="average")
            labels = agglo.fit_predict(embeddings_batch)
            logging.warning(f"Agglomerative clustering completed - Took {time.time() - cluster_start:.2f} seconds")
            
            # Фильтруем кластеры по размеру
            filter_start = time.time()
            clusters = []
            for l in set(labels):
                if len(np.nonzero(labels==l)[0]) > 3:
                    if l not in clusters:
                        clusters.append(l)
            
            cl_dict = {c: i for i, c in enumerate(clusters)}
            labels1 = [cl_dict[l] if l in clusters else -1 for l in labels]
            logging.warning(f"Found {len(clusters)} clusters with >3 posts - Took {time.time() - filter_start:.2f} seconds")
            
            # Вычисляем центроиды для каждого кластера в батче
            centroid_start = time.time()
            for i in range(len(clusters)):
                embs_cluster = [embeddings_batch[j] for j,label in enumerate(labels1) if label==i]
                av_embeddings_date.append(np.mean(embs_cluster, axis=0))
            logging.warning(f"Calculated cluster centroids - Took {time.time() - centroid_start:.2f} seconds")
            
            df_batch['label'] = labels1
            df_batch['cluster'] = -1
            
            # Сравниваем с существующими кластерами
            compare_start = time.time()
            if len(clusters_dict) > 0:
                keys = list(clusters_dict.keys())
                av_embeddings = [clusters_dict[k] for k in keys]
                similarity_matrix = cosine_similarity(np.array(av_embeddings_date), np.array(av_embeddings))
                
                for i in range(len(av_embeddings_date)):
                    sim_emb = similarity_matrix[i]
                    ind_sim = np.argmax(sim_emb)
                    ind = keys[ind_sim]
                    
                    if sim_emb[ind_sim] > 0.6:
                        # Присваиваем существующий кластер
                        df_batch.loc[df_batch['label']==i, 'cluster'] = ind
                        # Обновляем центроид кластера
                        cluster_posts = df_batch[df_batch['label']==i]
                        cluster_embs = [embeddings_batch[j] for j,_ in enumerate(cluster_posts.index)]
                        clusters_dict[ind] = np.mean([clusters_dict[ind]] + cluster_embs, axis=0)
                        
                        # Обновляем количество постов
                        if ind not in cluster_daily_counts:
                            cluster_daily_counts[ind] = {date: 0 for date in dates}
                        cluster_daily_counts[ind][d] = cluster_daily_counts[ind].get(d, 0) + len(cluster_posts)
                    else:
                        # Создаем новый кластер
                        new_cluster_id = i + last_cl
                        df_batch.loc[df_batch['label']==i, 'cluster'] = new_cluster_id
                        clusters_dict[new_cluster_id] = av_embeddings_date[i]
                        
                        # Инициализируем счетчик
                        cluster_daily_counts[new_cluster_id] = {date: 0 for date in dates}
                        cluster_daily_counts[new_cluster_id][d] = len(df_batch[df_batch['label']==i])
            else:
                # Для первого батча создаем новые кластеры
                for i, c in enumerate(clusters):
                    cluster_id = i + last_cl
                    df_batch.loc[df_batch['label']==i, 'cluster'] = cluster_id
                    clusters_dict[cluster_id] = av_embeddings_date[i]
                    
                    # Инициализируем счетчик
                    cluster_daily_counts[cluster_id] = {date: 0 for date in dates}
                    cluster_daily_counts[cluster_id][d] = len(df_batch[df_batch['label']==i])
            
            logging.warning(f"Compared with existing clusters - Took {time.time() - compare_start:.2f} seconds")
            
            # Обновляем last_cl
            if len(df_batch[df_batch['cluster'] != -1]) > 0:
                last_cl = max(df_batch['cluster'].max() + 1, last_cl)
            
            # Сохраняем метки кластеров для текущего батча
            batch_post_labels = {}
            for _, row in df_batch.iterrows():
                if row['cluster'] != -1:
                    batch_post_labels[(row['post_id'], row['public_id'])] = row['cluster']
                    all_post_labels[(row['post_id'], row['public_id'])] = row['cluster']
            
            # Обновляем метки в Qdrant для текущего батча
            if batch_post_labels:
                update_start = time.time()
                logging.warning("\nUpdating cluster labels in Qdrant for current batch...")
                update_qdrant_cluster_labels(batch_post_labels)
                logging.warning(f"Finished updating cluster labels for batch - Took {time.time() - update_start:.2f} seconds")
            
            # Обновляем current_df для последнего батча последнего дня
            if d == dates[-1]:
                current_df = df_batch.copy()
            
            # Подготовка к следующему батчу
            has_more_posts = len(df_batch) == batch_size and last_post_date is not None
            if not has_more_posts:
                logging.warning(f"Stopping pagination because: batch size condition: {len(df_batch) == batch_size}, last_post_date condition: {last_post_date is not None}")
            
            batch_number += 1
            
            # Очищаем память
            del embeddings_batch
            logging.warning(f"Batch processing completed - Took {time.time() - batch_time:.2f} seconds")
        
        logging.warning(f"Day processing completed - Total posts processed: {total_posts_for_day}")
        logging.warning(f"Day processing took {time.time() - day_start:.2f} seconds")
    
    total_time = time.time() - start_time
    logging.warning(f"\nTotal clustering process completed - Took {total_time:.2f} seconds")
    
    return current_df, cluster_daily_counts, all_post_labels

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

# BURST DETECTION (the code from https://github.com/nmarinsek/burst_detection/tree/master with small changes)

np.float = float

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

def analyze_trends_for_period(start_date, end_date):
    """
    Анализирует тренды за указанный период
    
    Args:
        start_date (datetime): начальная дата
        end_date (datetime): конечная дата
    
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
    
    # Кластеризуем посты, получая их день за днем
    df_last_day, cluster_counts, all_post_labels = cluster_all_posts(dates)
    
    if len(df_last_day) == 0:
        return {"error": "No posts found for the specified period"}
    
    # Получаем временные ряды для каждого события
    events = events_time_series(df_last_day, cluster_counts, dates)
    
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
                    'total_posts': sum(cluster_counts[event_id].values())  # Общее количество постов во всех днях
                }
    
    # Добавляем метки постов в возвращаемый результат
    trends['post_labels'] = all_post_labels
    
    return trends 