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
import time
import traceback

# Set root logger to WARNING
logging.getLogger().setLevel(logging.WARNING)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import time
import copy
from collections import defaultdict
# from .burst_detection import analyze_trends_for_period

def retry_qdrant_operation(operation_func, *args, **kwargs):
    """
    Retry a Qdrant operation indefinitely until it succeeds.
    
    Args:
        operation_func: Function to retry
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        The result of the operation if successful
    """
    # Commented out retry mechanism to fail immediately on errors
    while True:
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in Qdrant operation: {str(e)}")
            traceback.print_exc()
            logging.error("Retrying in 10 seconds...")
            time.sleep(10)
            continue
    # try:
    #     return operation_func(*args, **kwargs)
    # except Exception as e:
    #     logging.error(f"Error in Qdrant operation: {str(e)}")
    #     logging.error("Full traceback:")
    #     import traceback
    #     traceback.print_exc()
    #     raise  # Re-raise the exception to stop execution

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
        
        # Выполняем запрос к Qdrant с retry
        response = retry_qdrant_operation(
            client.scroll,
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
                                models.FieldCondition(key="post_id", match={"value": int(post_id)}),
                                models.FieldCondition(key="public_id", match={"value": int(public_id)})
                            ]
                        )
                    )
                
                # Получаем points_selector для всего батча с retry
                response = retry_qdrant_operation(
                    client.scroll,
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        should=should_conditions
                    ),
                    limit=batch_size
                )
                
                if response[0]:  # If we found any points
                    points_selector.extend(point.id for point in response[0])
            
            if points_selector:  # Only update if we found points
                # Обновляем payload для всех постов в кластере с retry
                retry_qdrant_operation(
                    client.set_payload,
                    collection_name=collection_name,
                    payload={"cluster_id": int(cluster_id)},
                    points=points_selector
                )
            
        except Exception as e:
            logging.error(f"Error updating cluster {cluster_id}: {str(e)}")
            raise  # Raise the exception instead of continuing

def update_cluster_centroid(cluster_id, new_vectors, target_date, existing_centroids=None):
    """
    Update the centroid of a cluster when new vectors are added.
    Returns the updated centroid data for batch processing.
    
    Args:
        cluster_id (int): The ID of the cluster to update
        new_vectors (list): List of new vectors to add to the cluster
        target_date (datetime): The date of the data being processed
        existing_centroids (dict, optional): Dictionary of existing centroids {cluster_id: vector}
    
    Returns:
        dict: Dictionary with centroid data for batch processing
    """
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    try:
        # Get current centroid and post count
        if existing_centroids and cluster_id in existing_centroids:
            centroid_data = existing_centroids[cluster_id]
            current_vector = centroid_data['vector']
            current_count = centroid_data['post_count']
            current_start_date = centroid_data['start_date']
            current_id = centroid_data['id']
        else:
            # Get full centroid data from Qdrant with retry
            response = retry_qdrant_operation(
                client.scroll,
                collection_name="cluster_centroids",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=int(cluster_id))
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=True
            )
            
            points = response[0]
            if not points:
                logging.warning(f"No centroid found for cluster {cluster_id}")
                return None
                
            current_centroid = points[0]
            current_vector = current_centroid.vector
            current_count = int(current_centroid.payload['post_count'])
            current_start_date = current_centroid.payload['start_date']
            current_id = current_centroid.id
        
        # Convert new vectors to numpy array for efficient computation
        new_vectors_array = np.array(new_vectors)
        num_new_vectors = len(new_vectors)
        
        # Calculate new centroid using weighted average
        new_centroid = (np.array(current_vector) * current_count + np.sum(new_vectors_array, axis=0)) / (current_count + num_new_vectors)
        
        # Calculate new count
        new_count = current_count + num_new_vectors
        
        # Update existing_centroids immediately with new data
        if existing_centroids is not None:
            existing_centroids[cluster_id] = {
                'vector': new_centroid.tolist(),
                'post_count': new_count,
                'start_date': current_start_date,
                'id': current_id
            }
        
        # Log the count update
        logging.warning(f"Updating cluster {cluster_id} count: {current_count} + {num_new_vectors} = {new_count}")
        
        # Return centroid data for batch processing
        return {
            'id': current_id,
            'vector': new_centroid.tolist(),
            'payload': {
                "cluster_id": int(cluster_id),
                "post_count": new_count,
                "start_date": current_start_date,
                "end_date": target_date.isoformat(),
                "last_updated": target_date.isoformat()
            }
        }
        
    except Exception as e:
        logging.error(f"Error updating centroid for cluster {cluster_id}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def create_cluster_centroid(cluster_id, vectors, start_date, end_date):
    """
    Create a new cluster centroid.
    Returns the centroid data for batch processing.
    
    Args:
        cluster_id (int): The ID of the new cluster
        vectors (list): List of vectors to create the centroid from
        start_date (datetime): Start date for the cluster
        end_date (datetime): End date for the cluster
    
    Returns:
        dict: Dictionary with centroid data for batch processing
    """
    try:
        # Calculate centroid from vectors
        vectors_array = np.array(vectors)
        centroid = np.mean(vectors_array, axis=0)
        
        # Create point ID from cluster_id
        point_id = hash(cluster_id) % (2**63 - 1)  # Ensure positive 64-bit integer
        
        # logging.warning(f"Prepared new centroid for cluster {cluster_id}:")
        # logging.warning(f"  - Point ID: {point_id}")
        # logging.warning(f"  - Number of posts: {len(vectors)}")
        # logging.warning(f"  - Start date: {start_date.isoformat()}")
        # logging.warning(f"  - End date: {end_date.isoformat()}")
        
        # Return centroid data for batch processing
        return {
            'id': point_id,
            'vector': centroid.tolist(),
            'payload': {
                "cluster_id": cluster_id,
                "post_count": len(vectors),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "last_updated": end_date.isoformat()
            }
        }
        
    except Exception as e:
        logging.error(f"Error creating centroid for cluster {cluster_id}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def save_centroids_batch(centroids_data, batch_size=1000):
    """
    Save centroids to Qdrant in batches.
    
    Args:
        centroids_data (list): List of dictionaries with centroid data
        batch_size (int): Size of each batch for saving
    """
    if not centroids_data:
        return
        
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    try:
        total_centroids = len(centroids_data)
        logging.warning(f"\nSaving {total_centroids} centroids to Qdrant in batches of {batch_size}")
        
        # Process centroids in batches
        for i in range(0, len(centroids_data), batch_size):
            batch = centroids_data[i:i + batch_size]
            batch_start = time.time()
            
            logging.warning(f"\nProcessing batch {i//batch_size + 1} of {(total_centroids + batch_size - 1)//batch_size}")
            logging.warning(f"Batch size: {len(batch)} centroids")
            
            # Log some stats about this batch
            cluster_ids = [int(c['payload']['cluster_id']) for c in batch]
            post_counts = [int(c['payload']['post_count']) for c in batch]
            
            points = [
                models.PointStruct(
                    id=int(centroid['id']),  # Convert numpy.int64 to Python int
                    vector=centroid['vector'],
                    payload={
                        "cluster_id": int(centroid['payload']['cluster_id']),  # Convert numpy.int64 to Python int
                        "post_count": int(centroid['payload']['post_count']),  # Convert numpy.int64 to Python int
                        "start_date": centroid['payload']['start_date'],
                        "end_date": centroid['payload']['end_date'],
                        "last_updated": centroid['payload']['last_updated']
                    }
                )
                for centroid in batch
            ]
            
            # Save batch to Qdrant with retry
            retry_qdrant_operation(
                client.upsert,
                collection_name="cluster_centroids",
                points=points
            )
            
            batch_time = time.time() - batch_start
            logging.warning(f"Saved batch to Qdrant - Took {batch_time:.2f} seconds")
            
        logging.warning(f"\nSuccessfully saved all {total_centroids} centroids to Qdrant")
            
    except Exception as e:
        logging.error(f"Error saving centroids batch: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Error saving centroids batch: {str(e)}")

def update_clusters_dict_from_centroids(centroids_data, clusters_dict):
    """
    Update clusters_dict with the latest data from saved centroids.
    
    Args:
        centroids_data (list): List of dictionaries with centroid data
        clusters_dict (dict): Dictionary to update with latest centroid data
    """
    for centroid in centroids_data:
        cluster_id = int(centroid['payload']['cluster_id'])
        clusters_dict[cluster_id] = {
            'vector': centroid['vector'],
            'post_count': int(centroid['payload']['post_count']),
            'start_date': centroid['payload']['start_date'],
            'id': centroid['id']
        }
    return clusters_dict

def cluster_all_posts(target_date, batch_size=10000):
    """
    Кластеризация постов для одного дня с использованием результатов кластеризации за прошлую неделю
    
    Args:
        target_date (datetime): дата для анализа
        batch_size (int): размер батча для кластеризации
    
    Returns:
        tuple: (df_last_day, all_post_labels)
        где all_post_labels это словарь {(post_id, public_id): cluster_id}
    """
    if batch_size != 10000:
        raise ValueError(f"Batch size must be 10000, got {batch_size}")
        
    start_time = time.time()
    logging.warning(f"Starting clustering process for date: {target_date.date()}")
    
    # Получаем существующие кластеры из Qdrant
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    clusters_dict = {}
    
    # Вычисляем дату неделю назад для фильтрации центроидов
    week_ago = target_date - timedelta(days=7)
    logging.warning(f"Retrieving centroids updated since: {week_ago.date()}")
    
    # Получаем центроиды из Qdrant с использованием батчинга, только те, что обновлялись за последнюю неделю
    offset_id = None
    centroid_batch_size = 1000
    centroid_start = time.time()
    
    while True:
        # Get next batch of points with retry
        points = retry_qdrant_operation(
            client.scroll,
            collection_name="cluster_centroids",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="last_updated",
                        range=models.DatetimeRange(
                            gte=week_ago.isoformat()
                        )
                    )
                ]
            ),
            limit=centroid_batch_size,
            offset=offset_id,
            with_payload=True,
            with_vectors=True
        )[0]
        
        if not points:
            break
            
        # Process points in this batch
        for point in points:
            cluster_id = int(point.payload['cluster_id'])  # Convert to Python int
            clusters_dict[cluster_id] = {
                'vector': point.vector,
                'post_count': int(point.payload['post_count']),  # Convert to Python int
                'start_date': point.payload['start_date'],
                'id': point.id
            }
        
        # Update offset_id for next batch
        offset_id = points[-1].id
        
        # If we got less than batch_size points, we've reached the end
        if len(points) < centroid_batch_size:
            break
    
    logging.warning(f"Retrieved {len(clusters_dict)} centroids from Qdrant (updated in the last week) - Took {time.time() - centroid_start:.2f} seconds")
    
    # Инициализируем DataFrame для текущего дня
    current_df = pd.DataFrame()
    
    # Словарь для хранения меток всех постов
    all_post_labels = {}
    
    # Get all existing cluster IDs from Qdrant with retry
    cluster_id_start = time.time()
    existing_clusters = retry_qdrant_operation(
        client.scroll,
        collection_name=os.getenv('QDRANT_COLLECTION'),
        limit=1,
        with_payload=True,
        with_vectors=False,
        order_by=models.OrderBy(
            key="cluster_id",
            direction="desc"
        )
    )[0]
    max_existing_cluster_id = int(existing_clusters[0].payload['cluster_id']) if existing_clusters else -1
    last_cl = max_existing_cluster_id + 1
    logging.warning(f"Retrieved max cluster ID: {max_existing_cluster_id}, next ID will be: {last_cl} - Took {time.time() - cluster_id_start:.2f} seconds")
    
    av_embeddings = []
    av_embeddings_date = []
    
    # Получаем посты батчами для текущего дня
    last_post_date = None
    batch_number = 1
    has_more_posts = True
    total_posts_for_day = 0
    
    # List to store centroids for batch processing
    centroids_to_save = []
    total_centroids_created = 0
    total_centroids_updated = 0
    
    while has_more_posts:
        batch_time = time.time()
        logging.warning(f"\nProcessing batch {batch_number} for {target_date.date()}")
        logging.warning(f"Total posts processed so far for this day: {total_posts_for_day}")
        logging.warning(f"Centroids in memory: {len(centroids_to_save)}")
        logging.warning(f"Total centroids created: {total_centroids_created}")
        logging.warning(f"Total centroids updated: {total_centroids_updated}")
        
        # Получаем следующий батч постов
        fetch_start = time.time()
        df_batch, embeddings_batch, last_post_date = get_posts_for_day(target_date, limit=batch_size, last_post_date=last_post_date)
        logging.warning(f"Fetched batch of posts - Took {time.time() - fetch_start:.2f} seconds")
        
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
        try:
            agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, metric='cosine', linkage="average")
            labels = agglo.fit_predict(embeddings_batch)
            logging.warning(f"Agglomerative clustering completed - Took {time.time() - cluster_start:.2f} seconds")
        except Exception as e:
            logging.error(f"Error during clustering: {str(e)}")
            raise RuntimeError(f"Clustering failed for batch {batch_number}: {str(e)}")
        
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
            av_embeddings = [clusters_dict[k]['vector'] for k in keys]
            similarity_matrix = cosine_similarity(np.array(av_embeddings_date), np.array(av_embeddings))
            
            for i in range(len(av_embeddings_date)):
                sim_emb = similarity_matrix[i]
                ind_sim = np.argmax(sim_emb)
                ind = keys[ind_sim]
                
                logging.warning(f"\nAnalyzing cluster {i} from current batch:")
                logging.warning(f"Best matching existing cluster: {ind}")
                logging.warning(f"Similarity score: {sim_emb[ind_sim]:.4f}")
                logging.warning(f"Number of posts in this cluster: {len(df_batch[df_batch['label']==i])}")
                if ind in clusters_dict:
                    logging.warning(f"Existing cluster info:")
                    logging.warning(f"  - Post count: {clusters_dict[ind]['post_count']}")
                    logging.warning(f"  - Start date: {clusters_dict[ind]['start_date']}")
                
                if sim_emb[ind_sim] > 0.6:
                    # Присваиваем существующий кластер
                    df_batch.loc[df_batch['label']==i, 'cluster'] = ind
                    
                    # Get sample posts from existing cluster
                    existing_posts = retry_qdrant_operation(
                        client.scroll,
                        collection_name=os.getenv('QDRANT_COLLECTION'),
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="cluster_id",
                                    match={"value": int(ind)}  # Convert numpy.int64 to Python int
                                )
                            ]
                        ),
                        limit=5,
                        with_payload=True,
                        with_vectors=False
                    )[0]
                    
                    # Log sample posts from existing cluster
                    if existing_posts:
                        logging.warning("\nSample posts from existing cluster %d:", ind)
                        sample_posts = random.sample(existing_posts, min(5, len(existing_posts)))
                        for post in sample_posts:
                            logging.warning("  - %s", post.payload['post_text'][:200] + "..." if len(post.payload['post_text']) > 200 else post.payload['post_text'])
                    
                    # Get posts being added to this cluster
                    cluster_posts = df_batch[df_batch['label']==i]
                    sample_new_posts = cluster_posts.sample(min(5, len(cluster_posts)))
                    
                    # Log posts being added
                    logging.warning("\nPosts being added to cluster %d:", ind)
                    for _, post in sample_new_posts.iterrows():
                        logging.warning("  - %s", post.get('post_text', '')[:200] + "..." if len(post.get('post_text', '')) > 200 else post.get('post_text', ''))
                    
                    # Обновляем центроид кластера
                    cluster_embs = [embeddings_batch[j] for j,_ in enumerate(cluster_posts.index)]
                    centroid_data = update_cluster_centroid(ind, cluster_embs, target_date, clusters_dict)
                    if centroid_data:
                        centroids_to_save.append(centroid_data)
                        total_centroids_updated += 1
                        logging.warning(f"Updated existing cluster {ind}")
                else:
                    # Создаем новый кластер
                    new_cluster_id = i + last_cl
                    df_batch.loc[df_batch['label']==i, 'cluster'] = new_cluster_id
                    cluster_posts = df_batch[df_batch['label']==i]  # Get posts for this cluster
                    clusters_dict[new_cluster_id] = {
                        'vector': av_embeddings_date[i],
                        'post_count': len(cluster_posts),  # Use actual number of posts
                        'start_date': target_date.isoformat(),
                        'id': hash(new_cluster_id) % (2**63 - 1)
                    }
                    
                    # Get posts for new cluster
                    sample_posts = cluster_posts.sample(min(5, len(cluster_posts)))
                    
                    # Log posts in new cluster
                    logging.warning("\nPosts in new cluster %d:", new_cluster_id)
                    for _, post in sample_posts.iterrows():
                        logging.warning("  - %s", post.get('post_text', '')[:200] + "..." if len(post.get('post_text', '')) > 200 else post.get('post_text', ''))
                    
                    # Создаем новый центроид
                    cluster_embs = [embeddings_batch[j] for j,_ in enumerate(cluster_posts.index)]
                    centroid_data = create_cluster_centroid(new_cluster_id, cluster_embs, target_date, target_date)
                    if centroid_data:
                        centroids_to_save.append(centroid_data)
                        total_centroids_created += 1
                        logging.warning(f"Created new cluster {new_cluster_id}")
        else:
            # Для первого батча создаем новые кластеры
            for i, c in enumerate(clusters):
                cluster_id = i + last_cl
                df_batch.loc[df_batch['label']==i, 'cluster'] = cluster_id
                cluster_posts = df_batch[df_batch['label']==i]  # Get posts for this cluster
                clusters_dict[cluster_id] = {
                    'vector': av_embeddings_date[i],
                    'post_count': len(cluster_posts),  # Use actual number of posts
                    'start_date': target_date.isoformat(),
                    'id': hash(cluster_id) % (2**63 - 1)
                }
                
                # Get posts for new cluster
                sample_posts = cluster_posts.sample(min(5, len(cluster_posts)))
                
                logging.warning(f"\nPosts in new cluster {cluster_id} (first batch):")
                for _, post in sample_posts.iterrows():
                    logging.warning(f"  - {post.get('post_text', '')[:200]}...")
                
                # Создаем новый центроид
                cluster_embs = [embeddings_batch[j] for j,_ in enumerate(cluster_posts.index)]
                centroid_data = create_cluster_centroid(cluster_id, cluster_embs, target_date, target_date)
                if centroid_data:
                    centroids_to_save.append(centroid_data)
                    total_centroids_created += 1
                    logging.warning(f"Created new cluster {cluster_id} (first batch)")
        
        # Save centroids in batches if we have enough
        if len(centroids_to_save) >= 1000:
            save_start = time.time()
            logging.warning(f"\nSaving batch of {len(centroids_to_save)} centroids to Qdrant")
            save_centroids_batch(centroids_to_save)
            # Update clusters_dict with the latest data
            clusters_dict = update_clusters_dict_from_centroids(centroids_to_save, clusters_dict)
            logging.warning(f"Centroid batch saved and clusters_dict updated - Took {time.time() - save_start:.2f} seconds")
            centroids_to_save = []
        
        logging.warning(f"Compared with existing clusters - Took {time.time() - compare_start:.2f} seconds")
        
        # Обновляем last_cl
        if len(df_batch[df_batch['cluster'] != -1]) > 0:
            last_cl = max(df_batch['cluster'].max() + 1, last_cl)
            logging.warning(f"Updated last_cl to: {last_cl}")
        
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
        
        # Обновляем current_df для последнего батча
        current_df = df_batch.copy()
        
        # Подготовка к следующему батчу
        has_more_posts = len(df_batch) == batch_size and last_post_date is not None
        if not has_more_posts:
            logging.warning(f"Stopping pagination because: batch size condition: {len(df_batch) == batch_size}, last_post_date condition: {last_post_date is not None}")
        
        batch_number += 1
        
        # Очищаем память
        del embeddings_batch
        logging.warning(f"Batch processing completed - Took {time.time() - batch_time:.2f} seconds")
    
    # Save any remaining centroids
    if centroids_to_save:
        save_start = time.time()
        logging.warning(f"\nSaving final batch of {len(centroids_to_save)} centroids to Qdrant")
        save_centroids_batch(centroids_to_save)
        # Update clusters_dict with the final batch data
        clusters_dict = update_clusters_dict_from_centroids(centroids_to_save, clusters_dict)
        logging.warning(f"Final centroid batch saved and clusters_dict updated - Took {time.time() - save_start:.2f} seconds")
    
    total_time = time.time() - start_time
    logging.warning(f"\nTotal clustering process completed - Took {total_time:.2f} seconds")
    logging.warning(f"Total posts processed: {total_posts_for_day}")
    logging.warning(f"Total clusters created: {total_centroids_created}")
    logging.warning(f"Total clusters updated: {total_centroids_updated}")
    logging.warning(f"Total clusters in memory: {len(centroids_to_save)}")
    
    return current_df, all_post_labels

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

def ensure_centroids_collection():
    """
    Ensures that the centroids collection exists in Qdrant.
    Creates it if it doesn't exist.
    """
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_ADDRESS"),
        api_key=os.getenv("QDRANT_API_KEY", None)
    )
    
    collection_name = "cluster_centroids"
    
    # Check if collection exists
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        logging.warning(f"Creating new collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embedding size
                distance=models.Distance.COSINE
            )
        )
        logging.warning(f"Collection {collection_name} created successfully")
    else:
        logging.warning(f"Collection {collection_name} already exists")

def calculate_cluster_centroids(start_date, end_date, cluster_ids):
    """
    Calculate centroids for specified clusters within a date range and save them to Qdrant.
    
    Args:
        start_date (datetime): Start date for the period
        end_date (datetime): End date for the period
        cluster_ids (list): List of cluster IDs to process
    
    Returns:
        dict: Dictionary with cluster centroids in format {cluster_id: {'vector': centroid_vector, 'count': post_count}}
    """
    logging.warning(f"Calculating cluster centroids for period {start_date.date()} to {end_date.date()}")
    logging.warning(f"Processing {len(cluster_ids)} clusters")
    
    # Ensure centroids collection exists
    ensure_centroids_collection()
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_ADDRESS"),
        api_key=os.getenv("QDRANT_API_KEY", None)
    )
    
    # Dictionary to store cluster centroids
    cluster_centroids = {}
    
    # Process each cluster separately
    for cluster_id in cluster_ids:
        logging.warning(f"Processing cluster {cluster_id}")
        
        # Get all posts for this cluster in the date range using batching
        all_vectors = []
        offset_id = None
        batch_size = 1000
        
        while True:
            # Get next batch of points
            points = qdrant_client.scroll(
                collection_name=os.getenv("QDRANT_COLLECTION"),
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="post_date",
                            range=models.DatetimeRange(
                                gte=start_date.isoformat(),
                                lt=end_date.isoformat()
                            )
                        ),
                        models.FieldCondition(
                            key="cluster_id",
                            match={"value": cluster_id}
                        )
                    ]
                ),
                limit=batch_size,
                offset=offset_id,
                with_payload=True,
                with_vectors=True
            )[0]
            
            if not points:
                break
                
            # Add vectors from this batch
            all_vectors.extend([point.vector for point in points])
            
            # Update offset_id for next batch
            offset_id = points[-1].id
            
            # If we got less than batch_size points, we've reached the end
            if len(points) < batch_size:
                break
        
        if not all_vectors:
            logging.warning(f"No posts found for cluster {cluster_id}")
            continue
        
        # Calculate centroid for this cluster
        centroid_vector = np.mean(all_vectors, axis=0)
        
        # Store centroid info
        cluster_centroids[cluster_id] = {
            'vector': centroid_vector.tolist(),
            'count': len(all_vectors)
        }
        
        # Save centroid to Qdrant
        point_id = hash(cluster_id) % (2**63 - 1)  # Ensure positive 64-bit integer
        
        payload = {
            "cluster_id": cluster_id,
            "post_count": len(all_vectors),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        qdrant_client.upsert(
            collection_name="cluster_centroids",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=centroid_vector.tolist(),
                    payload=payload
                )
            ]
        )
        
        logging.warning(f"Processed cluster {cluster_id}: {len(all_vectors)} posts")
    
    logging.warning(f"Finished processing {len(cluster_centroids)} clusters")
    return cluster_centroids 
