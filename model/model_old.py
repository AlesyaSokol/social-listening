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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import time
import copy
import datetime

print("Current working directory:", os.getcwd())
print("Loading .env from:", os.path.abspath('.env'))
dotenv.load_dotenv('.env')
print("Environment variables loaded")
print("OPENAI_APIKEY:", os.getenv("OPENAI_APIKEY"))
print("QDRANT_ADDRESS:", os.getenv("QDRANT_ADDRESS"))
print("QDRANT_COLLECTION:", os.getenv("QDRANT_COLLECTION"))

def get_posts_from_qdrant(start_date, end_date):
    """
    Получает посты из Qdrant за указанный период
    
    Args:
        start_date (datetime): начальная дата
        end_date (datetime): конечная дата
    
    Returns:
        pd.DataFrame: DataFrame с постами и их эмбеддингами
    """
    client = QdrantClient(url=os.getenv('QDRANT_ADDRESS'))
    
    # Формируем запрос к Qdrant с фильтром по датам
    response = client.scroll(
        collection_name=os.getenv('QDRANT_COLLECTION'),
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="post_date",
                    range=models.Range(
                        gte=start_date.isoformat(),
                        lte=end_date.isoformat()
                    )
                )
            ]
        ),
        with_payload=True,
        with_vectors=True
    )
    
    # Преобразуем результаты в DataFrame
    posts = []
    embeddings = []
    
    for point in response[0]:
        posts.append({
            'public_id': point.payload['public_id'],
            'post_text': point.payload['post_text'],
            'post_date': datetime.fromisoformat(point.payload['post_date']),
            'post_id': point.payload['post_id'],
            'views': point.payload['views'],
            'reposts': point.payload['reposts'],
            'likes': point.payload['likes'],
            'comments': point.payload['comments']
        })
        embeddings.append(point.vector)
    
    df = pd.DataFrame(posts)
    return df, embeddings

# MAIN FUNCTION FOR CLUSTERING

def cluster_all_posts(df, dates, embeddings):

    # This function is the main result of the hackathon

    ind_label_df = np.zeros(len(df))
    for i in range(len(df)):
        ind_label_df[i] = -1

    last_cl = 0
    av_embeddings = []
    av_embeddings_date = []
    
    clusters_dict = {}
    
    for d in dates:
    
        if len(av_embeddings_date)>0:
            av_embeddings += copy.deepcopy(av_embeddings_date)
    
        av_embeddings_date = []
        
        df_date = df[df['post_date']==d]
        embeddings_date = np.array(list(df_date['post_text']))
        
        agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, metric='cosine', linkage="average")
        labels = agglo.fit_predict(embeddings_date)
    
        clusters = []
        for l in labels:
            if len(np.nonzero(labels==l)[0])>3:
                if l not in clusters:
                    clusters.append(l)
    
        cl_dict = {}
        for i,c in enumerate(clusters):
            cl_dict[c] = i#+last_cl
    
        labels1 = [cl_dict[l] if l in clusters else -1 for l in labels]
    
        for i in range(len(clusters)):
            embs_cluster = [e for j,e in enumerate(embeddings_date) if labels1[j]==i]
            # print(len(embs_cluster))
            av_embeddings_date.append(np.mean(embs_cluster, axis = 0))
    
        df_date['label'] = labels1
    
        print(len(av_embeddings_date),len(av_embeddings))
        print(d)
    
        
        if d!=dates[0]:
            keys = list(clusters_dict.keys())
            av_embeddings = [clusters_dict[k] for k in keys]
            similarity_matrix = cosine_similarity(np.array(av_embeddings_date), np.array(av_embeddings))
            for i in range(len(av_embeddings_date)):
                sim_emb = similarity_matrix[i]
                ind_sim = np.argmax(sim_emb)
                ind = keys[ind_sim]
                if sim_emb[ind_sim]>0.5:
                    for j in df_date[df_date['label']==i].index:
                        ind_label_df[j]=ind
                    # print(j, i,ind)
                    embs_cluster = [e for j,e in enumerate(embeddings) if ind_label_df[j]==ind]
                    clusters_dict[ind] = np.mean(embs_cluster, axis = 0)
                else:
                    print(i)
                    for k,j in enumerate(df_date[df_date['label']==i].index):
                        ind_label_df[j]=i+last_cl
                    clusters_dict[i+last_cl] = av_embeddings_date[i]
        else:
            for i,c in enumerate(clusters):
                ind_label_df[df_date[df_date['label']==i].index] = [i+last_cl]*len(df_date[df_date['label']==i])
                clusters_dict[i+last_cl] = av_embeddings_date[i]
    
        last_cl += max(ind_label_df)

    df['cluster'] = ind_label_df

    return df


def events_time_series(df, dates):

    post_events = {}
    for e in sorted(list(set(df['cluster'])))[1:]:
        l = []
        df_event = df[df['cluster']==e]
        for d in dates:
            l.append(len(df_event[df_event['post_date']==d]))
        post_events[e] = l
    
    l = []
    for d in dates:
        l.append(len(df[df['post_date']==d]))
        post_events['total'] = l

    return post_events


# BURST DETECTION (the code from https://github.com/nmarinsek/burst_detection/tree/master with small changes)

np.float = float

import pandas as pd
import numpy as np
import sympy.functions.combinatorial.factorials as c


#define the transition cost tau: cost of switching states
#there's a cost to move up states, no cost to move down
#based on definition on pg. 8
#inputs
#   i1: current state
#   i2: next state
#   gam: gamma, penalty for moving up a state
#   n: number of timepoints
def tau(i1,i2,gamma,n):
    if i1>=i2:
        return 0
    else: 
        return (i2-i1) * gamma * np.log(n)
    
#define the fit cost: goodness of fit to the expected outputs of each state
#based on equation on bottom of pg. 14
#    d: number of events in each time period (1xn)
#    r: number of target events in each time period (1xn)
#    p: expected proportions of each state (1xk)
def fit(d,r,p):
    return -np.log(float(c.binomial(d,r)) * (p**r) * (1-p)**(d-r))


#define the burst detection function for a two-state automaton
#inputs:
#   r: number of target events in each time period (1xn)
#   d: number of events in each time period (1xn)
#   n: number of timepoints
#   s: multiplicative distance between states
#   gamma: difficulty to move up a state
#   smooth_win: width of smoothing window (use odd numbers)
#output:
#   q: optimal state sequence (1xn)
def burst_detection(r,d,n,s,gamma,smooth_win):
    
    k = 2 #two states
    
    #smooth the data if the smoothing window is greater than 1
    if smooth_win > 1:
        temp_p = r/d #calculate the proportions over time and smooth
        temp_p = temp_p.rolling(window=smooth_win, center=True).mean()
        #update r to reflect the smoothed proportions
        r = temp_p*d
        real_n = sum(~np.isnan(r))  #update the number of timepoints
    else: 
        real_n = n
          
    #calculate the expected proportions for states 0 and 1
    p = {}
    p[0] = np.nansum(r) / float(np.nansum(d))   #overall proportion of events, baseline state
    p[1] = p[0] * s                             #proportion of events during active state
    if p[1] > 1:                                #p1 can't be bigger than 1
        p[1] = 0.99999

    #initialize matrices to hold the costs and optimal state sequence
    cost = np.full([n,k],np.nan)
    q = np.full([n,1],np.nan)

    #use the Viterbi algorithm to find the optimal state sequence
    for t in range(int((smooth_win-1)/2),(int((smooth_win-1)/2))+real_n):

        #calculate the cost to transition to each state
        for j in range(k): 

            #for the first timepoint, calculate the fit cost only
            if t == int((smooth_win-1)/2):
                cost[t,j] = fit(d[t],r[t],p[j])

            #for all other timepoints, calculate the fit and transition cost
            else:
                cost[t,j] = tau(q[t-1],j,gamma,real_n) + fit(d[t],r[t],p[j])

        #add the state with the minimum cost to the optimal state sequence
        q[t] = np.argmin(cost[t, :])

    return q, d, r, p

#define a function to enumerate the bursts
#input: 
#   q: optimal state sequence
#output:
#   bursts: dataframe with beginning and end of each burst
def enumerate_bursts(q, label):
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

    #if the burst is still going, set end to last timepoint
    if burst == True:
        bursts.loc[b,'end']=t

    bursts.loc[:,'label'] = label
            
    return bursts

#define a function that finds the weights associated with each burst
#find the difference in the cost functions for p0 and p1 in each burst 
#inputs:
#   bursts: dataframe containing the beginning and end of each burst
#   r: number of target events in each time period
#   d: number of events in each time period
#   p: expected proportion for each state
#output:
#   bursts: dataframe containing the weights of each burst, in order
def burst_weights(bursts, r, d, p):
    
    #loop through bursts
    for b in range(len(bursts)):

        cost_diff_sum = 0

        for t in range(bursts.loc[b,'begin'], bursts.loc[b,'end']):

            cost_diff_sum = cost_diff_sum + (fit(d[t],r[t],p[0]) - fit(d[t],r[t],p[1]))

        bursts.loc[b,'weight'] = cost_diff_sum
        
    return bursts.sort_values(by='weight', ascending=False)



def find_bursts(post_events, dates, e):
    
    q, d, r, p = burst_detection(post_events[e],post_events['total'],len(dates),s=2,gamma=1,smooth_win=1)
    # print(q.T)
    bursts = bd.enumerate_bursts(q, 'burstLabel')
    weighted_bursts = bd.burst_weights(bursts,r,d,p)

    return weighted_bursts


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



def extract_news(post_events, dates, e):
    weighted_bursts = find_bursts(post_events, dates, e)

    rows = []
    for i,item in weighted_bursts.iterrows():
        if item['weight']>12:
            row = {}
            df_posts = df[(df['post_date']==dates[item['begin']])&(df['cluster']==e)]
            if len(df_posts)>5:
                links = []
                for j,item1 in df_posts.iterrows():
                    link = 'https://vk.com/wall'+str(item1['public_id'])+'_'+str(item1['post_date'])
                    links.append(link)
                row['links'] = links
                row['post_texts'] = list(df_posts['post_text'])
                row['regions'] = list(df_posts['region'])
                row['n_posts'] = len(df_posts)
                row['topic_gpt'] = ask_gpt(prompt, random.sample(row['post_texts'],min(20,len(row['post_texts']))))
                row['date'] = dates[item['begin']]
                rows.append(row)

    return rows



df = pd.read_csv('posts_embeddings.csv')
df = df.drop_duplicates(subset = 'post_text')
df['post_text'] = [[float(e1) for e1 in e[1:-1].split(', ')] for e in df['post_text']]
df = df[[len(str(t).split())>5 for t in df['post_text']]]
dates = sorted(set(df['post_date'].dt.date))

post_events = events_time_series(df, dates)

embeddings = np.array(list(df['post_text']))

rows = []
for i,e in enumerate(sorted(list(set(df['cluster'])))[1:]):
    print(i)
    rows += extract_news(post_events, dates, e)

def analyze_trends_for_period(start_date, end_date):
    """
    Анализирует тренды за указанный период
    
    Args:
        start_date (datetime): начальная дата
        end_date (datetime): конечная дата
    
    Returns:
        dict: Словарь с трендами и их характеристиками
    """
    # Получаем данные из Qdrant
    df, embeddings = get_posts_from_qdrant(start_date, end_date)
    
    if len(df) == 0:
        return {"error": "No posts found for the specified period"}
    
    # Получаем уникальные даты для анализа
    dates = sorted(df['post_date'].dt.date.unique())
    
    # Кластеризуем посты
    df_clustered = cluster_all_posts(df, dates, embeddings)
    
    # Получаем временные ряды для каждого события
    events = events_time_series(df_clustered, dates)
    
    # Анализируем всплески для каждого кластера
    trends = {}
    for event_id in events:
        if event_id != 'total':
            bursts = find_bursts(events, dates, event_id)
            if len(bursts) > 0:
                # Получаем тексты постов для этого тренда
                trend_posts = df_clustered[df_clustered['cluster'] == event_id]['post_text'].tolist()
                trends[event_id] = {
                    'bursts': bursts,
                    'posts': trend_posts[:5],  # Берем первые 5 постов как пример
                    'total_posts': len(trend_posts)
                }
    
    return trends

def analyze_from_csv():
    """Анализ трендов из CSV файла (старая версия)"""
    df = pd.read_csv('posts_embeddings.csv')
    df = df.drop_duplicates(subset = 'post_text')
    df['post_text'] = [[float(e1) for e1 in e[1:-1].split(', ')] for e in df['post_text']]
    df = df[[len(str(t).split())>5 for t in df['post_text']]]
    dates = sorted(set(df['post_date'].dt.date))

    post_events = events_time_series(df, dates)
    embeddings = np.array(list(df['post_text']))

    rows = []
    for i,e in enumerate(sorted(list(set(df['cluster'])))[1:]):
        print(i)
        rows += extract_news(post_events, dates, e)
    return rows

if __name__ == "__main__":
    # Этот код будет выполняться только при прямом запуске файла
    analyze_from_csv()

