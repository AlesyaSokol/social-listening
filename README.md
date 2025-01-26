**SOCIAL LISTENING**

We are making a tool to monitor what topics people in Russia are really interested in using “social listening” - monitoring user posts and regional ones in Russian on the most popular social network in Russia - VK. 

The tool tracks the most popular issues that people care about at the nascent stage of public interest - localized stories that often escape independent media attention despite public interest. The analysis is performed using machine learning and natural language processing techniques.

**model/model.py**

The model for extracting trending topics from posts cluster posts into topics and consists of the following steps:

- cluster_all_posts clusters posts embeddings with linkage algorithm. It finds linkage clusters for each day. If the centroid of the clusters for a new day differs from the previous clusters' centroids less than for a threshold, it is united with the previous most similar cluster

- events_time_series forms a time series of posts for each topic (cluster)

- tau, fit, burst_detection, enumerate_bursts, burst_weights, and find_bursts  implement Kleinberg’s burst detection algorithm to bursts in topic appearance (based on https://github.com/nmarinsek/burst_detection/tree/master)

- ask_gpt makes a topic summary using openai models

- extract_news unifies the information about clusters and their bursts into easily interpretable data that are used in frontend

**database/main.py**

Daily Multithreaded Scraping of Data from VKontakte News Publics and Text Embedding Transformation for Subsequent Clustering

The ScrappingPosts function downloads posts from each public starting from the last download date and saves the results to a database. The scraping process utilizes two VK API tokens and proxies to bypass bans.

The get_embedding function processes every 1000 posts, converting their texts into embeddings.

**database/create_db.py**

Creates Postgres Database with tables from `create_tables.sql`

**frontend/app**

Svelte app to display topics with sample posts

Launch:
```
cd frontend/app
npm run dev
```

**Dependencies**

The dependencies for running python scripts are defined in the `requirements.txt` file

The structure of the `.env` file:
```
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_HOST=
POSTGRES_PORT=
POSTGRES_DB=
OPENAI_APIKEY=
PROXY_HTTP=
PROXY_HTTPS=
TOKEN_1=
TOKEN_2=
```

