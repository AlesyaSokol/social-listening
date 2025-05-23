import pandas as pd
import json
import threading
import requests
import time
from datetime import datetime, timezone, timedelta
import concurrent.futures
import schedule
from openai import OpenAI
import numpy as np
import tiktoken
import dotenv
import os
import numpy as np
from model.model_new import cluster_all_posts
from model.burst_detection import analyze_trends_for_period

from database.fill_database import Database

dotenv.load_dotenv('.env')

OPENAI_APIKEY = os.getenv('OPENAI_APIKEY')  
TOKEN_1 = os.getenv('TOKEN_1')  
TOKEN_2 = os.getenv('TOKEN_2')  

def main_scraping_and_clustering():
    cluster_all_posts(datetime.now() - timedelta(days=1))
    analyze_trends_for_period(datetime.now() - timedelta(days=1), 7)


# Главная функция
def main():
    main_scraping_and_clustering()


db = Database()

if __name__ == "__main__":
    main()