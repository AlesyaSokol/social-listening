import psycopg2
import os
import dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
import uuid
import traceback

dotenv.load_dotenv('../.env')

class Database:
    DB_NAME = os.getenv('POSTGRES_DB')  
    DB_USER = os.getenv('POSTGRES_USER')   
    DB_PASS = os.getenv('POSTGRES_PASSWORD') 
    DB_HOST = os.getenv('POSTGRES_HOST')  
    DB_PORT = os.getenv('POSTGRES_PORT')   
    QDRANT_ADDRESS = os.getenv('QDRANT_ADDRESS')
    QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION')

    def __init__(self):
        """Initialize db class variables"""
        self.conn = psycopg2.connect(
            dbname=self.DB_NAME, 
            user=self.DB_USER,
            password=self.DB_PASS,
            host=self.DB_HOST,
            port=self.DB_PORT
        )
        self.conn.autocommit = True 
        self.cur = self.conn.cursor()
        self.client_qdr = QdrantClient(url=self.QDRANT_ADDRESS, timeout=60)

    def close(self):
        self.conn.close()

    def commit(self):
        """commit changes to database"""
        self.conn.commit()

    def add_region(self, r_id, name):
        sql = """INSERT INTO regions (id, name) VALUES (%s, %s)"""
        try:
            self.cur.execute(sql, (r_id, name))
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def add_public(self, p_id, name, region_id, region_name, city_id, city_name):
        sql = """INSERT INTO publics (id, name, region_id, region_name, city_id, city_name) VALUES (%s, %s, %s, %s, %s, %s)"""
        try:
            self.cur.execute(sql, (p_id, name, region_id, region_name, city_id, city_name))
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def add_posts(self, posts, embeddings):
        try:
            uuids = [str(uuid.uuid4()) for i in range(len(posts))]
            self.client_qdr.upsert(
                        collection_name=self.QDRANT_COLLECTION,
                        points=models.Batch(
                            ids=uuids, payloads=posts, vectors=embeddings
                        ))
        except Exception as e:
            print(e)

    def add_last_upds(self, last_upds):
        sql0 = """DELETE FROM last_upds WHERE public_id = %s AND update_date < %s;"""
        sql = """INSERT INTO last_upds (public_id, update_date) VALUES (%s, %s);"""
        try:
            self.cur.executemany(sql0, last_upds)
            self.conn.commit()

            self.cur.executemany(sql, last_upds)
            self.conn.commit()
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.conn.rollback()

    def add_embeddings(self, embeddings):
        sql = """INSERT INTO embeddings (post_id, embedding) VALUES (%s, %s);"""
        try:
            self.cur.executemany(sql, embeddings)
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def get_last_upds(self):
        try:
            self.cur.execute("SELECT public_id, update_date FROM last_upds;")
            rows = self.cur.fetchall()
            self.conn.commit()

            return rows
        except Exception as e:
            print(e)
            self.conn.rollback()


def fill_regions():
    import pandas as pd

    db = Database()
    df = pd.read_csv('test_data/cities.csv')

    for i, region in enumerate(list(set(df['region']))):
        db.add_region(i, region)

    db.close()


def fill_publics():
    import pandas as pd

    db = Database()
    df = pd.read_csv('test_data/publics.csv')
    df['RegionID'] = df['RegionID'].fillna(0).astype(int)
    df['CityID'] = df['CityID'].fillna(0).astype(int)

    for i in range(len(df)):
        db.add_public(str(df['OwnerID'].iloc[i]), df['PublicName'].iloc[i], 
                      int(df['RegionID'].iloc[i]), df['RegionName'].iloc[i],
                      int(df['CityID'].iloc[i]), df['CityName'].iloc[i])

    db.close()


def fill_last_upds():
    import pandas as pd

    db = Database()
    df = pd.read_csv('test_data/ids_last_date.txt')
    last_upds = list(zip(df['OwnerID'], df['PostDate']))

    db.add_last_upds(last_upds)

    db.close()

if __name__ == "__main__":
    # fill_last_upds()
    pass