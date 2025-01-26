import psycopg2
import os
import dotenv

dotenv.load_dotenv('../.env')

class Database:
    DB_NAME = os.getenv('POSTGRES_DB')  
    DB_USER = os.getenv('POSTGRES_USER')   
    DB_PASS = os.getenv('POSTGRES_PASSWORD') 
    DB_HOST = os.getenv('POSTGRES_HOST')  
    DB_PORT = os.getenv('POSTGRES_PORT')   

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

    def add_public(self, p_id, name, region):
        sql = """INSERT INTO publics (id, name, region) VALUES (%s, %s, %s)"""
        try:
            self.cur.execute(sql, (p_id, name, region))
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def add_posts(self, posts):
        sql = """INSERT INTO posts (public_id, post_text, post_date, post_id, views, reposts, likes, comments) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"""
        try:
            self.cur.executemany(sql, posts)
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def add_last_upds(self, last_upds):
        sql = """INSERT INTO last_upds (public_id, update_date) VALUES (%s, %s);"""
        try:
            self.cur.executemany(sql, last_upds)
            self.conn.commit()
        except Exception as e:
            print(e)
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