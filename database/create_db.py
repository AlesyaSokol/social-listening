import psycopg2
from psycopg2 import sql
import os
import dotenv

dotenv.load_dotenv('../.env')

#psql postgres
# create database social_listening;
# create user test with login password 'testtest';

DB_NAME = os.getenv('POSTGRES_DB')  
DB_USER = os.getenv('POSTGRES_USER')   
DB_PASS = os.getenv('POSTGRES_PASSWORD') 
DB_HOST = os.getenv('POSTGRES_HOST')  
DB_PORT = os.getenv('POSTGRES_PORT')   

def create_database():
    try:
        # Connect to PostgreSQL (default database, usually 'postgres')
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Check if the database exists
        cur.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s;"), [DB_NAME])
        if cur.fetchone():
            print(f"Database '{DB_NAME}' already exists.")
        else:
            print(f"Creating database '{DB_NAME}'...")

            cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(DB_NAME)))
            print(f"Database '{DB_NAME}' created successfully.")

        with open('create_tables.sql', 'r') as file:
            sql_script = file.read()
        cur.execute(sql.SQL(sql_script))
        
        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error while creating database: {e}")


def install_pgvector_extension():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()

        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("The 'pgvector' extension is already installed.")
        else:
            print("Installing 'pgvector' extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("The 'pgvector' extension has been installed successfully.")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error while installing pgvector: {e}")

# Main function to run the entire process
def main():
    create_database()
    install_pgvector_extension()


# Run the main function
if __name__ == "__main__":
    main()