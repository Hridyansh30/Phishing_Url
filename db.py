import os
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = os.getenv("DATABASE_URL")

def get_connection():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            url TEXT,
            prediction INTEGER,
            probability DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_prediction(url, prediction, probability):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (url, prediction, probability)
        VALUES (%s, %s, %s);
    """, (url, prediction, probability))
    conn.commit()
    cur.close()
    conn.close()
