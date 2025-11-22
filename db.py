# db.py
import sqlite3
import os
from datetime import datetime

DB_PATH = "predictions.db"
print("DB absolute path:", os.path.abspath(DB_PATH))

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            prediction INTEGER,
            probability REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(url, prediction, probability):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (url, prediction, probability, timestamp)
        VALUES (?, ?, ?, ?)
    """, (url, prediction, probability, datetime.now().isoformat()))
    conn.commit()
    print("Saved prediction for URL:", url)
    conn.close()
