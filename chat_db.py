import sqlite3
from datetime import datetime
from config import CHAT_DB_PATH

def init_db():
    conn = sqlite3.connect(CHAT_DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            collection TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            PRIMARY KEY (collection, timestamp)
        )
    ''')
    conn.commit()
    conn.close()

def save_message(collection: str, role: str, content: str):
    init_db()
    conn = sqlite3.connect(CHAT_DB_PATH)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute(
        "INSERT OR REPLACE INTO chats (collection, timestamp, role, content) VALUES (?, ?, ?, ?)",
        (collection, ts, role, content)
    )
    conn.commit()
    conn.close()

def load_chat_history(collection: str) -> list:
    init_db()
    conn = sqlite3.connect(CHAT_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chats WHERE collection = ? ORDER BY timestamp ASC", (collection,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def clear_chat_history(collection: str):
    init_db()
    conn = sqlite3.connect(CHAT_DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE collection = ?", (collection,))
    conn.commit()
    conn.close()