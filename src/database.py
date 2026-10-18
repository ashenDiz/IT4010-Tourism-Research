import os
import sqlite3
import hashlib
import json
import pandas as pd

DB_PATH = os.path.join('data', 'tourism.db')

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    """Hashes a password with SHA-256 and a standard domain salt."""
    salt = "SLIIT_IT4010_TOURISM_SALT"
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def init_db():
    """Creates the database schema if tables do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Tourist',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Live Reviews Audit Log Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT DEFAULT 'Guest',
            destination TEXT,
            review_text TEXT NOT NULL,
            predicted_sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            aspects_detected TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. User Favorites Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            destination TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, destination)
        )
    """)

    # Seed default admin user if users table is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        admin_pass = hash_password("admin123")
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", admin_pass, "Tourism Officer")
        )
        # Also seed a sample tourist user
        tourist_pass = hash_password("tourist123")
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("traveler_sam", tourist_pass, "Tourist")
        )

    conn.commit()
    conn.close()

def register_user(username, password, role="Tourist"):
    """Registers a new user account."""
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    conn = get_connection()
    cursor = conn.cursor()
    try:
        p_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, p_hash, role)
        )
        conn.commit()
        return True, f"Account '{username}' successfully registered! You can now log in."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists. Please choose a different one."
    finally:
        conn.close()

def authenticate_user(username, password):
    """Verifies user credentials and returns user details if valid."""
    username = username.strip().lower()
    p_hash = hash_password(password)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, role, created_at FROM users WHERE username = ? AND password_hash = ?",
        (username, p_hash)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'id': row['id'],
            'username': row['username'],
            'role': row['role'],
            'created_at': row['created_at']
        }
    return None

def log_review(username, destination, review_text, predicted_sentiment, confidence, aspects=None):
    """Logs a live review classification into the SQLite audit table."""
    conn = get_connection()
    cursor = conn.cursor()
    
    aspects_json = json.dumps(aspects) if aspects else "[]"
    cursor.execute("""
        INSERT INTO reviews_log (username, destination, review_text, predicted_sentiment, confidence, aspects_detected)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (username or 'Guest', destination or 'General', review_text, predicted_sentiment, round(confidence, 2), aspects_json))
    
    conn.commit()
    conn.close()

def get_recent_reviews(limit=25):
    """Fetches recently analyzed reviews from SQLite as a DataFrame."""
    conn = get_connection()
    query = """
        SELECT id, username, destination, review_text, predicted_sentiment, confidence, created_at 
        FROM reviews_log 
        ORDER BY id DESC 
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df

def add_favorite(user_id, destination):
    """Bookmarks a destination for a logged-in user."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO user_favorites (user_id, destination) VALUES (?, ?)",
            (user_id, destination)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def get_favorites(user_id):
    """Retrieves all bookmarked destinations for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT destination, created_at FROM user_favorites WHERE user_id = ? ORDER BY id DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [r['destination'] for r in rows]

def remove_favorite(user_id, destination):
    """Removes a bookmarked destination."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM user_favorites WHERE user_id = ? AND destination = ?",
        (user_id, destination)
    )
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("SQLite database initialized successfully at:", DB_PATH)
