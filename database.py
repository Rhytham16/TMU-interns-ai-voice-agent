"""
Database module for Budger AI Voice Assistant
Handles SQLite database operations with connection pooling
"""

import sqlite3
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "budger_users.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT,
                query TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def create_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Create a new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password)
            )
            conn.commit()
            return {"success": True, "message": "User registered successfully."}
        except sqlite3.IntegrityError:
            return {"success": False, "message": "Username or email already exists."}
        finally:
            conn.close()
    
    def authenticate_user(self, login_id: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (login_id, login_id))
        user = cursor.fetchone()
        conn.close()
        
        if user and user["password"] == password:
            return {"username": user["username"], "email": user["email"], "id": user["id"]}
        return None
    
    def save_chat_history(self, user_id: int, session_id: str, query: str, response: str) -> bool:
        """Save chat history to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO chat_history (user_id, session_id, query, response) VALUES (?, ?, ?, ?)",
                (user_id, session_id, query, response)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_user_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get user by session ID (username or email)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (session_id, session_id))
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None