"""
Configuration module for Budger AI Voice Assistant
Handles environment variables, logging setup, and application configuration
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

def setup_logger():
    """Setup application logger with file rotation"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")

    handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
    )
    handler.suffix = "%Y-%m-%d"
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def load_environment():
    """Load environment variables and validate required keys"""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("OPENAI API key loaded:", openai_api_key[:10], "****")
    return openai_api_key

class AppConfig:
    """Application configuration class"""
    def __init__(self):
        self.openai_api_key = load_environment()
        self.title = "Budger AI Voice Assistant - Optimized"
        self.version = "3.0.0"
        self.db_path = "budger_users.db"
        self.persist_dir = "enhanced_chroma_store"
        self.host = "0.0.0.0"
        self.port = 8000
        
        # CORS settings
        self.cors_origins = ["*"]
        self.cors_credentials = True
        self.cors_methods = ["*"]
        self.cors_headers = ["*"]
        
        # Static files
        self.static_dir = "static"
        
        # Model settings
        self.model_name = "gpt-3.5-turbo"
        self.temperature = 0.7
        self.streaming = True
        
        # RAG settings
        self.chunk_size = 800
        self.chunk_overlap = 100
        self.search_k = 3
        self.batch_size = 50
        self.max_history = 10
        self.recent_history = 3