"""
Modular Budger AI Voice Assistant
Main application file that integrates all modules
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import custom modules
from config import setup_logger, AppConfig
from database import DatabaseManager
from rag_system import OptimizedRAGSystem
from routes import APIRoutes

# Initialize logging
setup_logger()

# Load configuration
config = AppConfig()

# Initialize components
db_manager = DatabaseManager(config.db_path)
rag_system = OptimizedRAGSystem()
routes = APIRoutes(db_manager, rag_system)

# Create FastAPI app
app = FastAPI(title=config.title, version=config.version)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_credentials,
    allow_methods=config.cors_methods,
    allow_headers=config.cors_headers,
)

# Mount static files
app.mount("/static", StaticFiles(directory=config.static_dir), name="static")

# Register routes
app.get("/")(routes.read_root)
app.get("/home")(routes.read_home)
app.get("/health")(routes.health_check)
app.post("/chat")(routes.chat_endpoint)
app.post("/chat/stream")(routes.chat_stream_endpoint)
app.post("/signup")(routes.signup_user)
app.post("/login")(routes.login_user)
app.delete("/sessions/{session_id}")(routes.delete_session)
app.websocket("/ws/{session_id}")(routes.websocket_endpoint)
app.get("/debug/context")(routes.debug_context)

# Startup event
app.on_event("startup")(routes.startup_event)

if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)