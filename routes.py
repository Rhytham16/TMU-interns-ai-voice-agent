"""
API routes module for Budger AI Voice Assistant
Handles all HTTP endpoints and WebSocket connections
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from database import DatabaseManager
from models import (
    QueryRequest, QueryResponse, DocumentUploadRequest, 
    SignupRequest, LoginRequest
)
from rag_system import OptimizedRAGSystem

logger = logging.getLogger(__name__)

class APIRoutes:
    def __init__(self, db_manager: DatabaseManager, rag_system: OptimizedRAGSystem):
        self.db_manager = db_manager
        self.rag_system = rag_system
        self.active_connections: List[WebSocket] = []
        self.user_sessions: Dict[str, Any] = {}

    def read_root(self):
        """Root endpoint - serves login page"""
        return FileResponse("static/login.html")

    def read_home(self):
        """Home endpoint - serves chat page"""
        return FileResponse("static/chat.html")

    async def health_check(self):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model": "gpt-3.5-turbo",
            "timestamp": datetime.now().isoformat()
        }

    async def chat_endpoint(self, request: QueryRequest):
        """Standard chat endpoint (non-streaming)"""
        result = await self.rag_system.get_response(
            query=request.query, 
            session_id=request.session_id
        )
        
        # Save to database
        user = self.db_manager.get_user_by_session(request.session_id)
        if user:
            self.db_manager.save_chat_history(
                user["id"], 
                request.session_id, 
                request.query, 
                result["response"]
            )
        
        return QueryResponse(**result)

    async def chat_stream_endpoint(self, request: QueryRequest):
        """Streaming chat endpoint"""
        async def generate():
            async for chunk in self.rag_system.stream_response(
                request.query, 
                request.session_id
            ):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(generate(), media_type="text/plain")

    async def signup_user(self, request: SignupRequest):
        """User registration endpoint"""
        result = self.db_manager.create_user(
            request.username, 
            request.email, 
            request.password
        )
        
        if result["success"]:
            return {"message": result["message"]}
        else:
            return JSONResponse(
                status_code=400, 
                content={"error": result["message"]}
            )

    async def login_user(self, request: LoginRequest):
        """User login endpoint"""
        user = self.db_manager.authenticate_user(
            request.login_id, 
            request.password
        )
        
        if user:
            return {
                "message": "Login successful", 
                "username": user["username"]
            }
        else:
            return JSONResponse(
                status_code=401, 
                content={"error": "Invalid username/email or password."}
            )

    async def delete_session(self, session_id: str):
        """Clear session data endpoint"""
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]
        
        self.rag_system.clear_session_history(session_id)
        
        return {"message": f"Session {session_id} cleared."}

    async def websocket_endpoint(self, websocket: WebSocket, session_id: str):
        """Real-time WebSocket endpoint with streaming"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                data = await websocket.receive_text()
                data_json = json.loads(data)
                
                query = data_json.get("query", "")
                stream = data_json.get("stream", True)
                
                if stream:
                    # Send streaming response
                    async for chunk in self.rag_system.stream_response(query, session_id):
                        await websocket.send_text(chunk)
                else:
                    # Send complete response
                    result = await self.rag_system.get_response(query, session_id)
                    await websocket.send_text(json.dumps(result))
                    
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def debug_context(self, query: str):
        """Debug endpoint to see what context is returned for a query"""
        context_texts, sources = await self.rag_system.get_context_documents(query)
        return {
            "context_texts": context_texts, 
            "sources": sources, 
            "count": len(context_texts)
        }

    async def startup_event(self):
        """Startup event handler"""
        logger.info("Budger AI Assistant started with OpenAI RAG optimization")
        logger.info(f"Active model: gpt-3.5-turbo")
        logger.info("Real-time streaming enabled")
        self.rag_system.log_vectorstore_stats()