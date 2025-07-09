import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, List, AsyncGenerator
import time

from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
import httpx

# ───────────────────────────────────────────────
# Logger Setup
# ───────────────────────────────────────────────

def setup_logger():
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

setup_logger()
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# Environment
# ───────────────────────────────────────────────

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

print("OPENAI API key loaded:", openai_api_key[:10], "****")

# ───────────────────────────────────────────────
# FastAPI App
# ───────────────────────────────────────────────

app = FastAPI(title="Budger AI Voice Assistant - Optimized", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

active_connections: List[WebSocket] = []
user_sessions: Dict[str, Any] = {}

# ───────────────────────────────────────────────
# SQLite Setup with Connection Pool
# ───────────────────────────────────────────────

class DatabaseManager:
    def __init__(self, db_path: str = "budger_users.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

db_manager = DatabaseManager()

# ───────────────────────────────────────────────
# Pydantic Models
# ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    language: str = "en"
    stream: bool = True

class QueryResponse(BaseModel):
    response: str
    sources: List[str] = []
    session_id: str
    timestamp: str
    response_time: float = 0.0

class DocumentUploadRequest(BaseModel):
    file_path: str
    collection_name: str = "default"

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    login_id: str
    password: str

# ───────────────────────────────────────────────
# Optimized RAG System with GPT-3.5 Turbo
# ───────────────────────────────────────────────

class OptimizedRAGSystem:
    def __init__(self):
        self.persist_dir = "enhanced_chroma_store"
        # Use OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        # Use OpenAI LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )
        self.vectorstore = None
        self.retriever = None
        self.conversation_histories = {}
        self.setup_vectorstore()
        self.log_vectorstore_stats()

    def setup_vectorstore(self):
        """Setup vector store with optimized settings"""
        try:
            client_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                client_settings=client_settings
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
        except Exception as e:
            logger.error(f"Error setting up vectorstore: {str(e)}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def log_vectorstore_stats(self):
        """Log statistics about the vectorstore"""
        try:
            if self.vectorstore is not None:
                count = len(self.vectorstore.get()['ids'])
                logger.info(f"Vectorstore contains {count} documents.")
            else:
                logger.warning("Vectorstore is not initialized.")
        except Exception as e:
            logger.error(f"Error getting vectorstore stats: {str(e)}")

    async def add_documents_async(self, file_path: str, collection_name: str = "default") -> bool:
        """Async document addition for better performance"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = DirectoryLoader(file_path, glob="**/*.pdf")
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            for split in splits:
                split.metadata["collection"] = collection_name
                split.metadata["timestamp"] = datetime.now().isoformat()
            batch_size = 50
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                await asyncio.get_event_loop().run_in_executor(
                    None, self.vectorstore.add_documents, batch
                )
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.vectorstore.persist
                )
            except AttributeError:
                pass
            logger.info(f"Added {len(splits)} document chunks to collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for session"""
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []
        return self.conversation_histories[session_id]

    def update_conversation_history(self, session_id: str, user_query: str, ai_response: str):
        """Update conversation history"""
        history = self.get_conversation_history(session_id)
        history.append({
            "user": user_query,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges for performance
        if len(history) > 10:
            self.conversation_histories[session_id] = history[-10:]

    async def get_context_documents(self, query: str) -> tuple[List[str], List[str]]:
        """Retrieve relevant documents asynchronously"""
        try:
            if self.vectorstore is None:
                logger.error("Vectorstore is not initialized!")
                return [], []
            docs = await asyncio.get_event_loop().run_in_executor(
                None, self.retriever.get_relevant_documents, query
            )
            if not docs:
                logger.warning("No relevant documents found for query.")
            context_texts = []
            sources = []
            for doc in docs:
                context_texts.append(doc.page_content)
                source_info = f"Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('source', 'Unknown')}"
                sources.append(source_info)
            return context_texts, sources
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return [], []

    async def stream_response(self, query: str, session_id: str = "default") -> AsyncGenerator[str, None]:
        """Stream response from gpt-3.5-turbo with enhanced context and conversation history"""
        start_time = time.time()
        
        try:
            # Get conversation history
            history = self.get_conversation_history(session_id)
            
            # Get relevant context
            context_texts, sources = await self.get_context_documents(query)
            context = "\n\n".join(context_texts) if context_texts else "No relevant context found."
            
            # Build conversation context
            conversation_context = ""
            if history:
                recent_history = history[-3:]  # Last 3 exchanges
                for item in recent_history:
                    conversation_context += f"User: {item['user']}\nAssistant: {item['assistant']}\n\n"
            
            # Enhanced system prompt for Budger
            system_prompt = """You are Budger, an advanced AI customer service agent for Cogent Infotech Corporation. You are helpful, professional, and knowledgeable about the company's services and policies.

Key Guidelines:
- Provide accurate, helpful responses based on the context provided
- Be conversational and friendly while maintaining professionalism
- If you don't know something, admit it rather than guessing
- Keep responses concise but comprehensive
- Use the conversation history to maintain context

Previous Conversation:
{conversation_context}

Relevant Context:
{context}

Current User Query: {query}

Please provide a helpful and accurate response:"""

            prompt = system_prompt.format(
                conversation_context=conversation_context,
                context=context,
                query=query
            )
            
            # Stream response from OpenAI
            response = await self.llm.ainvoke(prompt)
            full_response = response.content

            # Stream the response word by word for real-time effect
            words = full_response.split()
            streamed_response = ""
            
            for i, word in enumerate(words):
                streamed_response += word + " "
                
                # Send partial response
                yield json.dumps({
                    "type": "partial",
                    "content": word + " ",
                    "full_response": streamed_response.strip(),
                    "sources": sources,
                    "session_id": session_id
                })
                
                # Small delay for streaming effect
                await asyncio.sleep(0.05)
            
            # Update conversation history
            self.update_conversation_history(session_id, query, full_response)
            
            response_time = time.time() - start_time
            
            # Send final response
            yield json.dumps({
                "type": "complete",
                "content": full_response,
                "sources": sources,
                "session_id": session_id,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield json.dumps({
                "type": "error",
                "content": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                "sources": [],
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })

    async def get_response(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Get complete response (non-streaming)"""
        start_time = time.time()
        
        try:
            # Get conversation history
            history = self.get_conversation_history(session_id)
            
            # Get relevant context
            context_texts, sources = await self.get_context_documents(query)
            context = "\n\n".join(context_texts) if context_texts else "No relevant context found."
            
            # Build conversation context
            conversation_context = ""
            if history:
                recent_history = history[-3:]
                for item in recent_history:
                    conversation_context += f"User: {item['user']}\nAssistant: {item['assistant']}\n\n"
            
            system_prompt = """You are Budger, an advanced AI customer service agent for Cogent Infotech Corporation. You are helpful, professional, and knowledgeable about the company's services and policies.

Previous Conversation:
{conversation_context}

Relevant Context:
{context}

Current User Query: {query}

Please provide a helpful and accurate response:"""

            prompt = system_prompt.format(
                conversation_context=conversation_context,
                context=context,
                query=query
            )
            
            response = await self.llm.ainvoke(prompt)
            full_response = response.content

            
            # Update conversation history
            self.update_conversation_history(session_id, query, full_response)
            
            response_time = time.time() - start_time
            
            return {
                "response": full_response,
                "sources": sources,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                "sources": [],
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "response_time": time.time() - start_time
            }

# Initialize optimized RAG system
rag_system = OptimizedRAGSystem()

# ───────────────────────────────────────────────
# API Routes
# ───────────────────────────────────────────────

@app.get("/")
def read_root():
    return FileResponse("static/login.html")

@app.get("/home")
def read_root():
    return FileResponse("static/chat.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "gemini-1.5-flash",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """Standard chat endpoint (non-streaming)"""
    result = await rag_system.get_response(query=request.query, session_id=request.session_id)
    
    # Save to database
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (request.session_id, request.session_id))
    user = cursor.fetchone()
    
    if user:
        cursor.execute(
            "INSERT INTO chat_history (user_id, session_id, query, response) VALUES (?, ?, ?, ?)",
            (user["id"], request.session_id, request.query, result["response"])
        )
        conn.commit()
    
    conn.close()
    
    return QueryResponse(**result)

@app.post("/chat/stream")
async def chat_stream_endpoint(request: QueryRequest):
    """Streaming chat endpoint"""
    async def generate():
        async for chunk in rag_system.stream_response(request.query, request.session_id):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/upload")
async def upload_document(request: DocumentUploadRequest):
    """Upload and process documents"""
    success = await rag_system.add_documents_async(
        file_path="C:/Users/Asus/OneDrive/Desktop/TMU-interns-ai-voice-agent-gpt3.5-version/data/sales_ai_knowledgebase.pdf",
        collection_name=request.collection_name
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add documents.")
    return {"message": "Documents added successfully."}

@app.post("/signup")
async def signup_user(request: SignupRequest):
    """User registration"""
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (request.username, request.email, request.password)
        )
        conn.commit()
        return {"message": "User registered successfully."}
    except sqlite3.IntegrityError:
        return JSONResponse(status_code=400, content={"error": "Username or email already exists."})
    finally:
        conn.close()

@app.post("/login")
async def login_user(request: LoginRequest):
    """User login"""
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (request.login_id, request.login_id))
    user = cursor.fetchone()
    conn.close()
    
    if user and user["password"] == request.password:
        return {"message": "Login successful", "username": user["username"]}
    else:
        return JSONResponse(status_code=401, content={"error": "Invalid username/email or password."})

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear session data"""
    if session_id in user_sessions:
        del user_sessions[session_id]
    if session_id in rag_system.conversation_histories:
        del rag_system.conversation_histories[session_id]
    return {"message": f"Session {session_id} cleared."}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Real-time WebSocket endpoint with streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            query = data_json.get("query", "")
            stream = data_json.get("stream", True)
            
            if stream:
                # Send streaming response
                async for chunk in rag_system.stream_response(query, session_id):
                    await websocket.send_text(chunk)
            else:
                # Send complete response
                result = await rag_system.get_response(query, session_id)
                await websocket.send_text(json.dumps(result))
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/debug/context")
async def debug_context(query: str):
    """Debug endpoint to see what context is returned for a query."""
    context_texts, sources = await rag_system.get_context_documents(query)
    return {"context_texts": context_texts, "sources": sources, "count": len(context_texts)}

# ───────────────────────────────────────────────
# Startup Event
# ───────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Budger AI Assistant started with OpenAI RAG optimization")
    logger.info(f"Active model: gpt-3.5-turbo")
    logger.info("Real-time streaming enabled")
    rag_system.log_vectorstore_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)