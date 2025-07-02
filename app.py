"""
This module implements a real-time AI assistant using FastAPI, OpenAI's GPT model, Chroma for RAG,
and Supabase for authentication and chat logging.

The assistant supports:
- Voice and text chat
- PDF-based knowledge retrieval (RAG)
- Real-time streaming responses
- Session memory
- Supabase-backed message persistence
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from rag_system import AdvancedRAGSystem
import asyncio
import json
import logging
import os
import uuid

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system
try:
    rag_system = AdvancedRAGSystem(openai_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

# Create FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# Session memory store
user_sessions = {}  # Tracks conversation history by session

@app.get("/")
async def root():
    """Serve the login HTML page."""
    return FileResponse("static/login.html")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    chat_task = None

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            query = payload.get("query", "").strip()
            user_id = payload.get("user_id")

            if not query:
                continue

            if chat_task and not chat_task.done():
                chat_task.cancel()

            if rag_system is None:
                await websocket.send_text(json.dumps({
                    "response": "AI system is not available. Please check the server logs.",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "partial": False
                }))
                continue

            chat_task = asyncio.create_task(
                handle_user_query(query, session_id, user_id, websocket)
            )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

async def handle_user_query(query, session_id, user_id, websocket):
    try:
        # Ensure user exists in users table before inserting message
        user_exists = False
        try:
            res = supabase.table("users").select("id").eq("id", user_id).execute()
            if res.data and len(res.data) > 0:
                user_exists = True
        except Exception as e:
            logger.error(f"Error checking user existence: {e}")
        if not user_exists and user_id:
            try:
                supabase.table("users").insert({"id": user_id}).execute()
                logger.info(f"Inserted new user with id: {user_id}")
            except Exception as e:
                logger.error(f"Error inserting new user: {e}")
        supabase.table("messages").insert({
            "user_id": user_id,
            "message": query,
            "is_ai": False,
            "session_id": session_id
        }).execute()
    except Exception as e:
        logger.error(f"Error storing user message: {e}")

    full_response = ""

    try:
        # ✅ Use RAG system to get answer + sources
        result = await asyncio.get_event_loop().run_in_executor(None, rag_system.get_response, query)
        if isinstance(result, dict):
            answer = result.get("response", "")
            sources = result.get("sources", [])
        else:
            answer, sources = result

        for word in answer.split():
            full_response += word + " "
            await websocket.send_text(json.dumps({
                "response": full_response.strip(),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "partial": True
            }))
            await asyncio.sleep(0.05)

        await websocket.send_text(json.dumps({
            "response": full_response.strip(),
            "sources": sources,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "partial": False
        }))

        try:
            supabase.table("messages").insert({
                "user_id": user_id,
                "message": full_response.strip(),
                "is_ai": True,
                "session_id": session_id
            }).execute()
        except Exception as e:
            logger.error(f"Error storing AI message: {e}")

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        await websocket.send_text(json.dumps({
            "response": "⚠️ AI is having trouble responding.",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "partial": False
        }))
