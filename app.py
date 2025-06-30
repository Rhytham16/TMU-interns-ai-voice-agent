from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse  # Add this import
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI key loaded:", openai_api_key[:10], "****")


if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# FastAPI app setup
app = FastAPI(title="Budger AI Voice Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
active_connections: List[WebSocket] = []
user_sessions: Dict[str, Any] = {}

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    language: str = "en"

class QueryResponse(BaseModel):
    response: str
    sources: List[str] = []
    session_id: str
    timestamp: str

class DocumentUploadRequest(BaseModel):
    file_path: str
    collection_name: str = "default"

# Enhanced RAG Setup
class AdvancedRAGSystem:
    def __init__(self):
        self.persist_dir = "enhanced_chroma_store"
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
            openai_api_key=openai_api_key,
            streaming=True
        )
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.setup_vectorstore()
        self.setup_qa_chain()

    def setup_vectorstore(self):
        """Initialize or load existing vector store"""
        if os.path.exists(self.persist_dir):
            logger.info("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            logger.info("Creating new vector store...")
            # Create empty vectorstore
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def add_documents(self, file_path: str, collection_name: str = "default"):
        """Add documents to vector store"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                # Support for directory of documents
                loader = DirectoryLoader(file_path, glob="**/*.pdf")
            
            documents = loader.load()
            
            # Enhanced text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            splits = text_splitter.split_documents(documents)
            
            # Add collection metadata
            for split in splits:
                split.metadata["collection"] = collection_name
                split.metadata["timestamp"] = datetime.now().isoformat()
            
            # Add to vectorstore
            self.vectorstore.add_documents(splits)
            self.vectorstore.persist()
            
            logger.info(f"Added {len(splits)} document chunks to collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    def setup_qa_chain(self):
        """Setup conversational retrieval chain with custom prompts"""
        
        # Enhanced system prompt for customer service
        system_template = """You are Budger, an advanced AI customer service agent for Cogent Infotech Corporation. You are helpful, professional, and knowledgeable.

INSTRUCTIONS:
1. Always maintain a friendly, professional tone
2. If the user speaks in Hindi (Roman or Devanagari), respond in the same script/style
3. Use the provided context to answer questions accurately about Cogent Infotech services
4. If you don't know something from the context, say so politely
5. For customer service inquiries, be empathetic and solution-oriented
6. Keep responses concise but complete
7. Always aim to resolve customer issues effectively
8. Focus on Cogent Infotech's services: Digital Transformation, Workforce Solutions, Analytics & AI/ML, Application Development, Cloud Solutions, Cybersecurity

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{chat_history}

Remember: Match the user's language preference and provide helpful, accurate responses based on Cogent Infotech's knowledge base."""

        human_template = """User Query: {question}

Please provide a helpful response based on the context and conversation history about Cogent Infotech's services."""

        # Create prompt templates
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

        # Setup memory for conversation
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )

        # Create conversational retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": chat_prompt},
            return_source_documents=True,
            verbose=True,
            output_key="answer"
        )

    async def get_response(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Get AI response with source documents"""
        try:
            # Create session-specific memory if not exists
            if session_id not in user_sessions:
                user_sessions[session_id] = {
                    "memory": ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        k=10
                    ),
                    "created_at": datetime.now().isoformat()
                }

            # Get response from chain
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.qa_chain.invoke,
                {"question": query, "chat_history": []}  # add chat_history
            )

            

            # Extract sources
            sources = []
            if result.get("source_documents"):
                sources = [
                    f"Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('source', 'Unknown')}"
                    for doc in result["source_documents"][:3]  # Top 3 sources
                ]

            return {
                "response": result["answer"],
                "sources": sources,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                "sources": [],
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

# Initialize RAG system
rag_system = AdvancedRAGSystem()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# API Endpoints

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse('static/index.html')

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(req: QueryRequest):
    """Enhanced chat endpoint with session management"""
    try:
        result = await rag_system.get_response(req.query, req.session_id)
        
        return QueryResponse(
            response=result["response"],
            sources=result["sources"],
            session_id=result["session_id"],
            timestamp=result["timestamp"]
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-documents")
async def upload_documents(req: DocumentUploadRequest):
    """Upload and process documents for RAG"""
    try:
        success = rag_system.add_documents(req.file_path, req.collection_name)
        
        if success:
            return {"message": f"Documents successfully added to collection '{req.collection_name}'"}
        else:
            raise HTTPException(status_code=400, detail="Failed to process documents")
            
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_active_sessions():
    """Get all active user sessions"""
    return {
        "active_sessions": len(user_sessions),
        "sessions": {k: v["created_at"] for k, v in user_sessions.items()}
    }

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear specific user session"""
    if session_id in user_sessions:
        del user_sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process the query
            query = message_data.get("query", "")
            if query:
                result = await rag_system.get_response(query, session_id)
                
                # Send response back to client
                await manager.send_personal_message(
                    json.dumps(result), 
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket connection closed for session: {session_id}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vectorstore_ready": rag_system.vectorstore is not None,
        "active_sessions": len(user_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)