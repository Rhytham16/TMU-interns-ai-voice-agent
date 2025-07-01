from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Form
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
from supabase import create_client, Client  

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain

def setup_logger():
    """
    Sets up logging configuration with rotating file and console handlers.

    Logs are saved in the 'logs' directory with rotation every midnight.
    """
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

    # Optional: Console log output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ChromaEmbeddingFunction:
    """Wraps LangChain embeddings to be compatible with Chroma's expected function signature"""
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(input)

setup_logger()
logger = logging.getLogger(__name__)

# Load environment variables first
load_dotenv()

# Validate required environment variables
required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Get environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print("OpenAI key loaded:", openai_api_key[:10] + "****" if openai_api_key else "Not found")
print("Supabase URL:", SUPABASE_URL[:20] + "****" if SUPABASE_URL else "Not found")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    supabase = None

app = FastAPI(title="Budger AI Voice Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

active_connections: List[WebSocket] = []
user_sessions: Dict[str, Any] = {}

class QueryRequest(BaseModel):
    """Model for user query requests."""
    query: str
    session_id: str = "default"
    language: str = "en"
    user_id: str = None

class QueryResponse(BaseModel):
    """Model for chatbot response."""
    response: str
    sources: List[str] = []
    session_id: str
    timestamp: str

class DocumentUploadRequest(BaseModel):
    """Model for document upload requests."""
    file_path: str
    collection_name: str = "default"

class AdvancedRAGSystem:
    """
    Advanced Retrieval-Augmented Generation (RAG) system for handling user queries with document context.
    """

    def __init__(self):
        """Initializes RAG system, sets up vector store and QA chain."""
        self.persist_dir = "enhanced_chroma_store"
        try:
            self.embeddings = ChromaEmbeddingFunction(OpenAIEmbeddings(openai_api_key=openai_api_key))
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.7,
                openai_api_key=openai_api_key,
                streaming=True
            )
            logger.info("OpenAI models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI models: {str(e)}")
            raise
            
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.setup_vectorstore()
        self.setup_qa_chain()

    def setup_vectorstore(self):
        """Sets up the vector store either by loading existing or creating a new one."""
        try:
            if os.path.exists(self.persist_dir):
                logger.info("Loading existing vector store...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
            else:
                logger.info("Creating new vector store...")
                os.makedirs(self.persist_dir, exist_ok=True)
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            logger.info("Vector store setup completed")
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    def add_documents(self, file_path: str, collection_name: str = "default") -> bool:
        """
        Adds documents to the vector store after splitting into chunks.

        Args:
            file_path (str): Path to the PDF file or directory.
            collection_name (str): Name of the collection to associate.

        Returns:
            bool: True if documents were added successfully, False otherwise.
        """
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = DirectoryLoader(file_path, glob="**/*.pdf")

            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            splits = text_splitter.split_documents(documents)

            for split in splits:
                split.metadata["collection"] = collection_name
                split.metadata["timestamp"] = datetime.now().isoformat()

            self.vectorstore.add_documents(splits)
            self.vectorstore.persist()

            logger.info(f"Added {len(splits)} document chunks to collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    def setup_qa_chain(self):
        """Sets up the Conversational Retrieval QA chain with system and human prompts."""
        try:
            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
            You are Budger, an advanced AI customer service agent for Cogent Infotech Corporation.

            Answer the question using ONLY the context provided. If the context doesn't contain relevant information, politely say so and ask for clarification.

            Context:
            {context}

            Question:
            {question}

            Helpful Answer:
            """
            )

            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10,
                output_key="answer"
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                return_source_documents=True,
                verbose=True,
                output_key="answer"
            )
            logger.info("QA chain setup completed")
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise

        
    async def get_response(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Processes a user query and returns a response with sources.

        Args:
            query (str): The user's query.
            session_id (str): Session identifier.

        Returns:
            Dict[str, Any]: Dictionary containing response, sources, session ID, and timestamp.
        """
        try:
            if session_id not in user_sessions:
                user_sessions[session_id] = {
                    "memory": ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        k=10,
                        output_key="answer"
                    ),
                    "created_at": datetime.now().isoformat()
                }

            # Check if we have documents in the vector store
            if not self.vectorstore or self.vectorstore._collection.count() == 0:
                return {
                    "response": "I don't have any documents loaded yet. Please load the knowledge base first using the load_documents.py script.",
                    "sources": [],
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }

            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.qa_chain.invoke,
                {"question": query, "chat_history": []}
            )

            sources = []
            if result.get("source_documents"):
                sources = [
                    f"Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('source', 'Unknown')}"
                    for doc in result["source_documents"][:3]
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
try:
    rag_system = AdvancedRAGSystem()
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}")
    rag_system = None

@app.get("/")
async def serve_frontend():
    """Serve the login page"""
    return FileResponse("static/login.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "Budger AI is running",
        "timestamp": datetime.now().isoformat(),
        "rag_system": "initialized" if rag_system else "failed",
        "supabase": "connected" if supabase else "disconnected"
    }

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """HTTP endpoint for chat when WebSocket fails"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        response = await rag_system.get_response(
            request.query, 
            session_id=request.session_id
        )
        return QueryResponse(**response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    # Use Supabase to validate user credentials
    result = supabase.table("users").select("*").eq("username", username).execute()
    user = result.data[0] if result.data else None

    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"status": "success", "user_id": user["id"]}

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if session_id in user_sessions:
        del user_sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected for session: {session_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            query = data_json.get("query", "")
            language = data_json.get("language", "en")
            user_id = data_json.get("user_id", None)

            if not query.strip():
                await websocket.send_text(json.dumps({
                    "response": "Please provide a valid query.",
                    "sources": [],
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }))
                continue

            # Store user message in Supabase if available
            if user_id and supabase:
                try:
                    supabase.table("messages").insert({
                        "user_id": user_id,
                        "message": query,
                        "is_ai": False,
                        "session_id": session_id
                    }).execute()
                except Exception as e:
                    logger.error(f"Error storing user message: {str(e)}")

            # Get AI response
            if rag_system:
                response = await rag_system.get_response(query, session_id=session_id)
            else:
                response = {
                    "response": "AI system is not available. Please check the server logs.",
                    "sources": [],
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }

            # Store AI response in Supabase if available
            if user_id and supabase:
                try:
                    supabase.table("messages").insert({
                        "user_id": user_id,
                        "message": response["response"],
                        "is_ai": True,
                        "session_id": session_id
                    }).execute()
                except Exception as e:
                    logger.error(f"Error storing AI message: {str(e)}")

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("Starting AI Assistant...")
    print("Visit: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
