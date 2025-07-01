import asyncio
import json
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, List

from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

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

print("OpenAI key loaded:", openai_api_key[:10], "****")

# ───────────────────────────────────────────────
# FastAPI App
# ───────────────────────────────────────────────

app = FastAPI(title="Budger AI Voice Assistant", version="2.0.0")

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
# Pydantic Models
# ───────────────────────────────────────────────


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

# ───────────────────────────────────────────────
# Advanced RAG System
# ───────────────────────────────────────────────


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
        client_settings = Settings(anonymized_telemetry=False)

        if os.path.exists(self.persist_dir):
            logger.info("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                client_settings=client_settings
            )
        else:
            logger.info("Creating new vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                client_settings=client_settings
            )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def add_documents(self, file_path: str, collection_name: str = "default") -> bool:
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

            logger.info(
                f"Added {len(splits)} document chunks to collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    def setup_qa_chain(self):
        system_template = """You are Budger, an advanced AI customer service agent for Cogent Infotech Corporation..."""

        human_template = """Context:\n{context}\n\nUser Query: {question}\n\nPlease provide a helpful response..."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template)

        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

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
            combine_docs_chain_kwargs={"prompt": chat_prompt},
            return_source_documents=True,
            verbose=True,
            output_key="answer"
        )

    async def get_response(self, query: str, session_id: str = "default") -> Dict[str, Any]:
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


rag_system = AdvancedRAGSystem()

# ───────────────────────────────────────────────
# API Routes
# ───────────────────────────────────────────────


@app.get("/")
def read_root():
    return FileResponse("static/chat.html")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    result = await rag_system.get_response(
        query=request.query,
        session_id=request.session_id
    )
    return QueryResponse(**result)


@app.post("/upload")
def upload_document(request: DocumentUploadRequest):
    success = rag_system.add_documents(
        file_path=request.file_path,
        collection_name=request.collection_name
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add documents.")
    return {"message": "Documents added successfully."}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in user_sessions:
        del user_sessions[session_id]
        return {"message": f"Session {session_id} cleared."}
    else:
        return {"message": f"Session {session_id} not found."}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)

            query = data_json.get("query", "")
            language = data_json.get("language", "en")

            result = await rag_system.get_response(
                query=query,
                session_id=session_id
            )

            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
