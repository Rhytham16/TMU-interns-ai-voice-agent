from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler  

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

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

setup_logger()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI key loaded:", openai_api_key[:10], "****")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

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

class QueryRequest(BaseModel):
    """Model for user query requests."""
    query: str
    session_id: str = "default"
    language: str = "en"

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
        """Sets up the vector store either by loading existing or creating a new one."""
        if os.path.exists(self.persist_dir):
            logger.info("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            logger.info("Creating new vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

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
        system_template = """You are Budger, an advanced AI customer service agent for Cogent Infotech Corporation..."""

        human_template = """User Query: {question}\n\nPlease provide a helpful response..."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

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
