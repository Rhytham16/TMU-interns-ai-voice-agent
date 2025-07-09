"""
RAG (Retrieval-Augmented Generation) system module for Budger AI Voice Assistant
Handles document processing, vector storage, and AI response generation
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, AsyncGenerator

from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

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

    def clear_session_history(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversation_histories:
            del self.conversation_histories[session_id]

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