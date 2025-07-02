"""
Load documents into the Budger AI Assistant knowledge base.
"""

import asyncio
import sys
import os
from rag_system import AdvancedRAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize RAG system
rag_system = AdvancedRAGSystem(openai_key=os.getenv("OPENAI_API_KEY"))

async def load_knowledge_base():
    pdf_files = [
        "Sales AI Agent Knowledgebase (1).pdf",
        "data/Sales AI Agent Knowledgebase (1).pdf",
        "documents/Sales AI Agent Knowledgebase (1).pdf"
    ]

    pdf_path = next((f for f in pdf_files if os.path.exists(f)), None)
    if not pdf_path:
        print("❌ PDF not found in known locations.")
        return False

    print(f"📄 Found: {pdf_path}")
    print("🔄 Loading into vector store...")

    try:
        success = rag_system.add_documents(pdf_path, "cogent_sales")
        if success:
            print("✅ Knowledge base loaded successfully!")
            return True
        else:
            print("❌ Failed to load documents.")
            return False
    except Exception as e:
        print("❌ Exception occurred while loading:", e)
        return False

def check_prerequisites():
    print("🔍 Checking environment...")
    if not os.path.exists('.env'):
        print("⚠️ .env file not found")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment.")
        return False

    if not os.path.exists("enhanced_chroma_store"):
        os.makedirs("enhanced_chroma_store", exist_ok=True)
        print("📁 Created vector store directory.")
    return True

async def main():
    print("Budger AI Assistant - Document Loader\n" + "="*40)
    if not check_prerequisites():
        return
    await load_knowledge_base()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("⏹️ Interrupted.")