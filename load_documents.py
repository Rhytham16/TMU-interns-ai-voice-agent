
"""
Load documents into the Budger AI Assistant knowledge base
Run this script to populate the RAG system with your Cogent Infotech knowledge base
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path to import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import rag_system
    print("✅ Successfully imported RAG system")
except ImportError as e:
    print(f"❌ Failed to import RAG system: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

async def load_knowledge_base():
    """Load the Cogent Infotech knowledge base into the vector store"""
    
    # Look for the PDF file in current directory
    pdf_files = [
        "Sales AI Agent Knowledgebase (1).pdf",
        "data/Sales AI Agent Knowledgebase (1).pdf",
        "documents/Sales AI Agent Knowledgebase (1).pdf"
    ]
    
    pdf_path = None
    for path in pdf_files:
        if os.path.exists(path):
            pdf_path = path
            break
    
    if not pdf_path:
        print("❌ Could not find the PDF file. Please ensure one of these files exists:")
        for path in pdf_files:
            print(f"   - {path}")
        return False
    
    print(f"📄 Found PDF file: {pdf_path}")
    print("🔄 Loading documents into vector store...")
    
    try:
        # Load the documents
        success = rag_system.add_documents(pdf_path, "cogent_sales")
        
        if success:
            print("✅ Successfully loaded Cogent Infotech knowledge base!")
            print("📊 Your AI assistant is now ready with:")
            print("   - Company information")
            print("   - Service offerings")
            print("   - Client success stories")
            print("   - Sales talking points")
            return True
        else:
            print("❌ Failed to load documents")
            return False
            
    except Exception as e:
        print(f"❌ Error loading documents: {str(e)}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found")
        print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
    else:
        print("✅ .env file found")
    
    # Check if OpenAI API key is set
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key in the .env file")
        return False
    else:
        print("✅ OpenAI API key configured")
    
    # Check if vector store directory exists
    if not os.path.exists('enhanced_chroma_store'):
        print("📁 Creating vector store directory...")
        os.makedirs('enhanced_chroma_store', exist_ok=True)
    
    print("✅ Prerequisites check complete")
    return True

async def main():
    """Main function to run the document loading process"""
    print(" Budger AI Assistant - Knowledge Base Loader")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n Prerequisites not met. Please fix the issues above and try again.")
        return
    
    print("\n🔄 Starting document loading process...")
    
    # Load the knowledge base
    success = await load_knowledge_base()
    
    if success:
        print("\n Knowledge base loaded successfully!")
        print("\n Next steps:")
        print("1. Start your FastAPI server: python app.py")
        print("2. Open your browser to: http://localhost:8000")
        print("3. Use password: komal123")
        print("4. Start chatting with your AI assistant!")
    else:
        print("\n Failed to load knowledge base. Please check the errors above.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()