"""
Load documents into the Budger AI Assistant knowledge base.
Run this script to populate the RAG system with your Cogent Infotech knowledge base.
Updated to work with modular structure.
"""

import asyncio
import sys
import os
from pathlib import Path
from glob import glob

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import setup_logger, load_environment
    from rag_system import OptimizedRAGSystem
    
    # Initialize logging and environment
    setup_logger()
    load_environment()
    
    # Initialize RAG system
    rag_system = OptimizedRAGSystem()
    
    print("✅ Successfully imported modular RAG system")
except ImportError as e:
    print(f"❌ Failed to import RAG system: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

async def load_knowledge_base():
    """
    Loads all PDF files from the 'data/' folder into the vector store.
    
    Returns:
        bool: True if at least one file was loaded successfully, False if all failed or no files found.
    """
 
    # Check if data directory exists
    if not os.path.exists("data"):
        print("❌ 'data/' directory not found. Please create it and add your PDF files.")
        return False
    
    # Find all PDF files (case-insensitive)
    pdf_files = glob("data/*.pdf")
    
    if not pdf_files:
        print("❌ No PDF files found in the 'data/' directory.")
        print("💡 Please add your PDF files to the 'data/' folder and try again.")
        return False
    
    print(f"📄 Found {len(pdf_files)} PDF file(s) in 'data/' directory:")
    for pdf in pdf_files:
        print(f"   - {pdf}")
    
    success_count = 0
    failed_files = []
    
    print("\n🔄 Starting document loading process...")
    
    for pdf_path in pdf_files:
        try:
            file_name = os.path.basename(pdf_path)
            print(f"🔄 Loading {file_name}...")
            
            success = await rag_system.add_documents_async(pdf_path, "cogent_sales")
            
            if success:
                print(f"✅ Successfully loaded: {file_name}")
                success_count += 1
            else:
                print(f"❌ Failed to load: {file_name}")
                failed_files.append(pdf_path)
                
        except Exception as e:
            print(f"❌ Error loading {os.path.basename(pdf_path)}: {str(e)}")
            failed_files.append(pdf_path)
    
    # Summary
    print(f"\n📊 Loading Summary:")
    print(f"   ✅ Successfully loaded: {success_count}/{len(pdf_files)} files")
    
    if failed_files:
        print(f"   ❌ Failed files:")
        for failed_file in failed_files:
            print(f"      - {os.path.basename(failed_file)}")
    
    # Log vector store stats after loading
    if success_count > 0:
        print(f"\n📈 Vector store updated!")
        rag_system.log_vectorstore_stats()
        return True
    else:
        print(f"\n❌ No files were loaded successfully.")
        return False

def check_prerequisites():
    """
    Checks all required files and configurations before loading documents.

    Returns:
        bool: True if all prerequisites are met, False otherwise.
    """
    print("🔍 Checking prerequisites...")

    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found")
        print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
    else:
        print("✅ .env file found")

    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key in the .env file")
        return False
    else:
        print("✅ OpenAI API key configured")

    if not os.path.exists('enhanced_chroma_store'):
        print("📁 Creating vector store directory...")
        os.makedirs('enhanced_chroma_store', exist_ok=True)

    print("✅ Prerequisites check complete")
    return True

async def main():
    """
    Main function to run the knowledge base loader script.

    Handles prerequisite checking and invokes document loading.
    """
    print("🚀 Budger AI Assistant - Knowledge Base Loader (Modular Version)")
    print("=" * 60)

    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return

    print("\n🔄 Starting document loading process...")

    success = await load_knowledge_base()

    if success:
        print("\n✅ Knowledge base loaded successfully!")
        print("\n📋 Next steps:")
        print("1. Start your FastAPI server: python app.py")
        print("2. Open your browser to: http://localhost:8000")
        
    else:
        print("\n❌ Failed to load knowledge base. Please check the errors above.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()