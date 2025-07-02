from rag_system import AdvancedRAGSystem
import os

if __name__ == "__main__":
    openai_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY:", openai_key)
    rag = AdvancedRAGSystem(openai_key=openai_key)
    print("Testing RAG system with a simple query...")
    response, sources = rag.get_response("What is your name?")
    print("ANSWER:", response)
    print("SOURCES:", sources)
