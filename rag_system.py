from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
import os

class AdvancedRAGSystem:
    def __init__(self, persist_dir: str = "enhanced_chroma_store", openai_key: str = None):
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        self.embedding = OpenAIEmbeddings(openai_api_key=openai_key)
        self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embedding)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_key, streaming=True)
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"
        )

    def get_response(self, query: str):
        try:
            result = self.qa_chain.invoke({"question": query})
            answer = result.get("answer", "I couldn't find a relevant answer.")
            sources = [
                f"Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('source', 'Unknown')}"
                for doc in result.get("source_documents", [])[:3]
            ]
            return answer, sources
        except Exception as e:
            print("❌ Error in get_response():", e)
            return "⚠️ AI system is not available. Please check the server logs.", []

    def add_documents(self, file_path: str, collection_name: str = "default") -> bool:
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = splitter.split_documents(documents)

            for doc in splits:
                doc.metadata["collection"] = collection_name
                doc.metadata["timestamp"] = datetime.now().isoformat()

            self.vectorstore.add_documents(splits)
            self.vectorstore.persist()

            print(f"✅ Added {len(splits)} document chunks from: {file_path}")
            return True
        except Exception as e:
            print("❌ Error adding documents:", e)
            return False
