# 📄 `load_documents.py` – Document Loader & Vectorizer

The `load_documents.py` script powers the backend’s ability to **understand and retrieve information from your local documents**.

It loads files (PDFs, Word docs, and text files), breaks them into chunks, converts them into embeddings using **OpenAI or other LLM models**, and stores them in **ChromaDB** for lightning-fast semantic search.

---

## 🎯 Purpose

To preprocess and convert documents into a **vector store** that the voice agent can later query intelligently.

> This allows the assistant to answer user questions based on your custom documents using embeddings + similarity search.

---

## 🛠️ What the Script Does

| Step | Description |
|------|-------------|
| 📂 Load Files | Reads `.pdf`, `.txt`, `.docx` files from `data/` folder |
| ✂️ Chunking | Splits large documents into manageable text chunks |
| 🧠 Embedding | Converts each chunk into a vector using OpenAI or HuggingFace |
| 🗃️ Store | Saves the vectors in **ChromaDB** (local vector database) |

---

## 🧾 Sample Workflow

```bash
python load_documents.py

## 🔍 Behind the Scenes

### 📥 1. Load Documents

```python
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader("data/")
documents = loader.load()

###✂️ 2. Split into Chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

###🧠 3. Create Embeddings

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

You replace OpenAIEmbeddings() with HuggingFaceEmbeddings, CohereEmbeddings, or others.

###🗃️ 4. Store in ChromaDB

from langchain.vectorstores import Chroma

db = Chroma.from_documents(chunks, embeddings, persist_directory="db/")
db.persist()

✅ Done!
After running this script, your documents are transformed into a searchable vector format — enabling the AI assistant to answer questions from your custom files.

You’re now ready to ask your assistant anything based on your data!



