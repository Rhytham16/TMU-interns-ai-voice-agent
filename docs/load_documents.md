#  `load_documents.py` – Knowledge Base Loader for Budger AI

The `load_documents.py` script is responsible for **loading a knowledge base PDF into the vector store**, allowing Budger to answer user queries based on relevant company documents.

It uses:
-  `PyPDFLoader` from `langchain_community` to read PDFs
-  `RecursiveCharacterTextSplitter` to break text into chunks
-  `GoogleGenerativeAIEmbeddings` for semantic vectorization
-  `Chroma` to store and persist embeddings for fast search

---

##  What It Does

| Step | Description |
|------|-------------|
|  Detects PDF | Searches for `Sales AI Agent Knowledgebase (1).pdf` in common folders |
|  Checks Prereqs | Confirms `.env` and `GEMINI_API_KEY` are set |
|  Splits Text | Breaks PDF content into smaller chunks for embedding |
|  Embeds | Converts chunks into vectors using Gemini embeddings |
|  Saves | Stores the vectors in `enhanced_chroma_store` via ChromaDB |

---

## ⚙️ Usage

### Run the loader script

```bash
python load_documents.py

---

## ✅ What You’ll See If It Works

If the PDF is successfully loaded and embedded into the vector store, you’ll see:

```text
✅ Successfully loaded Cogent Infotech knowledge base!

---

??? info " Where to Place Your PDF"
    The script automatically checks the following paths in order:

    ```text
    ./Sales AI Agent Knowledgebase (1).pdf
    ./data/Sales AI Agent Knowledgebase (1).pdf
    ./documents/Sales AI Agent Knowledgebase (1).pdf
    ```

    ✅ Make sure your PDF file is saved in one of these locations before running the script.


##  Key Code Breakdown

###  1. Load PDF

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(pdf_path)
documents = loader.load()

### 2. Split into Chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)
splits = splitter.split_documents(documents)

### 3. Add Metadata

from datetime import datetime

for split in splits:
    split.metadata["collection"] = "cogent_sales"
    split.metadata["timestamp"] = datetime.now().isoformat()


### 4. Embed & Store Vectors

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Chroma(
    persist_directory="enhanced_chroma_store",
    embedding_function=embeddings
)

vectorstore.add_documents(splits)
vectorstore.persist()


✅ Done!




