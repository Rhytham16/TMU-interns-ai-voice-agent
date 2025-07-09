# 🛠️ Setup & Installation Guide

Welcome to the setup guide for the **TMU AI Voice Agent Project (Budger)**.  
This assistant uses OpenAI GPT-3.5 Turbo, LangChain, ChromaDB, and FastAPI to let users search documents and chat with AI — in real time.

---

## 🧩 Prerequisites

Make sure the following are installed:

- ✅ Python 3.10+
- ✅ `uv` (or Python `venv`)
- ✅ Git (optional but useful)

Check your versions:

```bash
python --version
```

---

## 🖼️ Architecture Diagram

![Architecture Overview](images/rag_diagram.png)


### 🔧 Installation Steps

### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/Rhytham16/TMU-interns-ai-voice-agent.git
cd TMU-interns-ai-voice-agent
```
### 🔹 Step 2: Create and Activate a Virtual Environment

```bash
uv venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

> 🛠️ Make sure your virtual environment is activated in every new terminal session before continuing.

---

### 🔹 Step 3: Install All Required Dependencies

This project requires several Python packages for FastAPI, LangChain, ChromaDB, OpenAI, and more.

Run the following:

uv sync 

Or manually, you can install core dependencies:

```bash
pip install fastapi uvicorn langchain openai chromadb \
python-dotenv pydantic aiofiles requests PyPDF2 \
langchain-openai langchain-core langchain-community \
langchain-text-splitters websockets SQLAlchemy \
coloredlogs humanfriendly langchain_chroma pypdf
```

---

### 🔹 Step 4: Configure Environment Variables

Create a .env file at the project root and add your GPT API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 🔹 Step 5: Set Up the SQLite Database

```bash
python setup_db.py
```
### 🔹 Step 6: Load PDFs (Document Embedding)
Make sure your PDF is placed in one of the following locations:

```bash
./
./data/
./documents/
```
Then run:

```
python load_documents.py
```

### 🔹 Step 7: Run the FastAPI App

Start the server:

```bash
python app.py or uvicorn app:app --reload
```

Visit the app in your browser:

```
http://localhost:8000
```

You can now:
- Sign up and log in
- Ask questions from your uploaded documents
- Chat in real time via WebSocket (/ws/{session_id})

---


✅ You're All Set!
Your voice-enabled AI assistant is now up and running!
You can explore:

🔎 Document-based Q&A via OpenAI

⚡ Real-time streaming chat using WebSockets

🧠 Smart document memory powered by LangChain & ChromaDB

Happy building! 🎉