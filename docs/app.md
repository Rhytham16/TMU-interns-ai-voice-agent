# 🚀 `app.py` – Main Application (Budger AI Assistant)

The `app.py` file is the **main entry point** for Budger — an AI-powered assistant built using FastAPI, GPT-3.5 Turbo(OpenAI), LangChain, ChromaDB, and SQLite. It handles all backend functionality, including chat routing, user authentication, PDF ingestion, and real-time streaming responses.

---

## 📌 Key Responsibilities

- 🔧 Initialize FastAPI with middleware and CORS
- 🔐 Load API key from `.env`
- 🧠 Run GPT-3.5 Turbo-powered OptimizedRAGSystem
- 📄 Handle PDF uploads and vector embeddings
- 💬 Enable real-time WebSocket chat
- 🗂️ Manage users and chat logs via SQLite
- 🔁 Serve a dynamic chat UI via static files

---

## 🧩 Core Technologies

| Component | Purpose |
|----------|---------|
| **FastAPI** | Web API and routing |
| **GPT-3.5 Turbo (OpenAI)** | Language model for response generation |
| **LangChain** | Document loading and chunking |
| **ChromaDB** | Vector database for semantic search |
| **SQLite** | Stores users and chat history |
| **WebSocket** | Real-time streaming of AI responses |
| **chat.html** | Custom ChatGPT-style frontend UI |

---

## 🔐 Environment Configuration

The OpenAI API key is loaded from the `.env` file and configured using the OpenAI SDK:
```python
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure OpenAI client
client = OpenAI(api_key=openai_api_key)
```

## 🧠 OptimizedRAGSystem

`OptimizedRAGSystem` is a custom class that powers Budger's **Retrieval-Augmented Generation (RAG)** pipeline. It integrates document search with OpenAIs’s LLM to generate smart, context-aware responses.

This class lives inside `app.py` and handles the core AI logic for both chat and streaming endpoints.

---

### 🔍 Key Responsibilities

| Component | Description |
|----------|-------------|
| 📄 `load_documents()` | Loads and parses documents from PDF |
| ✂️ `split_documents()` | Splits content into smaller, overlapping text chunks |
| 🧠 `embed_documents()` | Converts chunks into vectors using OpenAI embeddings |
| 🔍 `query_documents()` | Performs similarity search in ChromaDB |
| 💬 `generate_response()` | Sends prompt + context to OpenAI |
| 🔁 `stream_response()` | Streams response back word-by-word |
| 🧠 `memory` | Maintains session-level history (chat memory) |

---

### 🛠️ How It Works

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

self.llm = ChatOpenAI(
    temperature=0.3,
    streaming=True,
    model="gpt-3.5-turbo",
    api_key=openai_api_key
)
self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
self.vectorstore = Chroma(
    persist_directory="enhanced_chroma_store",
    embedding_function=self.embeddings
)
```


## 🌐 API Endpoints

The following API routes are handled in `app.py`:

| Method     | Path                    | Description                               |
|------------|-------------------------|-------------------------------------------|
| `GET`      | `/`                     | Loads login page (`login.html`)           |
| `GET`      | `/home`                 | Loads chat UI (`chat.html`)               |
| `GET`      | `/health`               | Returns model and server status    |
| `POST`     | `/chat`                 | Returns full GPT-generated response    |
| `POST`     | `/chat/stream`          | Streams GPT response word-by-word       |
| `POST`     | `/upload`               | Uploads PDF and stores in ChromaDB        |
| `POST`     | `/signup`               | Creates a new user in SQLite              |
| `POST`     | `/login`                | Validates credentials                     |
| `DELETE`   | `/sessions/{session_id}`| Clears memory for a session               |
| `WebSocket`| `/ws/{session_id}`      | Enables real-time two-way chat            |

---

✅ These routes power the complete functionality of Budger's backend — including login, document ingestion, AI chat, and live streaming.

---

## 🔁 WebSocket Streaming

Budger uses a WebSocket endpoint to support **real-time AI conversation**, streaming GPT-3.5 Turbo responses word-by-word like ChatGPT.

```python
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # Connect and receive user message
    # Process with OptimizedRAGSystem
    # Stream back OpenAI's reply word-by-word
```


## 🧾 SQLite Database

The backend uses a local SQLite database (`budger_users.db`) to manage user credentials and chat history.

---

### 🗂️ `users` Table

| Field      | Description                      |
|------------|----------------------------------|
| `id`       | Auto-incremented user ID         |
| `username` | Unique username for login        |
| `email`    | User's email address             |
| `password` | Raw password (⚠️ should be hashed in production) |

---

### 🗃️ `chat_history` Table

| Field        | Description                                 |
|--------------|---------------------------------------------|
| `user_id`    | Foreign key linked to `users.id`            |
| `session_id` | Unique session ID for each conversation     |
| `query`      | The user's message/question                 |
| `response`   | The AI assistant’s answer (from GPT)     |
| `timestamp`  | Time of the interaction (auto-filled)       |

---

ℹ️ The table schemas are created by `setup_db.py` and used by `DatabaseManager` in `app.py` for inserting and retrieving data.

---

## 📁 Static Files and Frontend

Budger serves a simple web-based chat UI using static files located in the `/static` directory. These are rendered directly via FastAPI routes.

---

### 📂 Files and Their Purpose

| File         | Description                              |
|--------------|------------------------------------------|
| `login.html` | Login screen shown at `/`                |
| `chat.html`  | Main chat interface shown at `/home`     |
| `script.js`  | Handles WebSocket connection and messages|
| `style.css`  | Styles the chat UI                       |

---

### ⚙️ Static Mounting in FastAPI

```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
```
---

## 💻 How to Run the App

Follow these steps to launch the Budger assistant:

1. Make sure your `.env` file exists and contains your Gemini API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

2. Run the FastAPI app:
```
python app.py
```

3.Open your browser and go to:

http://localhost:8000

4. Sign up or log in with your credentials to start chatting with the AI.
