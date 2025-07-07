# 🗂️ Project Folder Structure

Here is a complete overview of the folder and file organization for the **TMU AI Voice Agent** (Budger) project.

This structure helps you navigate the codebase efficiently during development, debugging, or contribution.

---

## 📁 Root Directory

```plaintext
TMU-INTERNS-AI-VOICE-AGENT/
│
├── app.py                    # 🚀 Main FastAPI application with Gemini + WebSocket
├── load_documents.py         # 📄 Loads and embeds PDF into ChromaDB
├── setup_db.py               # 🗃️ Creates SQLite user + chat tables
├── requirements.txt          # 📦 Python dependencies list
├── pyproject.toml            # ⚙️ Project metadata (optional for uv/dev install)
├── .env                      # 🔐 Environment variables (e.g., GEMINI_API_KEY)
├── README.md                 # 📝 Optional: Project intro on GitHub
│
├── chroma_db/                # 🧠 ChromaDB default storage (vector data)
├── enhanced_chroma_store/    # 🧠 Persistent store used by LangChain + Chroma
├── budger_users.db           # 🗄️ SQLite DB storing users + chat history
├── logs/
│   └── app.log               # 📋 Runtime logs (auto-rotated)
│
├── data/                     # 📂 Input folder for your PDFs/docs
├── documents/                # 📂 (Optional) Additional document folder
│
├── static/                   # 🎧 Web frontend & static assets
│   ├── login.html            # 🔐 Login page
│   ├── chat.html             # 💬 Chat UI (ChatGPT-style)
│   ├── script.js             # ⚡ WebSocket logic for real-time streaming
│   └── style.css             # 🎨 Optional styling for frontend
│
├── my-docs/                  # 📘 MkDocs documentation root
│   ├── mkdocs.yml            # ⚙️ MkDocs configuration (site navigation)
│   └── docs/
│       ├── index.md          # 🏁 Project introduction
│       ├── app.md            # 🚀 FastAPI & backend details
│       ├── setup.md          # 🔧 Installation & usage
│       ├── load_documents.md # 📄 Document embedding pipeline
│       ├── structure.md      # 🗂️ Project folder layout (this file)
│       └── overview.md       # 🧠 Architecture + features
