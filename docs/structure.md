# 🗂️ Project Folder Structure

Here is an overview of the directory and file organization of the **TMU AI Voice Agent** project.

Use this structure to navigate the project during development, debugging, or contribution.

---

## 📁 Root Directory

```plaintext
TMU-INTERNS-AI-VOICE-AGENT/
│
├── app.py                   # 🚀 Main FastAPI application entry point
├── load_documents.py        # 📄 Script to load & index documents into ChromaDB
├── requirements.txt         # 📦 List of required Python dependencies
├── .env                     # 🔐 Environment variables (e.g., API keys)
│
├── chroma_db/               # 🧠 Persistent vector store (ChromaDB)
├── enhanced_chroma_store/   # 🔧 Custom vector DB storage (optional/extended)
├── data/                    # 📂 Folder for user-uploaded documents (.pdf, .txt)
├── static/                  # 🎧 Static files like audio or frontend assets
│
├── my-docs/                 # 📘 MkDocs documentation folder
│   ├── mkdocs.yml           # ⚙️ MkDocs configuration file
│   └── docs/                # 📄 All markdown-based documentation files
