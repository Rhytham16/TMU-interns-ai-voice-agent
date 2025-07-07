# 🎙️ TMU Interns AI Voice Agent – Budger

Welcome to the official documentation for **Budger**, an AI-powered voice and chat assistant developed by interns at **Teerthanker Mahaveer University (TMU)** in collaboration with **Cogent Infotech Corporation**.

This intelligent system supports **natural voice and text-based conversations**, enabling efficient document search and real-time query resolution using cutting-edge AI tools.

---

## 🧠 Project Overview

This project combines **speech recognition**, **natural language understanding**, **vector-based document retrieval**, and **real-time AI streaming** to create a smart, scalable assistant.

The assistant uses:

- 🗣️ Voice + text input/output via a custom ChatGPT-style UI
- 🤖 FastAPI backend with WebSocket support
- 🧾 LangChain + ChromaDB for document embedding and semantic search
- ✨ Gemini 1.5 Flash (Google Generative AI) for high-speed, context-aware response generation
- 🧠 SQLite database for user authentication and chat history logging

---

## ✨ Key Features

✅ Voice + text interaction  
✅ Real-time streaming AI responses  
✅ Document ingestion with vector-based querying  
✅ Fast and scalable FastAPI backend  
✅ User login/signup system  
✅ Chat history stored in SQLite  
✅ Clean, responsive UI using custom `chat.html`

---

## 🧰 Tech Stack

| Component                     | Description |
|-------------------------------|-------------|
| **Python**                    | Core language |
| **FastAPI**                   | Web backend framework |
| **LangChain**                 | Document parsing & LLM pipelines |
| **ChromaDB**                  | Vector DB for semantic search |
| **Gemini 1.5 Flash**          | Google’s LLM for response generation |
| **SQLite**                    | User login and chat history |
| **SpeechRecognition / pyttsx3** | Optional voice input/output |
| **WebSocket**                 | Real-time chat streaming |
| **Custom Frontend (chat.html)** | Modern UI, ChatGPT-style UX |

---

## 🗂️ Documentation Sections

| Page                  | Description |
|-----------------------|-------------|
| [Overview](overview.md)         | High-level architecture and flow |
| [App](app.md)                   | FastAPI backend, endpoints, models |
| [Load Documents](load_documents.md) | Document ingestion and vector indexing |
| [Setup](setup.md)               | Environment setup and dependencies |
| [Structure](structure.md)       | File and folder layout of the project |

---

## 📍 Tip

Use this documentation to understand the **architecture**, **APIs**, and **deployment steps** for Budger. Whether you are maintaining the system or contributing new features — this is your guide.

---
