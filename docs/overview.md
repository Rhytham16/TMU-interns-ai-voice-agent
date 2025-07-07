# 🧠 Project Overview: TMU Interns AI Voice Agent (Budger)

The **TMU Interns AI Voice Agent** — code-named **Budger** — is a smart, real-time conversational assistant developed by students at **Teerthanker Mahaveer University (TMU)** in collaboration with **Cogent Infotech**.

It listens to user queries, understands them using advanced language models, and responds with relevant information — even from uploaded documents and company PDFs.

---

## 🎙️ What Does It Do?

This AI Voice Agent enables users to:

- 🗣️ Speak or type their questions
- 📄 Upload and embed documents for semantic search
- 💬 Get intelligent responses using Google’s Gemini Flash model
- ⚡ Experience real-time streaming chat, like talking to a person
- 🔐 Log in securely with session-based memory and chat history

---

## ✨ Features at a Glance

| 🔧 Feature                         | ✅ Description |
|----------------------------------|----------------|
| 🎤 **Voice & Text Input**         | Optional speech input or keyboard typing |
| ⚙️ **FastAPI + WebSocket Backend**| Real-time streaming via `/chat/stream` and `/ws` |
| 📄 **PDF Ingestion & Indexing**   | Documents processed into semantic vectors |
| 🧠 **Gemini-powered AI**          | Uses **Gemini 1.5 Flash** from Google Generative AI |
| 🔎 **ChromaDB + LangChain**       | Vector search engine to retrieve relevant content |
| 🗂️ **User Authentication**        | Login/signup using SQLite |
| 🧑‍💻 **Custom Frontend UI**        | Modern, responsive interface (`chat.html`) |

---

## 🔬 How It Works

1. **User asks a question** 🗣️  
   → Input is received via text or optional voice

2. **Query is analyzed** 🤖  
   → Gemini interprets the request using prior chat context and vector search

3. **Knowledge is retrieved** 📄  
   → ChromaDB finds relevant document chunks via semantic similarity

4. **AI responds intelligently** 💡  
   → Gemini replies, streamed back word-by-word in real-time

---

## 🧰 Tech Stack

| Technology | Role |
|------------|------|
| **Python** | Core programming language |
| **FastAPI** | Backend APIs + WebSocket chat |
| **LangChain** | Document processing pipeline |
| **ChromaDB** | Vector storage and similarity search |
| **Gemini (Google GenAI)** | Main language model for generating responses |
| **SQLite** | Stores user accounts and chat history |
| **SpeechRecognition / pyttsx3** *(optional)* | Voice interface |
| **chat.html** | Custom-built ChatGPT-style frontend |

---

## 🏁 Who Is It For?

- 🧑‍🎓 AI/ML students & researchers at TMU  
- 📚 Faculty needing fast document search  
- 🛠️ Developers building intelligent assistants  
- 🧠 Anyone exploring AI chat + document Q&A systems

---

*Built with ❤️ by TMU Interns using modern AI technology.*

Explore the rest of the documentation to get started!
