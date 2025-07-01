# 🛠️ Setup & Installation Guide

Welcome to the setup guide for the **TMU AI Voice Agent Project**.  
This guide will walk you through installing dependencies, setting up your environment, and running the application locally.

---

## 🧩 Prerequisites

Make sure you have the following tools installed on your system:

- ✅ **Python 3.10+**
- ✅ **`uv`** or built-in `venv` for virtual environments
- ✅ **Git** (optional but recommended)

> 🔍 You can check your Python version with:  
> `python --version`

---

## 🖼️ Architecture Diagram

![Architecture Overview](images/example.png)

> 📌 Replace `example.png` with your actual diagram in `docs/images/`.

---

## 🔧 Installation Steps

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/your/repo.git
cd TMU-INTERNS-AI-VOICE-AGENT

### 🔹 Step2: Create and Activate a Virtual Environment

uv venv
.\.venv\Scripts\activate

🪟 For Windows, use .\.venv\Scripts\activate
🐧 For Linux/macOS, use source .venv/bin/activate

### 🔹 Step 3: Install Dependencies

Using uv:
uv pip install -r requirements.txt

Or with standard pip:
pip install -r requirements.txt

📦 This will install FastAPI, LangChain, ChromaDB, and other required libraries.

### 🔹 Step 4: Run the App

python app.py

Once the server is running, open your browser and visit:
http://localhost:8000


🚀 You're Ready to Go!
Your app is now running locally with all dependencies installed.
You can now begin testing endpoints, loading documents, or building a frontend for the voice agent.


🧪 Troubleshooting Tips
Problem	Solution
🔺 activate.ps1 cannot be loaded	Run Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
🔺 Module not found	Recheck if the virtual environment is activated
🔺 localhost not loading	Ensure nothing else is using port 8000 or change the port in app.py