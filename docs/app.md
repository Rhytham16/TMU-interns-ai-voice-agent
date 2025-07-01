# 🚀 `app.py` – Main Application

The `app.py` file serves as the **entry point** for running the FastAPI server.  
It configures middleware, handles routing, and powers the AI Voice Assistant backend.

---

## 🔹 Key Responsibilities

- Initialize the FastAPI application
- Configure CORS for cross-origin API calls
- Start the Uvicorn server on `localhost:8000`
- Integrate routing for the assistant

---

## 🧩 1. Importing Dependencies

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

These libraries are essential:

FastAPI: For creating the backend APIs

CORSMiddleware: For allowing frontend (e.g., React, HTML, Postman) to make API calls

uvicorn: ASGI server that runs the FastAPI app

## ⚙️ 2. App Initialization

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Accepts requests from all domains (use caution in production)
    allow_credentials=True,
    allow_methods=["*"],          # Allows GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],          # Accepts all headers
)
🛡️ In production, replace "*" in allow_origins with your domain name for security.

## 📡 3. Running the Server

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

Explanation:
__name__ == "__main__" ensures the server starts only when this file is run directly.

host="0.0.0.0" lets the app accept external requests (on LAN, Docker, etc.)

port=8000 specifies the port

## 💻 Run the App Locally

python app.py
Then open your browser and go to:
http://127.0.0.1:8000