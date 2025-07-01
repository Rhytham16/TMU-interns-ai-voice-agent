# app.py – Main Application

The `app.py` file initializes and runs the FastAPI server. It handles API routing and integrates with the voice assistant logic.

## Key Components

### 1. Importing Dependencies

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
 
