"""
Pydantic models for Budger AI Voice Assistant
Defines request/response models for API endpoints
"""

from typing import List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    language: str = "en"
    stream: bool = True

class QueryResponse(BaseModel):
    response: str
    sources: List[str] = []
    session_id: str
    timestamp: str
    response_time: float = 0.0

class DocumentUploadRequest(BaseModel):
    file_path: str
    collection_name: str = "default"

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    login_id: str
    password: str