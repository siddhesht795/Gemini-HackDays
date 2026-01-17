import sys
import os

# Add the parent directory to sys.path to allow imports from ai_backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
from stt import router as stt_router
from tts import router as tts_router
from ai_backend.speech_chat import router as speech_chat_router
from ai_backend.rag import router as rag_router

app = FastAPI(title="MIRAGE Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for hackathon; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(speech_chat_router)
app.include_router(rag_router)
  
@app.get("/")
def root():
    return {"status": "ok"}