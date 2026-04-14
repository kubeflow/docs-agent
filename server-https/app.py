import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, List, Optional, AsyncGenerator

# Import the compiled graph from core_agent.graph
from core_agent.graph import agent_graph

# Config
PORT = int(os.getenv("PORT", "8000"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

app = FastAPI(title="Kubeflow Docs API Service (LangGraph)", version="1.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = False # LangGraph streaming is different, defaulting to False for simplicity here

@app.get("/")
async def hello():
    return {"message": "Hello from Kubeflow Docs API with LangGraph!", "service": "https-api"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "https-api"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Invokes the LangGraph state machine.
    """
    try:
        logger.info(f"Processing message: {request.message[:100]}")
        
        # Initial state for the graph
        initial_state = {
            "user_query": request.message,
            "intent": "",
            "retrieved_context": [],
            "citations": [],
            "generation": "",
            "error_count": 0
        }
        
        # Invoke the graph
        # For simplicity in this boilerplate, we're using .ainvoke()
        final_state = await agent_graph.ainvoke(initial_state)
        
        return {
            "response": final_state.get("generation"),
            "citations": final_state.get("citations"),
            "metadata": {
                "intent": final_state.get("intent"),
                "error_count": final_state.get("error_count")
            }
        }
        
    except Exception as e:
        logger.error(f"Chat handling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")

if __name__ == "__main__":
    logger.info(f"🚀 Starting Kubeflow Docs HTTP API Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
