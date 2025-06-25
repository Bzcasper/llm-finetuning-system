import modal
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import uvicorn
from contextlib import asynccontextmanager

# Import modular API routers
from api.health import router as health_router
from api.training import router as training_router
from api.datasets import router as datasets_router
from api.modal_status import router as modal_status_router
from api.models import TrainingConfig, TrainingStatus, TrainingJob, ModalStatus
from api.services import ModalService, TrainingService, storage_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan management for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM Fine-Tuning API Server")
    await ModalService.initialize()
    yield
    # Shutdown
    logger.info("Shutting down LLM Fine-Tuning API Server")

# Define the FastAPI app
api_app = FastAPI(
    title="LLM Fine-Tuning Studio API",
    description="Professional-grade API for fine-tuning Large Language Models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:5173",  # Vite dev server
        "https://*.vercel.app",   # Vercel deployments
        "*" if os.getenv("ENVIRONMENT") == "development" else "https://your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include modular routers
api_app.include_router(health_router, prefix="/api", tags=["health"])
api_app.include_router(training_router, prefix="/api", tags=["training"])
api_app.include_router(datasets_router, prefix="/api", tags=["datasets"])
api_app.include_router(modal_status_router, prefix="/api", tags=["modal"])

# Root endpoint
@api_app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Fine-Tuning Studio API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "modal_status": "/api/modal/status",
        "endpoints": {
            "training": "/api/training",
            "datasets": "/api/datasets",
            "models": "/api/models"
        }
    }

# Legacy endpoints for backward compatibility
@api_app.get("/api/models", tags=["models"])
async def list_models():
    """List available models"""
    try:
        modal_service = ModalService()
        if modal_service.is_connected():
            result = await modal_service.list_models_and_datasets()
            return {"models": result.get("models", [])}
        else:
            # Return mock data for development
            return {
                "models": [
                    {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "type": "huggingface"},
                    {"name": "microsoft/DialoGPT-medium", "type": "huggingface"},
                    {"name": "meta-llama/Llama-2-7b-chat-hf", "type": "huggingface"},
                    {"name": "finetuned_model_v1", "type": "volume", "path": "/vol/finetuned_model_v1"}
                ]
            }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

# Error handlers
@api_app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

@api_app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

