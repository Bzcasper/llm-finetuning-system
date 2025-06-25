from fastapi import APIRouter, HTTPException
import os
import time
import json
from typing import Dict, Any

router = APIRouter(prefix="/modal", tags=["modal"])


@router.get("/status")
async def get_modal_status() -> Dict[str, Any]:
    """
    Check Modal.com connection status and app information
    """
    try:
        # Check Modal connection
        modal_token_id = os.environ.get("MODAL_TOKEN_ID")
        modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        modal_profile = os.environ.get("MODAL_PROFILE", "ai-tool-pool")

        connected = False
        error_message = None
        app_status = "unknown"

        if modal_token_id and modal_token_secret:
            try:
                import modal

                # Set environment variables for Modal
                os.environ["MODAL_TOKEN_ID"] = modal_token_id
                os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret

                # Try to lookup the app
                app = modal.App.lookup(
                    "llm-finetuner", create_if_missing=False)
                connected = True
                app_status = "deployed"

            except Exception as e:
                error_message = str(e)
                if "No such environment" in str(e):
                    app_status = "environment_not_found"
                elif "not found" in str(e):
                    app_status = "app_not_deployed"
                else:
                    app_status = "connection_error"
        else:
            error_message = "Modal credentials not configured"
            app_status = "credentials_missing"

        response = {
            "connected": connected,
            "environment": modal_profile,
            "app_name": "llm-finetuner",
            "app_status": app_status,
            "error": error_message,
            "timestamp": time.time(),
            "credentials_configured": bool(modal_token_id and modal_token_secret)
        }

        return response

    except ImportError:
        return {
            "connected": False,
            "environment": "unknown",
            "app_name": "llm-finetuner",
            "app_status": "modal_not_installed",
            "error": "Modal library not available in serverless environment",
            "timestamp": time.time(),
            "credentials_configured": False
        }
    except Exception as e:
        return {
            "connected": False,
            "environment": "unknown",
            "app_name": "llm-finetuner",
            "app_status": "error",
            "error": str(e),
            "timestamp": time.time(),
            "credentials_configured": False
        }


@router.get("/resources")
async def get_modal_resources() -> Dict[str, Any]:
    """
    Get Modal resource information (GPU, storage, etc.)
    """
    try:
        # Mock response for now since we'd need actual Modal connection
        return {
            "gpu_info": {
                "available_types": ["T4", "A100"],
                "current_usage": 0,
                "max_concurrent": 4
            },
            "storage_info": {
                "total_gb": 100,
                "used_gb": 25,
                "available_gb": 75
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
