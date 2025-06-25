"""
Dataset and Model Management API endpoints for LLM Fine-tuning System
Provides comprehensive data management, file uploads, and model listing
"""

import os
import json
import time
import uuid
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import modal
from enum import Enum
import aiofiles
import pandas as pd
from pathlib import Path as PathLib

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Pydantic Models
class DatasetType(str, Enum):
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    TXT = "txt"
    PARQUET = "parquet"

class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    MODAL_VOLUME = "modal_volume"
    LOCAL = "local"
    S3 = "s3"
    MINIO = "minio"

class DatasetInfo(BaseModel):
    """Dataset information"""
    name: str
    path: str
    size: str
    type: DatasetType
    created: datetime
    modified: datetime
    rows: Optional[int] = None
    columns: Optional[List[str]] = None
    description: Optional[str] = None
    tags: List[str] = []
    source: str = "upload"
    is_valid: bool = True
    validation_errors: List[str] = []

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    source: ModelSource
    path: Optional[str] = None
    description: Optional[str] = None
    architecture: Optional[str] = None
    parameters: Optional[str] = None
    size: Optional[str] = None
    created: Optional[datetime] = None
    downloads: Optional[int] = None
    likes: Optional[int] = None
    tags: List[str] = []
    is_finetuned: bool = False
    base_model: Optional[str] = None

class DatasetUploadResponse(BaseModel):
    """Dataset upload response"""
    success: bool
    dataset_id: str
    name: str
    path: str
    size: str
    type: DatasetType
    rows: Optional[int] = None
    message: str
    validation_errors: List[str] = []

class DatasetValidationResult(BaseModel):
    """Dataset validation result"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    statistics: Dict[str, Any] = {}

class HuggingFaceDataset(BaseModel):
    """HuggingFace dataset information"""
    id: str
    name: str
    description: Optional[str] = None
    downloads: Optional[int] = None
    likes: Optional[int] = None
    tags: List[str] = []
    size: Optional[str] = None
    formats: List[str] = []

# Storage for dataset metadata (replace with database in production)
datasets_metadata: Dict[str, DatasetInfo] = {}
models_cache: Dict[str, List[ModelInfo]] = {}

# Modal app connection
try:
    deployed_app = modal.App.lookup("llm-finetuner", environment_name="ai-tool-pool")
    list_function = deployed_app.list_models_and_datasets if hasattr(deployed_app, 'list_models_and_datasets') else None
    print("✅ Connected to Modal app for dataset management")
except Exception as e:
    print(f"⚠️ Warning: Could not connect to Modal app: {e}")
    deployed_app = None
    list_function = None

# Utility functions
def validate_dataset_file(file_path: str, file_type: DatasetType) -> DatasetValidationResult:
    """Validate dataset file format and content"""
    errors = []
    warnings = []
    statistics = {}
    
    try:
        if file_type == DatasetType.JSONL:
            # Validate JSONL format
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                valid_lines = 0
                for i, line in enumerate(lines):
                    try:
                        data = json.loads(line.strip())
                        if not isinstance(data, dict):
                            errors.append(f"Line {i+1}: Expected JSON object, got {type(data).__name__}")
                        else:
                            valid_lines += 1
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i+1}: Invalid JSON - {str(e)}")
                
                statistics = {
                    "total_lines": len(lines),
                    "valid_lines": valid_lines,
                    "empty_lines": len([l for l in lines if not l.strip()])
                }
                
        elif file_type == DatasetType.JSON:
            # Validate JSON format
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    statistics["total_records"] = len(data)
                    if data and isinstance(data[0], dict):
                        statistics["columns"] = list(data[0].keys())
                elif isinstance(data, dict):
                    statistics["keys"] = list(data.keys())
                    
        elif file_type in [DatasetType.CSV, DatasetType.TSV]:
            # Validate CSV/TSV format
            separator = ',' if file_type == DatasetType.CSV else '\t'
            df = pd.read_csv(file_path, sep=separator, nrows=1000)  # Sample first 1000 rows
            
            statistics = {
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "sample_rows": min(1000, len(df)),
                "null_values": df.isnull().sum().to_dict()
            }
            
        elif file_type == DatasetType.TXT:
            # Basic text file validation
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            statistics = {
                "total_lines": len(lines),
                "total_characters": len(content),
                "empty_lines": len([l for l in lines if not l.strip()])
            }
            
    except Exception as e:
        errors.append(f"File validation error: {str(e)}")
    
    return DatasetValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        statistics=statistics
    )

def get_file_size_str(file_path: str) -> str:
    """Get human-readable file size"""
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def infer_dataset_type(filename: str) -> DatasetType:
    """Infer dataset type from filename"""
    ext = filename.lower().split('.')[-1]
    if ext == 'jsonl':
        return DatasetType.JSONL
    elif ext == 'json':
        return DatasetType.JSON
    elif ext == 'csv':
        return DatasetType.CSV
    elif ext == 'tsv':
        return DatasetType.TSV
    elif ext == 'txt':
        return DatasetType.TXT
    elif ext == 'parquet':
        return DatasetType.PARQUET
    else:
        return DatasetType.TXT  # Default fallback

# API Endpoints

@router.get("/", response_model=Dict[str, Any])
async def list_datasets(
    source: Optional[str] = Query(None, description="Filter by source (upload, huggingface, modal)"),
    type: Optional[DatasetType] = Query(None, description="Filter by dataset type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of datasets to return"),
    offset: int = Query(0, ge=0, description="Number of datasets to skip")
):
    """
    List all available datasets with filtering and pagination
    
    Returns datasets from various sources including uploaded files,
    Modal volumes, and HuggingFace datasets.
    """
    all_datasets = []
    
    # Get uploaded datasets
    uploaded_datasets = list(datasets_metadata.values())
    if source is None or source == "upload":
        all_datasets.extend(uploaded_datasets)
    
    # Get Modal volume datasets
    modal_datasets = []
    if (source is None or source == "modal") and list_function:
        try:
            result = await list_function.remote.aio()
            modal_datasets = result.get("datasets", [])
            # Convert to DatasetInfo format
            for ds in modal_datasets:
                dataset_info = DatasetInfo(
                    name=ds.get("name", ""),
                    path=ds.get("path", ""),
                    size=ds.get("size", "Unknown"),
                    type=infer_dataset_type(ds.get("name", "")),
                    created=datetime.fromtimestamp(ds.get("created", time.time())),
                    modified=datetime.fromtimestamp(ds.get("modified", time.time())),
                    source="modal"
                )
                all_datasets.append(dataset_info)
        except Exception as e:
            print(f"Error fetching Modal datasets: {e}")
    
    # Filter by type if specified
    if type:
        all_datasets = [ds for ds in all_datasets if ds.type == type]
    
    # Sort by creation time (newest first)
    all_datasets.sort(key=lambda x: x.created, reverse=True)
    
    # Apply pagination
    total_datasets = len(all_datasets)
    datasets_page = all_datasets[offset:offset + limit]
    
    return {
        "datasets": datasets_page,
        "total": total_datasets,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total_datasets,
        "sources": {
            "uploaded": len(uploaded_datasets),
            "modal": len(modal_datasets)
        }
    }

@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file to upload"),
    name: Optional[str] = Form(None, description="Custom name for the dataset"),
    description: Optional[str] = Form(None, description="Dataset description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags")
):
    """
    Upload a new dataset file
    
    Supports various formats including JSONL, JSON, CSV, TSV, and TXT.
    Automatically validates the file format and content.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Generate unique dataset ID
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    # Use provided name or derive from filename
    dataset_name = name or file.filename
    
    # Infer dataset type
    dataset_type = infer_dataset_type(file.filename)
    
    # Create upload directory
    upload_dir = PathLib("/tmp/datasets")
    upload_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = upload_dir / f"{dataset_id}_{file.filename}"
    
    try:
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Get file size
        file_size = get_file_size_str(str(file_path))
        
        # Validate dataset
        validation_result = validate_dataset_file(str(file_path), dataset_type)
        
        # Extract row count if available
        rows = None
        columns = None
        if dataset_type == DatasetType.JSONL:
            rows = validation_result.statistics.get("valid_lines")
        elif dataset_type in [DatasetType.CSV, DatasetType.TSV]:
            rows = validation_result.statistics.get("sample_rows")
            columns = validation_result.statistics.get("column_names", [])
        elif dataset_type == DatasetType.JSON:
            rows = validation_result.statistics.get("total_records")
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Create dataset metadata
        dataset_info = DatasetInfo(
            name=dataset_name,
            path=str(file_path),
            size=file_size,
            type=dataset_type,
            created=datetime.now(),
            modified=datetime.now(),
            rows=rows,
            columns=columns,
            description=description,
            tags=tag_list,
            source="upload",
            is_valid=validation_result.is_valid,
            validation_errors=validation_result.errors
        )
        
        # Store metadata
        datasets_metadata[dataset_id] = dataset_info
        
        return DatasetUploadResponse(
            success=True,
            dataset_id=dataset_id,
            name=dataset_name,
            path=str(file_path),
            size=file_size,
            type=dataset_type,
            rows=rows,
            message="Dataset uploaded successfully" if validation_result.is_valid else "Dataset uploaded with validation errors",
            validation_errors=validation_result.errors
        )
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str = Path(..., description="Dataset ID")):
    """
    Get detailed information about a specific dataset
    
    Returns comprehensive metadata including validation status,
    column information, and statistics.
    """
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return datasets_metadata[dataset_id]

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str = Path(..., description="Dataset ID")):
    """
    Delete a dataset and its associated files
    
    Permanently removes the dataset file and metadata.
    """
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_metadata[dataset_id]
    
    try:
        # Remove file if it exists
        if os.path.exists(dataset.path):
            os.remove(dataset.path)
        
        # Remove metadata
        del datasets_metadata[dataset_id]
        
        return {
            "success": True,
            "message": f"Dataset '{dataset.name}' deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

@router.post("/{dataset_id}/validate", response_model=DatasetValidationResult)
async def validate_dataset(dataset_id: str = Path(..., description="Dataset ID")):
    """
    Validate a dataset file format and content
    
    Performs comprehensive validation including format checking,
    content analysis, and statistical summary.
    """
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_metadata[dataset_id]
    
    if not os.path.exists(dataset.path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    validation_result = validate_dataset_file(dataset.path, dataset.type)
    
    # Update metadata with validation results
    dataset.is_valid = validation_result.is_valid
    dataset.validation_errors = validation_result.errors
    
    return validation_result

@router.get("/{dataset_id}/download")
async def download_dataset(dataset_id: str = Path(..., description="Dataset ID")):
    """
    Download a dataset file
    
    Returns the original dataset file for download.
    """
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_metadata[dataset_id]
    
    if not os.path.exists(dataset.path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    return FileResponse(
        path=dataset.path,
        filename=dataset.name,
        media_type='application/octet-stream'
    )

@router.get("/{dataset_id}/preview", response_model=Dict[str, Any])
async def preview_dataset(
    dataset_id: str = Path(..., description="Dataset ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of rows to preview")
):
    """
    Preview dataset content
    
    Returns a sample of the dataset content for inspection.
    """
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_metadata[dataset_id]
    
    if not os.path.exists(dataset.path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    try:
        preview_data = []
        
        if dataset.type == DatasetType.JSONL:
            with open(dataset.path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    try:
                        preview_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
                        
        elif dataset.type == DatasetType.JSON:
            with open(dataset.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    preview_data = data[:limit]
                else:
                    preview_data = [data]
                    
        elif dataset.type in [DatasetType.CSV, DatasetType.TSV]:
            separator = ',' if dataset.type == DatasetType.CSV else '\t'
            df = pd.read_csv(dataset.path, sep=separator, nrows=limit)
            preview_data = df.to_dict('records')
            
        elif dataset.type == DatasetType.TXT:
            with open(dataset.path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                preview_data = [{"line": i+1, "content": line.strip()} for i, line in enumerate(lines[:limit])]
        
        return {
            "dataset_id": dataset_id,
            "name": dataset.name,
            "type": dataset.type,
            "preview": preview_data,
            "total_shown": len(preview_data),
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview dataset: {str(e)}")

# Model listing endpoints
@router.get("/models/available", response_model=Dict[str, Any])
async def list_available_models(
    source: Optional[ModelSource] = Query(None, description="Filter by model source"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of models to return"),
    search: Optional[str] = Query(None, description="Search models by name or description")
):
    """
    List all available models for fine-tuning
    
    Returns models from various sources including HuggingFace,
    Modal volumes, and local storage.
    """
    all_models = []
    
    # Popular HuggingFace models for fine-tuning
    if source is None or source == ModelSource.HUGGINGFACE:
        huggingface_models = [
            ModelInfo(
                name="microsoft/DialoGPT-small",
                source=ModelSource.HUGGINGFACE,
                description="Small conversational AI model",
                architecture="GPT-2",
                parameters="117M",
                tags=["conversational", "gpt2", "small"]
            ),
            ModelInfo(
                name="microsoft/DialoGPT-medium",
                source=ModelSource.HUGGINGFACE,
                description="Medium conversational AI model",
                architecture="GPT-2",
                parameters="345M",
                tags=["conversational", "gpt2", "medium"]
            ),
            ModelInfo(
                name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                source=ModelSource.HUGGINGFACE,
                description="Compact LLaMA model for chat",
                architecture="LLaMA",
                parameters="1.1B",
                tags=["llama", "chat", "efficient"]
            ),
            ModelInfo(
                name="meta-llama/Llama-2-7b-chat-hf",
                source=ModelSource.HUGGINGFACE,
                description="LLaMA 2 7B Chat model",
                architecture="LLaMA",
                parameters="7B",
                tags=["llama", "chat", "large"]
            ),
            ModelInfo(
                name="mistralai/Mistral-7B-Instruct-v0.1",
                source=ModelSource.HUGGINGFACE,
                description="Mistral 7B Instruct model",
                architecture="Mistral",
                parameters="7B",
                tags=["mistral", "instruct", "efficient"]
            ),
            ModelInfo(
                name="google/flan-t5-small",
                source=ModelSource.HUGGINGFACE,
                description="FLAN-T5 small model",
                architecture="T5",
                parameters="80M",
                tags=["t5", "flan", "small", "instruction"]
            ),
            ModelInfo(
                name="google/flan-t5-base",
                source=ModelSource.HUGGINGFACE,
                description="FLAN-T5 base model",
                architecture="T5",
                parameters="250M",
                tags=["t5", "flan", "base", "instruction"]
            )
        ]
        all_models.extend(huggingface_models)
    
    # Get Modal volume models
    if (source is None or source == ModelSource.MODAL_VOLUME) and list_function:
        try:
            result = await list_function.remote.aio()
            modal_models = result.get("models", [])
            for model in modal_models:
                model_info = ModelInfo(
                    name=model.get("name", ""),
                    source=ModelSource.MODAL_VOLUME,
                    path=model.get("path", ""),
                    description=model.get("description", "Fine-tuned model"),
                    size=model.get("size", "Unknown"),
                    created=datetime.fromtimestamp(model.get("created", time.time())),
                    is_finetuned=True,
                    tags=["finetuned", "modal"]
                )
                all_models.append(model_info)
        except Exception as e:
            print(f"Error fetching Modal models: {e}")
    
    # Apply search filter
    if search:
        search_lower = search.lower()
        all_models = [
            model for model in all_models
            if search_lower in model.name.lower() or 
               (model.description and search_lower in model.description.lower())
        ]
    
    # Apply pagination
    total_models = len(all_models)
    models_page = all_models[:limit]
    
    return {
        "models": models_page,
        "total": total_models,
        "limit": limit,
        "search": search,
        "sources": {
            "huggingface": len([m for m in all_models if m.source == ModelSource.HUGGINGFACE]),
            "modal_volume": len([m for m in all_models if m.source == ModelSource.MODAL_VOLUME])
        }
    }

@router.get("/models/popular", response_model=List[ModelInfo])
async def get_popular_models():
    """
    Get a curated list of popular models for fine-tuning
    
    Returns commonly used models optimized for fine-tuning tasks.
    """
    popular_models = [
        ModelInfo(
            name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            source=ModelSource.HUGGINGFACE,
            description="Efficient small language model, perfect for experimentation",
            architecture="LLaMA",
            parameters="1.1B",
            tags=["recommended", "efficient", "chat"],
            downloads=50000,
            likes=1200
        ),
        ModelInfo(
            name="microsoft/DialoGPT-medium",
            source=ModelSource.HUGGINGFACE,
            description="Balanced conversational model with good performance",
            architecture="GPT-2",
            parameters="345M",
            tags=["recommended", "conversational", "balanced"],
            downloads=75000,
            likes=2100
        ),
        ModelInfo(
            name="google/flan-t5-base",
            source=ModelSource.HUGGINGFACE,
            description="Instruction-tuned T5 model, excellent for task-specific training",
            architecture="T5",
            parameters="250M",
            tags=["recommended", "instruction", "versatile"],
            downloads=120000,
            likes=3500
        )
    ]
    
    return popular_models

@router.get("/huggingface/search", response_model=Dict[str, Any])
async def search_huggingface_datasets(
    query: str = Query(..., description="Search query for datasets"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results")
):
    """
    Search HuggingFace datasets (mock implementation)
    
    In a real implementation, this would use the HuggingFace API
    to search for datasets matching the query.
    """
    # Mock HuggingFace dataset search results
    mock_results = [
        HuggingFaceDataset(
            id="squad",
            name="Stanford Question Answering Dataset",
            description="Reading comprehension dataset",
            downloads=50000,
            likes=1250,
            tags=["question-answering", "english"],
            size="90MB",
            formats=["json", "parquet"]
        ),
        HuggingFaceDataset(
            id="imdb",
            name="IMDB Movie Reviews",
            description="Binary sentiment classification dataset",
            downloads=75000,
            likes=2100,
            tags=["sentiment", "classification", "english"],
            size="130MB",
            formats=["csv", "parquet"]
        ),
        HuggingFaceDataset(
            id="alpaca",
            name="Alpaca Dataset",
            description="Instruction-following dataset",
            downloads=25000,
            likes=850,
            tags=["instruction", "chat", "english"],
            size="45MB",
            formats=["json", "jsonl"]
        )
    ]
    
    # Filter by query (simple string matching)
    filtered_results = [
        result for result in mock_results
        if query.lower() in result.name.lower() or 
           query.lower() in result.description.lower() or
           any(query.lower() in tag.lower() for tag in result.tags)
    ]
    
    return {
        "query": query,
        "results": filtered_results[:limit],
        "total": len(filtered_results),
        "limit": limit
    }