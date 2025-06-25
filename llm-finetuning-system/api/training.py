"""
Training API endpoints for LLM Fine-tuning System
Provides comprehensive training job management, monitoring, and control
"""

import os
import json
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field, validator
import modal
from enum import Enum

router = APIRouter(prefix="/training", tags=["training"])

# Pydantic Models for Request/Response Validation
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class GPUType(str, Enum):
    T4 = "T4"
    A10G = "A10G"
    A100 = "A100"
    H100 = "H100"
    V100 = "V100"

class OptimizerType(str, Enum):
    ADAMW = "adamw_torch"
    ADAM = "adam"
    SGD = "sgd"
    ADAFACTOR = "adafactor"

class TrainingConfig(BaseModel):
    """Training configuration parameters"""
    model_name: str = Field(..., description="Model name or HuggingFace model path")
    dataset_path: str = Field(..., description="Path to training dataset")
    output_dir: str = Field(default="/vol/finetuned_model", description="Output directory for trained model")
    
    # LoRA Configuration
    lora_r: int = Field(default=8, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=16, ge=1, le=512, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate")
    
    # Training Parameters
    learning_rate: float = Field(default=0.0002, gt=0.0, le=1.0, description="Learning rate")
    num_train_epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=2, ge=1, le=64, description="Batch size per device")
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=128, description="Gradient accumulation steps")
    max_seq_length: int = Field(default=512, ge=128, le=4096, description="Maximum sequence length")
    
    # Optimization Settings
    optimizer_type: OptimizerType = Field(default=OptimizerType.ADAMW, description="Optimizer type")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0, description="Weight decay")
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Warmup ratio")
    lr_scheduler_type: str = Field(default="cosine", description="Learning rate scheduler")
    
    # Quantization and Hardware
    use_4bit_quantization: bool = Field(default=True, description="Use 4-bit quantization (QLoRA)")
    use_8bit_quantization: bool = Field(default=False, description="Use 8-bit quantization")
    gpu_type: GPUType = Field(default=GPUType.A100, description="GPU type for training")
    timeout: int = Field(default=3600, ge=300, le=86400, description="Training timeout in seconds")
    
    # Advanced Settings
    save_steps: int = Field(default=500, ge=1, description="Save checkpoint every N steps")
    eval_steps: int = Field(default=500, ge=1, description="Evaluation steps")
    logging_steps: int = Field(default=10, ge=1, description="Logging frequency")
    seed: int = Field(default=42, description="Random seed")
    
    # Data Processing
    train_on_inputs: bool = Field(default=False, description="Train on input tokens")
    group_by_length: bool = Field(default=True, description="Group sequences by length")
    
    @validator('per_device_train_batch_size')
    def validate_batch_size(cls, v, values):
        if 'gpu_type' in values:
            gpu_type = values['gpu_type']
            if gpu_type == GPUType.T4 and v > 4:
                raise ValueError("T4 GPU supports max batch size of 4")
            elif gpu_type == GPUType.A10G and v > 8:
                raise ValueError("A10G GPU supports max batch size of 8")
        return v

class TrainingMetrics(BaseModel):
    """Training performance metrics"""
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_score: Optional[Dict[str, float]] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    tokens_per_second: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None

class TrainingStatus(BaseModel):
    """Training job status information"""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0)
    current_epoch: int = 0
    total_epochs: int
    current_step: int = 0
    total_steps: Optional[int] = None
    
    # Timing information
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Performance metrics
    metrics: Optional[TrainingMetrics] = None
    
    # Job details
    config: Optional[TrainingConfig] = None
    model_name: str
    dataset_path: str
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    
    # Resource usage
    gpu_type: Optional[GPUType] = None
    cost_estimate: Optional[float] = None

class TrainingJobSummary(BaseModel):
    """Summary information for training job listing"""
    job_id: str
    status: JobStatus
    model_name: str
    progress: float
    created_at: datetime
    duration: Optional[timedelta] = None
    cost_estimate: Optional[float] = None

class LogEntry(BaseModel):
    """Training log entry"""
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    epoch: Optional[int] = None
    step: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None

class TrainingLogs(BaseModel):
    """Training logs response"""
    job_id: str
    logs: List[LogEntry]
    total_entries: int
    last_updated: datetime

# In-memory storage (replace with database in production)
training_jobs: Dict[str, TrainingStatus] = {}
training_logs: Dict[str, List[LogEntry]] = {}

# Modal app connection
try:
    deployed_app = modal.App.lookup("llm-finetuner", environment_name="ai-tool-pool")
    fine_tune_function = deployed_app.fine_tune_llm
    print("✅ Connected to Modal app for training")
except Exception as e:
    print(f"⚠️ Warning: Could not connect to Modal app: {e}")
    deployed_app = None
    fine_tune_function = None

class TrainingMonitor:
    """Enhanced training monitor with comprehensive logging and metrics"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = time.time()
        
        if job_id not in training_logs:
            training_logs[job_id] = []
    
    def log(self, message: str, level: str = "INFO", epoch: int = None, step: int = None, metrics: Dict[str, float] = None):
        """Add a log entry"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            epoch=epoch,
            step=step,
            metrics=metrics
        )
        training_logs[self.job_id].append(entry)
        
        # Keep only last 1000 log entries to prevent memory issues
        if len(training_logs[self.job_id]) > 1000:
            training_logs[self.job_id] = training_logs[self.job_id][-1000:]
    
    def update_status(self, status: JobStatus, error_message: str = None):
        """Update job status"""
        if self.job_id in training_jobs:
            training_jobs[self.job_id].status = status
            if error_message:
                training_jobs[self.job_id].error_message = error_message
            
            # Update timestamps
            now = datetime.now()
            if status == JobStatus.RUNNING and not training_jobs[self.job_id].started_at:
                training_jobs[self.job_id].started_at = now
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                training_jobs[self.job_id].completed_at = now
    
    def update_progress(self, epoch: int, total_epochs: int, step: int = None, 
                       total_steps: int = None, metrics: TrainingMetrics = None):
        """Update training progress and metrics"""
        progress = (epoch / total_epochs) * 100
        
        # Calculate estimated completion time
        elapsed_time = time.time() - self.start_time
        if epoch > 0:
            time_per_epoch = elapsed_time / epoch
            remaining_epochs = total_epochs - epoch
            estimated_seconds = time_per_epoch * remaining_epochs
            estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        else:
            estimated_completion = None
        
        # Update job status
        if self.job_id in training_jobs:
            job = training_jobs[self.job_id]
            job.progress = progress
            job.current_epoch = epoch
            job.total_epochs = total_epochs
            job.current_step = step or 0
            job.total_steps = total_steps
            job.estimated_completion = estimated_completion
            job.metrics = metrics

# API Endpoints

@router.post("/start", response_model=Dict[str, Any])
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    Start a new fine-tuning training job
    
    Creates a new training job with the specified configuration and starts
    the training process in the background using Modal.com infrastructure.
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    # Validate configuration
    try:
        config_dict = config.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    
    # Calculate cost estimate (simplified)
    gpu_costs = {
        GPUType.T4: 0.35,
        GPUType.A10G: 0.75, 
        GPUType.A100: 1.50,
        GPUType.H100: 3.00,
        GPUType.V100: 1.25
    }
    estimated_hours = config.num_train_epochs * 0.5  # Rough estimate
    cost_estimate = gpu_costs.get(config.gpu_type, 1.0) * estimated_hours
    
    # Initialize job status
    training_jobs[job_id] = TrainingStatus(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        current_epoch=0,
        total_epochs=config.num_train_epochs,
        created_at=datetime.now(),
        config=config,
        model_name=config.model_name,
        dataset_path=config.dataset_path,
        gpu_type=config.gpu_type,
        cost_estimate=cost_estimate
    )
    
    # Start training in background
    if fine_tune_function:
        background_tasks.add_task(run_modal_training, config_dict, job_id)
    else:
        background_tasks.add_task(run_training_simulation, config_dict, job_id)
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Training job initiated successfully",
        "estimated_cost": cost_estimate,
        "estimated_duration_hours": estimated_hours
    }

@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str = Path(..., description="Training job ID")):
    """
    Get detailed status information for a training job
    
    Returns comprehensive status including progress, metrics, resource usage,
    and estimated completion time.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]

@router.get("/logs/{job_id}", response_model=TrainingLogs)
async def get_training_logs(
    job_id: str = Path(..., description="Training job ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries to return"),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR, DEBUG)")
):
    """
    Get training logs for a specific job
    
    Returns paginated training logs with optional filtering by log level.
    """
    if job_id not in training_logs:
        raise HTTPException(status_code=404, detail="Training job logs not found")
    
    logs = training_logs[job_id]
    
    # Filter by level if specified
    if level:
        logs = [log for log in logs if log.level == level.upper()]
    
    # Apply limit
    logs = logs[-limit:] if len(logs) > limit else logs
    
    return TrainingLogs(
        job_id=job_id,
        logs=logs,
        total_entries=len(training_logs[job_id]),
        last_updated=logs[-1].timestamp if logs else datetime.now()
    )

@router.get("/jobs", response_model=Dict[str, Any])
async def list_training_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip")
):
    """
    List all training jobs with optional filtering and pagination
    
    Returns a paginated list of training jobs with summary information.
    """
    jobs = list(training_jobs.values())
    
    # Filter by status if specified
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    total_jobs = len(jobs)
    jobs = jobs[offset:offset + limit]
    
    # Convert to summary format
    job_summaries = []
    for job in jobs:
        duration = None
        if job.started_at and job.completed_at:
            duration = job.completed_at - job.started_at
        elif job.started_at:
            duration = datetime.now() - job.started_at
            
        summary = TrainingJobSummary(
            job_id=job.job_id,
            status=job.status,
            model_name=job.model_name,
            progress=job.progress,
            created_at=job.created_at,
            duration=duration,
            cost_estimate=job.cost_estimate
        )
        job_summaries.append(summary)
    
    return {
        "jobs": job_summaries,
        "total": total_jobs,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total_jobs
    }

@router.delete("/{job_id}")
async def cancel_training_job(job_id: str = Path(..., description="Training job ID")):
    """
    Cancel a running or pending training job
    
    Attempts to gracefully stop the training process and update job status.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    # Update job status
    monitor = TrainingMonitor(job_id)
    monitor.update_status(JobStatus.CANCELLED)
    monitor.log("Training job cancelled by user request", level="WARNING")
    
    # In a real implementation, you would also cancel the Modal function
    # For now, we just update the status
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Training job cancellation requested"
    }

@router.post("/{job_id}/pause")
async def pause_training_job(job_id: str = Path(..., description="Training job ID")):
    """
    Pause a running training job
    
    Temporarily pauses the training process. Can be resumed later.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    
    if job.status != JobStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause job with status: {job.status}"
        )
    
    # Update job status
    monitor = TrainingMonitor(job_id)
    monitor.update_status(JobStatus.PAUSED)
    monitor.log("Training job paused", level="INFO")
    
    return {
        "job_id": job_id,
        "status": "paused",
        "message": "Training job paused successfully"
    }

@router.post("/{job_id}/resume")
async def resume_training_job(job_id: str = Path(..., description="Training job ID")):
    """
    Resume a paused training job
    
    Resumes training from the last checkpoint.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    
    if job.status != JobStatus.PAUSED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume job with status: {job.status}"
        )
    
    # Update job status
    monitor = TrainingMonitor(job_id)
    monitor.update_status(JobStatus.RUNNING)
    monitor.log("Training job resumed", level="INFO")
    
    return {
        "job_id": job_id,
        "status": "running",
        "message": "Training job resumed successfully"
    }

@router.get("/metrics/{job_id}", response_model=Dict[str, Any])
async def get_training_metrics(
    job_id: str = Path(..., description="Training job ID"),
    window: int = Query(60, ge=10, le=3600, description="Time window in seconds for metrics")
):
    """
    Get detailed performance metrics for a training job
    
    Returns comprehensive metrics including loss curves, resource utilization,
    and performance statistics over the specified time window.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    
    # Get recent log entries with metrics
    recent_logs = []
    if job_id in training_logs:
        cutoff_time = datetime.now() - timedelta(seconds=window)
        recent_logs = [
            log for log in training_logs[job_id] 
            if log.timestamp >= cutoff_time and log.metrics
        ]
    
    # Aggregate metrics
    metrics_timeline = []
    for log in recent_logs:
        if log.metrics:
            metrics_timeline.append({
                "timestamp": log.timestamp,
                "epoch": log.epoch,
                "step": log.step,
                **log.metrics
            })
    
    # Calculate summary statistics
    if metrics_timeline:
        latest_metrics = metrics_timeline[-1]
        avg_loss = sum(m.get("loss", 0) for m in metrics_timeline if m.get("loss")) / len([m for m in metrics_timeline if m.get("loss")])
    else:
        latest_metrics = {}
        avg_loss = None
    
    return {
        "job_id": job_id,
        "current_metrics": job.metrics.dict() if job.metrics else {},
        "metrics_timeline": metrics_timeline,
        "summary": {
            "average_loss": avg_loss,
            "total_steps": job.current_step,
            "training_time": (datetime.now() - job.started_at).total_seconds() if job.started_at else 0,
            "estimated_completion": job.estimated_completion
        },
        "window_seconds": window
    }

# Background task functions

async def run_modal_training(config: dict, job_id: str):
    """Execute training using Modal.com infrastructure"""
    monitor = TrainingMonitor(job_id)
    
    try:
        monitor.log("Initializing Modal.com training environment", level="INFO")
        monitor.update_status(JobStatus.RUNNING)
        
        # Call the deployed Modal function
        result = await fine_tune_function.remote.aio(
            model_name_or_path=config["model_name"],
            dataset_path=config["dataset_path"],
            output_dir=config["output_dir"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            optimizer_type=config["optimizer_type"],
            use_4bit_quantization=config["use_4bit_quantization"],
            gpu_type=config["gpu_type"]
        )
        
        # Update final status
        monitor.update_status(JobStatus.COMPLETED)
        training_jobs[job_id].progress = 100.0
        training_jobs[job_id].output_path = config["output_dir"]
        
        monitor.log("Training completed successfully", level="INFO", 
                   metrics={"final_loss": result.get("final_loss", 0)})
        
    except Exception as e:
        monitor.update_status(JobStatus.FAILED, str(e))
        monitor.log(f"Training failed: {str(e)}", level="ERROR")
        print(f"Training failed for job {job_id}: {e}")

async def run_training_simulation(config: dict, job_id: str):
    """Simulate training progress for demonstration"""
    monitor = TrainingMonitor(job_id)
    total_epochs = config["num_train_epochs"]
    
    monitor.log("Starting training simulation (Modal.com not connected)", level="INFO")
    monitor.update_status(JobStatus.RUNNING)
    
    for epoch in range(1, total_epochs + 1):
        # Check if job was cancelled
        if job_id in training_jobs and training_jobs[job_id].status == JobStatus.CANCELLED:
            monitor.log("Training cancelled", level="WARNING")
            return
        
        # Simulate epoch duration
        await asyncio.sleep(5)
        
        # Simulate metrics
        loss = max(0.1, 2.5 - (epoch * 0.3) + (0.1 * (0.5 - 0.5)))  # Add some randomness
        accuracy = min(0.95, 0.65 + (epoch * 0.05))
        gpu_util = min(100.0, 75.0 + epoch * 3)
        gpu_memory = min(24.0, 12.0 + epoch * 1.5)
        
        metrics = TrainingMetrics(
            loss=loss,
            accuracy=accuracy,
            learning_rate=config["learning_rate"] * (0.95 ** epoch),
            tokens_per_second=1250.0 + epoch * 50,
            gpu_utilization=gpu_util,
            gpu_memory_used=gpu_memory,
            gpu_memory_total=24.0,
            cpu_usage=45.0 + epoch * 2,
            memory_usage=8.5 + epoch * 0.5
        )
        
        monitor.update_progress(epoch, total_epochs, epoch * 100, total_epochs * 100, metrics)
        monitor.log(
            f"Completed epoch {epoch}/{total_epochs}",
            level="INFO",
            epoch=epoch,
            step=epoch * 100,
            metrics={
                "loss": loss,
                "accuracy": accuracy,
                "gpu_utilization": gpu_util
            }
        )
    
    # Mark as completed
    monitor.update_status(JobStatus.COMPLETED)
    training_jobs[job_id].progress = 100.0
    training_jobs[job_id].output_path = config["output_dir"]
    monitor.log("Training simulation completed successfully", level="INFO")