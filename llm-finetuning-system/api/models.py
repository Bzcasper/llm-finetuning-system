"""
Pydantic models for the LLM Fine-Tuning Studio API
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import time


class JobStatus(str, Enum):
    """Training job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GPUType(str, Enum):
    """Available GPU types for training"""
    T4 = "T4"
    A100 = "A100"
    H100 = "H100"
    V100 = "V100"


class OptimizerType(str, Enum):
    """Available optimizer types"""
    ADAMW = "adamw_torch"
    ADAM = "adam"
    SGD = "sgd"
    ADAFACTOR = "adafactor"


class TrainingConfig(BaseModel):
    """Configuration for fine-tuning job"""
    model_name: str = Field(..., description="Model name or path (HuggingFace or local)")
    dataset_path: str = Field(..., description="Path to training dataset")
    output_dir: str = Field("/vol/finetuned_model", description="Output directory for trained model")
    
    # LoRA configuration
    lora_r: int = Field(8, ge=1, le=128, description="LoRA rank")
    lora_alpha: int = Field(16, ge=1, le=256, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0, description="LoRA dropout rate")
    
    # Training parameters
    learning_rate: float = Field(2e-4, gt=0.0, le=1.0, description="Learning rate")
    num_train_epochs: int = Field(3, ge=1, le=100, description="Number of training epochs")
    per_device_train_batch_size: int = Field(2, ge=1, le=32, description="Batch size per device")
    gradient_accumulation_steps: int = Field(1, ge=1, le=64, description="Gradient accumulation steps")
    
    # Optimizer and quantization
    optimizer_type: OptimizerType = Field(OptimizerType.ADAMW, description="Optimizer type")
    use_4bit_quantization: bool = Field(False, description="Enable 4-bit quantization (QLoRA)")
    
    # Infrastructure
    gpu_type: GPUType = Field(GPUType.A100, description="GPU type for training")
    timeout: int = Field(3600, ge=300, le=86400, description="Training timeout in seconds")
    
    # Optional advanced parameters
    warmup_steps: Optional[int] = Field(None, ge=0, description="Number of warmup steps")
    weight_decay: Optional[float] = Field(0.01, ge=0.0, le=1.0, description="Weight decay")
    max_grad_norm: Optional[float] = Field(1.0, gt=0.0, description="Maximum gradient norm")
    save_steps: Optional[int] = Field(500, ge=1, description="Save checkpoint every N steps")
    logging_steps: Optional[int] = Field(10, ge=1, description="Log metrics every N steps")

    @validator('dataset_path')
    def validate_dataset_path(cls, v):
        if not v or not v.strip():
            raise ValueError("Dataset path cannot be empty")
        return v.strip()

    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class TrainingMetrics(BaseModel):
    """Training metrics and progress information"""
    loss: Optional[float] = Field(None, description="Current training loss")
    accuracy: Optional[float] = Field(None, description="Current accuracy")
    learning_rate: Optional[float] = Field(None, description="Current learning rate")
    gpu_utilization: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU utilization %")
    gpu_memory_used: Optional[float] = Field(None, ge=0.0, description="GPU memory used (GB)")
    samples_per_second: Optional[float] = Field(None, gt=0.0, description="Training samples per second")


class TrainingStatus(BaseModel):
    """Status and progress of a training job"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Training progress percentage")
    current_epoch: int = Field(0, ge=0, description="Current training epoch")
    total_epochs: int = Field(1, ge=1, description="Total number of epochs")
    current_step: Optional[int] = Field(None, ge=0, description="Current training step")
    total_steps: Optional[int] = Field(None, ge=1, description="Total training steps")
    
    # Metrics
    metrics: Optional[TrainingMetrics] = Field(None, description="Training metrics")
    
    # Time tracking
    start_time: Optional[float] = Field(None, description="Job start timestamp")
    end_time: Optional[float] = Field(None, description="Job end timestamp")
    estimated_time_remaining: Optional[int] = Field(None, ge=0, description="Estimated remaining time (seconds)")
    
    # Configuration
    config: Optional[TrainingConfig] = Field(None, description="Training configuration")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate job duration in seconds"""
        if self.start_time:
            end = self.end_time or time.time()
            return end - self.start_time
        return None


class TrainingJob(BaseModel):
    """Complete training job information"""
    job_id: str = Field(..., description="Unique job identifier")
    config: TrainingConfig = Field(..., description="Training configuration")
    status: TrainingStatus = Field(..., description="Current job status")
    logs: List[str] = Field(default_factory=list, description="Job execution logs")
    created_at: float = Field(default_factory=time.time, description="Job creation timestamp")
    modal_job_id: Optional[str] = Field(None, description="Modal.com job identifier")


class DatasetInfo(BaseModel):
    """Information about a dataset"""
    name: str = Field(..., description="Dataset name")
    path: str = Field(..., description="Dataset file path")
    size: str = Field(..., description="Dataset file size (human readable)")
    type: str = Field(..., description="Dataset file type")
    created: Optional[float] = Field(None, description="Creation timestamp")
    num_samples: Optional[int] = Field(None, ge=0, description="Number of samples in dataset")
    description: Optional[str] = Field(None, description="Dataset description")


class ModelInfo(BaseModel):
    """Information about a model"""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (huggingface, volume, local)")
    path: Optional[str] = Field(None, description="Local path if applicable")
    size: Optional[str] = Field(None, description="Model size (human readable)")
    description: Optional[str] = Field(None, description="Model description")
    parameters: Optional[str] = Field(None, description="Number of parameters")
    architecture: Optional[str] = Field(None, description="Model architecture")


class ModalStatus(BaseModel):
    """Modal.com connection and status information"""
    connected: bool = Field(..., description="Whether Modal.com is connected")
    app_name: Optional[str] = Field(None, description="Modal app name")
    environment: Optional[str] = Field(None, description="Modal environment name")
    app_status: str = Field(..., description="Modal app deployment status")
    error: Optional[str] = Field(None, description="Error message if not connected")
    timestamp: float = Field(default_factory=time.time, description="Status check timestamp")
    credentials_configured: bool = Field(False, description="Whether Modal credentials are configured")
    functions_available: Optional[Dict[str, bool]] = Field(None, description="Available Modal functions")


class HealthStatus(BaseModel):
    """System health status"""
    status: str = Field("healthy", description="Overall system status")
    timestamp: float = Field(default_factory=time.time, description="Health check timestamp")
    modal_connected: bool = Field(False, description="Modal.com connection status")
    version: str = Field("1.0.0", description="API version")
    environment: str = Field("development", description="Runtime environment")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")


class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = Field(True, description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")


class TrainingStartResponse(BaseModel):
    """Response for training job start"""
    job_id: str = Field(..., description="Created job identifier")
    status: str = Field("started", description="Initial job status")
    message: str = Field("Training job started successfully", description="Success message")
    estimated_duration: Optional[int] = Field(None, description="Estimated training duration (seconds)")


class LogEntry(BaseModel):
    """Individual log entry"""
    timestamp: str = Field(..., description="Log entry timestamp")
    level: str = Field("INFO", description="Log level")
    message: str = Field(..., description="Log message")
    source: Optional[str] = Field(None, description="Log source component")


class JobLogsResponse(BaseModel):
    """Response for job logs request"""
    job_id: str = Field(..., description="Job identifier")
    logs: List[LogEntry] = Field(..., description="Job log entries")
    total_entries: int = Field(..., description="Total log entries")
    last_updated: float = Field(default_factory=time.time, description="Last log update timestamp")