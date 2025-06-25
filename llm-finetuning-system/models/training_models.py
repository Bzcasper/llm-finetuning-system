"""
Comprehensive data models for training job management and orchestration
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid
import json


class TrainingPhase(str, Enum):
    """Training phase enumeration"""
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    LOADING_DATASET = "loading_dataset"
    TRAINING = "training"
    VALIDATION = "validation"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class GPUType(str, Enum):
    """Available GPU types"""
    T4 = "T4"
    A10G = "A10G"
    A100 = "A100"
    H100 = "H100"
    V100 = "V100"


class OptimizerType(str, Enum):
    """Available optimizer types"""
    ADAMW = "adamw_torch"
    ADAMW_HF = "adamw_hf"
    SGD = "sgd"
    ADAFACTOR = "adafactor"


class SchedulerType(str, Enum):
    """Available learning rate schedulers"""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class TrainingConfig(BaseModel):
    """Enhanced training configuration with validation"""
    
    # Model configuration
    model_name: str = Field(..., description="Model name or path")
    model_revision: Optional[str] = Field(None, description="Model revision/branch")
    trust_remote_code: bool = Field(False, description="Trust remote code execution")
    
    # Dataset configuration
    dataset_path: str = Field(..., description="Dataset path or HuggingFace dataset name")
    dataset_split: str = Field("train", description="Dataset split to use")
    validation_split: Optional[str] = Field(None, description="Validation split")
    max_samples: Optional[int] = Field(None, description="Maximum samples to use")
    text_column: str = Field("text", description="Text column name")
    
    # Output configuration
    output_dir: str = Field("/vol/finetuned_model", description="Output directory")
    save_strategy: str = Field("epoch", description="Save strategy")
    save_steps: int = Field(500, description="Save steps")
    save_total_limit: int = Field(3, description="Total save limit")
    
    # LoRA configuration
    use_lora: bool = Field(True, description="Use LoRA fine-tuning")
    lora_r: int = Field(8, ge=1, le=512, description="LoRA rank")
    lora_alpha: int = Field(16, ge=1, le=1024, description="LoRA alpha")
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0, description="LoRA dropout")
    lora_target_modules: Optional[List[str]] = Field(None, description="LoRA target modules")
    
    # Training hyperparameters
    learning_rate: float = Field(0.0002, gt=0.0, le=1.0, description="Learning rate")
    num_train_epochs: int = Field(3, ge=1, le=100, description="Number of training epochs")
    per_device_train_batch_size: int = Field(2, ge=1, le=64, description="Training batch size")
    per_device_eval_batch_size: int = Field(2, ge=1, le=64, description="Evaluation batch size")
    gradient_accumulation_steps: int = Field(1, ge=1, le=64, description="Gradient accumulation steps")
    max_grad_norm: float = Field(1.0, ge=0.0, description="Max gradient norm")
    weight_decay: float = Field(0.01, ge=0.0, le=1.0, description="Weight decay")
    warmup_steps: int = Field(100, ge=0, description="Warmup steps")
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Warmup ratio")
    
    # Optimizer and scheduler
    optimizer_type: OptimizerType = Field(OptimizerType.ADAMW, description="Optimizer type")
    lr_scheduler_type: SchedulerType = Field(SchedulerType.LINEAR, description="LR scheduler type")
    
    # Quantization
    use_4bit_quantization: bool = Field(False, description="Use 4-bit quantization")
    use_8bit_quantization: bool = Field(False, description="Use 8-bit quantization")
    bnb_4bit_compute_dtype: str = Field("float16", description="BNB 4-bit compute dtype")
    bnb_4bit_quant_type: str = Field("nf4", description="BNB 4-bit quantization type")
    
    # Hardware configuration
    gpu_type: GPUType = Field(GPUType.A100, description="GPU type")
    num_gpus: int = Field(1, ge=1, le=8, description="Number of GPUs")
    mixed_precision: str = Field("fp16", description="Mixed precision")
    
    # Monitoring and logging
    logging_steps: int = Field(10, ge=1, description="Logging steps")
    eval_steps: int = Field(500, ge=1, description="Evaluation steps")
    evaluation_strategy: str = Field("steps", description="Evaluation strategy")
    metric_for_best_model: str = Field("eval_loss", description="Metric for best model")
    load_best_model_at_end: bool = Field(True, description="Load best model at end")
    
    # Resource limits
    timeout: int = Field(3600, ge=60, le=86400, description="Timeout in seconds")
    max_memory_gb: Optional[int] = Field(None, ge=4, le=80, description="Max memory in GB")
    
    # Advanced configuration
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")
    dataloader_num_workers: int = Field(4, ge=0, le=16, description="Dataloader workers")
    remove_unused_columns: bool = Field(True, description="Remove unused columns")
    fp16_full_eval: bool = Field(False, description="FP16 full evaluation")
    
    # Custom parameters
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="Custom parameters")
    
    @validator('lora_alpha')
    def validate_lora_alpha(cls, v, values):
        if 'lora_r' in values and v < values['lora_r']:
            raise ValueError("lora_alpha should be >= lora_r")
        return v
    
    @root_validator
    def validate_quantization(cls, values):
        if values.get('use_4bit_quantization') and values.get('use_8bit_quantization'):
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        return values
    
    @validator('warmup_steps')
    def validate_warmup_steps(cls, v, values):
        if 'warmup_ratio' in values and values['warmup_ratio'] > 0 and v > 0:
            raise ValueError("Cannot specify both warmup_steps and warmup_ratio")
        return v


class GPUMetrics(BaseModel):
    """GPU metrics tracking"""
    gpu_utilization: float = Field(0.0, ge=0.0, le=100.0, description="GPU utilization %")
    gpu_memory_used: float = Field(0.0, ge=0.0, description="GPU memory used (GB)")
    gpu_memory_total: float = Field(0.0, ge=0.0, description="GPU memory total (GB)")
    gpu_temperature: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU temperature")
    gpu_power_usage: Optional[float] = Field(None, ge=0.0, description="GPU power usage (W)")
    
    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage"""
        if self.gpu_memory_total > 0:
            return (self.gpu_memory_used / self.gpu_memory_total) * 100
        return 0.0


class ResourceUsage(BaseModel):
    """Resource usage metrics"""
    cpu_percent: float = Field(0.0, ge=0.0, le=100.0, description="CPU usage %")
    memory_used_gb: float = Field(0.0, ge=0.0, description="Memory used (GB)")
    memory_total_gb: float = Field(0.0, ge=0.0, description="Total memory (GB)")
    disk_used_gb: float = Field(0.0, ge=0.0, description="Disk used (GB)")
    disk_total_gb: float = Field(0.0, ge=0.0, description="Total disk (GB)")
    network_sent_mb: float = Field(0.0, ge=0.0, description="Network sent (MB)")
    network_recv_mb: float = Field(0.0, ge=0.0, description="Network received (MB)")
    
    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage"""
        if self.memory_total_gb > 0:
            return (self.memory_used_gb / self.memory_total_gb) * 100
        return 0.0


class TrainingMetrics(BaseModel):
    """Training metrics tracking"""
    epoch: int = Field(0, ge=0, description="Current epoch")
    step: int = Field(0, ge=0, description="Current step")
    total_steps: Optional[int] = Field(None, ge=0, description="Total steps")
    
    # Loss metrics
    train_loss: Optional[float] = Field(None, description="Training loss")
    eval_loss: Optional[float] = Field(None, description="Evaluation loss")
    
    # Performance metrics
    learning_rate: Optional[float] = Field(None, ge=0.0, description="Current learning rate")
    train_runtime: Optional[float] = Field(None, ge=0.0, description="Training runtime")
    train_samples_per_second: Optional[float] = Field(None, ge=0.0, description="Training samples/sec")
    train_steps_per_second: Optional[float] = Field(None, ge=0.0, description="Training steps/sec")
    
    # Evaluation metrics
    eval_runtime: Optional[float] = Field(None, ge=0.0, description="Evaluation runtime")
    eval_samples_per_second: Optional[float] = Field(None, ge=0.0, description="Evaluation samples/sec")
    eval_steps_per_second: Optional[float] = Field(None, ge=0.0, description="Evaluation steps/sec")
    
    # Custom metrics
    perplexity: Optional[float] = Field(None, ge=0.0, description="Perplexity")
    bleu_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="BLEU score")
    rouge_score: Optional[Dict[str, float]] = Field(None, description="ROUGE scores")
    
    # Gradient metrics
    grad_norm: Optional[float] = Field(None, ge=0.0, description="Gradient norm")
    
    # Hardware metrics
    gpu_metrics: Optional[GPUMetrics] = Field(None, description="GPU metrics")
    resource_usage: Optional[ResourceUsage] = Field(None, description="Resource usage")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_steps and self.total_steps > 0:
            return min((self.step / self.total_steps) * 100, 100.0)
        return 0.0


class CostEstimate(BaseModel):
    """Cost estimation for training jobs"""
    gpu_type: GPUType
    gpu_hours: float = Field(0.0, ge=0.0, description="Estimated GPU hours")
    cost_per_hour: float = Field(0.0, ge=0.0, description="Cost per GPU hour")
    estimated_cost: float = Field(0.0, ge=0.0, description="Estimated total cost")
    currency: str = Field("USD", description="Currency")
    
    # Cost breakdown
    compute_cost: float = Field(0.0, ge=0.0, description="Compute cost")
    storage_cost: float = Field(0.0, ge=0.0, description="Storage cost")
    network_cost: float = Field(0.0, ge=0.0, description="Network cost")
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost"""
        return self.compute_cost + self.storage_cost + self.network_cost


class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    value: Any
    error_type: str


class TrainingStatus(BaseModel):
    """Comprehensive training status tracking"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(JobStatus.PENDING, description="Job status")
    phase: TrainingPhase = Field(TrainingPhase.INITIALIZING, description="Training phase")
    
    # Progress tracking
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    current_epoch: int = Field(0, ge=0, description="Current epoch")
    total_epochs: int = Field(0, ge=0, description="Total epochs")
    current_step: int = Field(0, ge=0, description="Current step")
    total_steps: Optional[int] = Field(None, ge=0, description="Total steps")
    
    # Metrics
    current_metrics: Optional[TrainingMetrics] = Field(None, description="Current metrics")
    best_metrics: Optional[TrainingMetrics] = Field(None, description="Best metrics")
    
    # Time tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(None, description="Training start time")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(None, description="Training completion time")
    
    # Time estimates
    estimated_time_remaining: Optional[int] = Field(None, description="ETA in seconds")
    elapsed_time: Optional[int] = Field(None, description="Elapsed time in seconds")
    
    # Error tracking
    error_message: Optional[str] = Field(None, description="Error message")
    error_traceback: Optional[str] = Field(None, description="Error traceback")
    retry_count: int = Field(0, ge=0, description="Number of retries")
    
    # Resource info
    modal_job_id: Optional[str] = Field(None, description="Modal job ID")
    gpu_type: Optional[GPUType] = Field(None, description="GPU type used")
    num_gpus: Optional[int] = Field(None, description="Number of GPUs")
    
    # Cost tracking
    cost_estimate: Optional[CostEstimate] = Field(None, description="Cost estimate")
    actual_cost: Optional[float] = Field(None, description="Actual cost")
    
    # Configuration
    config_hash: Optional[str] = Field(None, description="Configuration hash")
    
    def update_timestamp(self):
        """Update the timestamp"""
        self.updated_at = datetime.now(timezone.utc)
    
    @property
    def is_running(self) -> bool:
        """Check if job is currently running"""
        return self.status == JobStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed"""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    @property
    def duration(self) -> Optional[int]:
        """Get job duration in seconds"""
        if self.started_at:
            end_time = self.completed_at or datetime.now(timezone.utc)
            return int((end_time - self.started_at).total_seconds())
        return None


class TrainingJobCreate(BaseModel):
    """Request model for creating training jobs"""
    config: TrainingConfig
    priority: int = Field(1, ge=1, le=10, description="Job priority (1=lowest, 10=highest)")
    tags: List[str] = Field(default_factory=list, description="Job tags")
    description: Optional[str] = Field(None, description="Job description")
    notification_webhook: Optional[str] = Field(None, description="Webhook for notifications")
    auto_retry: bool = Field(True, description="Auto retry on failure")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retries")


class TrainingJobResponse(BaseModel):
    """Response model for training job operations"""
    job_id: str
    status: JobStatus
    message: str
    created_at: datetime
    config: Optional[TrainingConfig] = None
    cost_estimate: Optional[CostEstimate] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingJobList(BaseModel):
    """Response model for listing training jobs"""
    jobs: List[TrainingStatus]
    total: int
    page: int = 1
    per_page: int = 50
    
    @property
    def has_next(self) -> bool:
        return self.page * self.per_page < self.total
    
    @property
    def has_prev(self) -> bool:
        return self.page > 1


class TrainingLogs(BaseModel):
    """Training logs model"""
    job_id: str
    logs: List[str]
    total_lines: int
    page: int = 1
    per_page: int = 100
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }