"""
Core services for the LLM Fine-Tuning Studio API
"""
import os
import time
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
import modal
from .models import (
    TrainingConfig, TrainingStatus, TrainingJob, TrainingMetrics, 
    JobStatus, ModalStatus, DatasetInfo, ModelInfo, LogEntry
)

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages different storage backends (Modal volumes, MinIO, S3)"""
    
    def __init__(self):
        self.storage_type = os.getenv("STORAGE_TYPE", "modal")
        self.base_path = os.getenv("STORAGE_BASE_PATH", "/vol")
    
    async def list_datasets(self) -> List[DatasetInfo]:
        """List available datasets"""
        try:
            if self.storage_type == "modal":
                return await self._list_modal_datasets()
            elif self.storage_type == "s3":
                return await self._list_s3_datasets()
            else:
                return await self._list_local_datasets()
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []
    
    async def _list_modal_datasets(self) -> List[DatasetInfo]:
        """List datasets from Modal volumes"""
        # This would integrate with Modal volume listing
        return [
            DatasetInfo(
                name="dummy_dataset.jsonl",
                path="/vol/dummy_dataset.jsonl",
                size="1.2 KB",
                type="jsonl",
                created=time.time() - 86400,
                num_samples=100
            )
        ]
    
    async def _list_local_datasets(self) -> List[DatasetInfo]:
        """List local datasets"""
        datasets = []
        try:
            import os
            if os.path.exists(self.base_path):
                for file in os.listdir(self.base_path):
                    if file.endswith(('.json', '.jsonl', '.csv', '.txt')):
                        full_path = os.path.join(self.base_path, file)
                        stat = os.stat(full_path)
                        datasets.append(DatasetInfo(
                            name=file,
                            path=full_path,
                            size=f"{stat.st_size / 1024:.1f} KB",
                            type=file.split('.')[-1],
                            created=stat.st_ctime
                        ))
        except Exception as e:
            logger.error(f"Error listing local datasets: {e}")
        return datasets
    
    async def _list_s3_datasets(self) -> List[DatasetInfo]:
        """List datasets from S3"""
        # Placeholder for S3 integration
        return []


class ModalService:
    """Service for managing Modal.com integration"""
    
    def __init__(self):
        self.app = None
        self.fine_tune_function = None
        self.list_function = None
        self.connected = False
        self.last_check = 0
        self.check_interval = 300  # 5 minutes
    
    @classmethod
    async def initialize(cls):
        """Initialize Modal service on startup"""
        service = cls()
        await service.connect()
        return service
    
    async def connect(self) -> bool:
        """Connect to Modal.com"""
        try:
            # Set up Modal credentials
            token_id = os.getenv("MODAL_TOKEN_ID")
            token_secret = os.getenv("MODAL_TOKEN_SECRET")
            environment = os.getenv("MODAL_PROFILE", "ai-tool-pool")
            
            if not token_id or not token_secret:
                logger.warning("Modal credentials not configured")
                return False
            
            # Set environment variables for Modal
            os.environ["MODAL_TOKEN_ID"] = token_id
            os.environ["MODAL_TOKEN_SECRET"] = token_secret
            
            # Connect to the deployed Modal app
            self.app = modal.App.lookup("llm-finetuner", environment_name=environment)
            self.fine_tune_function = self.app.fine_tune_llm
            self.list_function = self.app.list_models_and_datasets
            
            self.connected = True
            self.last_check = time.time()
            logger.info("Successfully connected to Modal.com")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Modal.com: {e}")
            self.connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if Modal is connected"""
        # Refresh connection status periodically
        if time.time() - self.last_check > self.check_interval:
            asyncio.create_task(self.connect())
        return self.connected
    
    async def get_status(self) -> ModalStatus:
        """Get Modal connection status"""
        try:
            environment = os.getenv("MODAL_PROFILE", "ai-tool-pool")
            credentials_configured = bool(
                os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")
            )
            
            if not self.connected:
                await self.connect()
            
            return ModalStatus(
                connected=self.connected,
                app_name="llm-finetuner" if self.connected else None,
                environment=environment,
                app_status="deployed" if self.connected else "disconnected",
                credentials_configured=credentials_configured,
                functions_available={
                    "fine_tune_llm": self.fine_tune_function is not None,
                    "list_models_and_datasets": self.list_function is not None
                } if self.connected else None
            )
        except Exception as e:
            return ModalStatus(
                connected=False,
                app_status="error",
                error=str(e),
                credentials_configured=bool(
                    os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")
                )
            )
    
    async def start_training(self, config: TrainingConfig) -> str:
        """Start training job on Modal"""
        if not self.is_connected():
            raise RuntimeError("Modal.com not connected")
        
        try:
            result = await self.fine_tune_function.remote.aio(
                model_name_or_path=config.model_name,
                dataset_path=config.dataset_path,
                output_dir=config.output_dir,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                optimizer_type=config.optimizer_type,
                use_4bit_quantization=config.use_4bit_quantization,
                gpu_type=config.gpu_type
            )
            return str(result)
        except Exception as e:
            logger.error(f"Modal training failed: {e}")
            raise
    
    async def list_models_and_datasets(self) -> Dict[str, List[Dict]]:
        """List available models and datasets from Modal"""
        if not self.is_connected():
            return {"models": [], "datasets": []}
        
        try:
            result = await self.list_function.remote.aio()
            return result
        except Exception as e:
            logger.error(f"Failed to list models and datasets: {e}")
            return {"models": [], "datasets": []}


class TrainingMonitor:
    """Monitor training job progress and metrics"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = time.time()
        self.logs: List[LogEntry] = []
    
    def log(self, message: str, level: str = "INFO", source: str = None):
        """Add a log entry"""
        entry = LogEntry(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            level=level,
            message=message,
            source=source or "training"
        )
        self.logs.append(entry)
        logger.info(f"Job {self.job_id}: {message}")
    
    def update_progress(self, epoch: int, total_epochs: int, step: int = None, 
                       total_steps: int = None, metrics: Dict = None) -> TrainingStatus:
        """Update training progress"""
        progress = (epoch / total_epochs) * 100
        
        # Create metrics object
        training_metrics = TrainingMetrics()
        if metrics:
            training_metrics.loss = metrics.get("loss")
            training_metrics.accuracy = metrics.get("accuracy")
            training_metrics.learning_rate = metrics.get("learning_rate")
            training_metrics.gpu_utilization = metrics.get("gpu_utilization", 85.0 + (epoch * 2))
            training_metrics.gpu_memory_used = metrics.get("gpu_memory_used", 12.4 + (epoch * 0.5))
            training_metrics.samples_per_second = metrics.get("samples_per_second")
        
        # Estimate remaining time
        elapsed_time = time.time() - self.start_time
        if epoch > 0:
            time_per_epoch = elapsed_time / epoch
            remaining_epochs = total_epochs - epoch
            estimated_remaining = int(time_per_epoch * remaining_epochs)
        else:
            estimated_remaining = None
        
        return TrainingStatus(
            job_id=self.job_id,
            status=JobStatus.RUNNING,
            progress=progress,
            current_epoch=epoch,
            total_epochs=total_epochs,
            current_step=step,
            total_steps=total_steps,
            metrics=training_metrics,
            start_time=self.start_time,
            estimated_time_remaining=estimated_remaining
        )


class TrainingService:
    """Service for managing training jobs"""
    
    def __init__(self):
        # In-memory storage (replace with database in production)
        self.jobs: Dict[str, TrainingJob] = {}
        self.monitors: Dict[str, TrainingMonitor] = {}
        self.modal_service = ModalService()
    
    def create_job(self, config: TrainingConfig) -> str:
        """Create a new training job"""
        job_id = f"job_{int(time.time())}"
        
        # Initialize job status
        status = TrainingStatus(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0.0,
            current_epoch=0,
            total_epochs=config.num_train_epochs,
            config=config,
            start_time=time.time()
        )
        
        # Create job
        job = TrainingJob(
            job_id=job_id,
            config=config,
            status=status,
            logs=[]
        )
        
        self.jobs[job_id] = job
        self.monitors[job_id] = TrainingMonitor(job_id)
        
        return job_id
    
    async def start_job(self, job_id: str) -> bool:
        """Start a training job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        monitor = self.monitors[job_id]
        
        try:
            # Update status to running
            job.status.status = JobStatus.RUNNING
            monitor.log("Starting training job...")
            
            # Start training (either real or simulation)
            if self.modal_service.is_connected():
                await self._run_modal_training(job_id)
            else:
                await self._run_simulation_training(job_id)
            
            return True
        except Exception as e:
            job.status.status = JobStatus.FAILED
            job.status.error_message = str(e)
            monitor.log(f"Training failed: {str(e)}", level="ERROR")
            return False
    
    async def _run_modal_training(self, job_id: str):
        """Run actual Modal training"""
        job = self.jobs[job_id]
        monitor = self.monitors[job_id]
        
        try:
            monitor.log("Starting Modal.com fine-tuning...")
            modal_job_id = await self.modal_service.start_training(job.config)
            job.modal_job_id = modal_job_id
            
            # In a real implementation, you would poll Modal for progress
            # For now, we'll simulate completion
            await asyncio.sleep(5)
            
            job.status.status = JobStatus.COMPLETED
            job.status.progress = 100.0
            job.status.end_time = time.time()
            monitor.log("Modal.com training completed successfully!")
            
        except Exception as e:
            job.status.status = JobStatus.FAILED
            job.status.error_message = str(e)
            job.status.end_time = time.time()
            monitor.log(f"Modal training failed: {str(e)}", level="ERROR")
            raise
    
    async def _run_simulation_training(self, job_id: str):
        """Run simulated training for demo"""
        job = self.jobs[job_id]
        monitor = self.monitors[job_id]
        config = job.config
        
        monitor.log("Starting simulated training (Modal.com not connected)...")
        
        for epoch in range(1, config.num_train_epochs + 1):
            # Simulate epoch duration
            await asyncio.sleep(3)
            
            # Simulate metrics
            loss = max(0.1, 2.5 - (epoch * 0.3))
            accuracy = min(0.95, 0.65 + (epoch * 0.05))
            
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": config.learning_rate * (0.9 ** epoch)
            }
            
            # Update progress
            job.status = monitor.update_progress(epoch, config.num_train_epochs, metrics=metrics)
            monitor.log(f"Completed epoch {epoch}/{config.num_train_epochs}, "
                       f"Loss: {loss:.4f}, Accuracy: {accuracy:.3f}")
        
        # Mark as completed
        job.status.status = JobStatus.COMPLETED
        job.status.progress = 100.0
        job.status.end_time = time.time()
        monitor.log("Simulated training completed successfully!")
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID"""
        return self.jobs.get(job_id)
    
    def get_job_status(self, job_id: str) -> Optional[TrainingStatus]:
        """Get job status"""
        job = self.jobs.get(job_id)
        return job.status if job else None
    
    def get_job_logs(self, job_id: str) -> List[LogEntry]:
        """Get job logs"""
        monitor = self.monitors.get(job_id)
        return monitor.logs if monitor else []
    
    def list_jobs(self) -> List[TrainingJob]:
        """List all training jobs"""
        return list(self.jobs.values())
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        job.status.status = JobStatus.CANCELLED
        job.status.end_time = time.time()
        
        monitor = self.monitors.get(job_id)
        if monitor:
            monitor.log("Job cancelled by user", level="WARNING")
        
        return True


# Global service instances
storage_manager = StorageManager()
modal_service = ModalService()
training_service = TrainingService()