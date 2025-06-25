"""
Configuration validation service for training parameters
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError
from models.training_models import TrainingConfig, ValidationError, GPUType


class ConfigValidator:
    """
    Validates training configurations against hardware constraints,
    best practices, and resource availability.
    """
    
    # GPU memory constraints (in GB)
    GPU_MEMORY_LIMITS = {
        GPUType.T4: 16,
        GPUType.A10G: 24,
        GPUType.A100: 40,
        GPUType.H100: 80,
        GPUType.V100: 32
    }
    
    # Cost per hour estimates (USD)
    GPU_COST_PER_HOUR = {
        GPUType.T4: 0.35,
        GPUType.A10G: 0.75,
        GPUType.A100: 1.20,
        GPUType.H100: 3.00,
        GPUType.V100: 0.90
    }
    
    # Recommended batch sizes for different GPU types
    RECOMMENDED_BATCH_SIZES = {
        GPUType.T4: {"min": 1, "max": 8, "optimal": 4},
        GPUType.A10G: {"min": 2, "max": 16, "optimal": 8},
        GPUType.A100: {"min": 4, "max": 32, "optimal": 16},
        GPUType.H100: {"min": 8, "max": 64, "optimal": 32},
        GPUType.V100: {"min": 2, "max": 16, "optimal": 8}
    }
    
    def __init__(self):
        self.validation_errors: List[ValidationError] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: TrainingConfig) -> Tuple[bool, List[ValidationError], List[str]]:
        """
        Comprehensive configuration validation
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.warnings = []
        
        # Validate basic configuration
        self._validate_model_config(config)
        self._validate_dataset_config(config)
        self._validate_output_config(config)
        
        # Validate training parameters
        self._validate_training_params(config)
        self._validate_lora_config(config)
        self._validate_quantization_config(config)
        
        # Validate hardware configuration
        self._validate_hardware_config(config)
        
        # Validate resource constraints
        self._validate_resource_constraints(config)
        
        # Performance recommendations
        self._generate_performance_warnings(config)
        
        return len(self.validation_errors) == 0, self.validation_errors, self.warnings
    
    def _validate_model_config(self, config: TrainingConfig):
        """Validate model configuration"""
        # Check model name format
        if not self._is_valid_model_identifier(config.model_name):
            self._add_error(
                "model_name",
                "Invalid model identifier format",
                config.model_name,
                "format_error"
            )
        
        # Warn about trust_remote_code
        if config.trust_remote_code:
            self.warnings.append(
                "trust_remote_code is enabled. Only use with trusted models."
            )
    
    def _validate_dataset_config(self, config: TrainingConfig):
        """Validate dataset configuration"""
        # Check dataset path format
        if not config.dataset_path:
            self._add_error(
                "dataset_path",
                "Dataset path cannot be empty",
                config.dataset_path,
                "required_field"
            )
        
        # Validate dataset split
        valid_splits = ["train", "validation", "test", "dev"]
        if config.dataset_split not in valid_splits:
            self.warnings.append(
                f"Unusual dataset split '{config.dataset_split}'. "
                f"Common splits are: {', '.join(valid_splits)}"
            )
        
        # Check max_samples
        if config.max_samples is not None and config.max_samples < 10:
            self.warnings.append(
                f"Very small dataset size ({config.max_samples} samples). "
                "Consider using more data for better results."
            )
    
    def _validate_output_config(self, config: TrainingConfig):
        """Validate output configuration"""
        # Check output directory format
        if not config.output_dir.startswith(("/vol/", "/tmp/")):
            self.warnings.append(
                "Output directory should typically start with /vol/ for persistence"
            )
        
        # Validate save strategy
        valid_strategies = ["no", "steps", "epoch"]
        if config.save_strategy not in valid_strategies:
            self._add_error(
                "save_strategy",
                f"Invalid save strategy. Must be one of: {valid_strategies}",
                config.save_strategy,
                "invalid_choice"
            )
    
    def _validate_training_params(self, config: TrainingConfig):
        """Validate training hyperparameters"""
        # Learning rate validation
        if config.learning_rate > 0.01:
            self.warnings.append(
                f"High learning rate ({config.learning_rate}). "
                "Consider using a lower value (1e-4 to 1e-3) for fine-tuning."
            )
        
        if config.learning_rate < 1e-6:
            self.warnings.append(
                f"Very low learning rate ({config.learning_rate}). "
                "Training might be very slow or ineffective."
            )
        
        # Epoch validation
        if config.num_train_epochs > 20:
            self.warnings.append(
                f"High number of epochs ({config.num_train_epochs}). "
                "Monitor for overfitting."
            )
        
        # Batch size validation
        total_batch_size = (
            config.per_device_train_batch_size * 
            config.gradient_accumulation_steps * 
            config.num_gpus
        )
        
        if total_batch_size > 128:
            self.warnings.append(
                f"Large effective batch size ({total_batch_size}). "
                "This might lead to poor convergence."
            )
        
        # Warmup validation
        if config.warmup_steps > 0 and config.warmup_ratio > 0:
            self._add_error(
                "warmup_steps",
                "Cannot specify both warmup_steps and warmup_ratio",
                f"steps={config.warmup_steps}, ratio={config.warmup_ratio}",
                "conflicting_params"
            )
    
    def _validate_lora_config(self, config: TrainingConfig):
        """Validate LoRA configuration"""
        if not config.use_lora:
            return
        
        # LoRA rank validation
        if config.lora_r > 128:
            self.warnings.append(
                f"High LoRA rank ({config.lora_r}). "
                "This might reduce the benefits of parameter-efficient training."
            )
        
        # LoRA alpha validation
        if config.lora_alpha < config.lora_r:
            self.warnings.append(
                f"LoRA alpha ({config.lora_alpha}) is less than rank ({config.lora_r}). "
                "Consider using alpha >= rank for better performance."
            )
        
        # LoRA dropout validation
        if config.lora_dropout > 0.3:
            self.warnings.append(
                f"High LoRA dropout ({config.lora_dropout}). "
                "This might hurt training performance."
            )
    
    def _validate_quantization_config(self, config: TrainingConfig):
        """Validate quantization configuration"""
        if config.use_4bit_quantization and config.use_8bit_quantization:
            self._add_error(
                "quantization",
                "Cannot use both 4-bit and 8-bit quantization",
                f"4bit={config.use_4bit_quantization}, 8bit={config.use_8bit_quantization}",
                "conflicting_params"
            )
        
        # Check quantization compatibility
        if (config.use_4bit_quantization or config.use_8bit_quantization):
            if not config.use_lora:
                self.warnings.append(
                    "Quantization without LoRA might lead to degraded performance. "
                    "Consider enabling LoRA."
                )
    
    def _validate_hardware_config(self, config: TrainingConfig):
        """Validate hardware configuration"""
        # GPU type validation
        if config.gpu_type not in self.GPU_MEMORY_LIMITS:
            self._add_error(
                "gpu_type",
                f"Unsupported GPU type: {config.gpu_type}",
                config.gpu_type,
                "invalid_choice"
            )
            return
        
        # Batch size recommendations
        gpu_recommendations = self.RECOMMENDED_BATCH_SIZES.get(config.gpu_type, {})
        if config.per_device_train_batch_size > gpu_recommendations.get("max", 32):
            self.warnings.append(
                f"Batch size ({config.per_device_train_batch_size}) might be too large "
                f"for {config.gpu_type}. Consider reducing to avoid OOM errors."
            )
        
        # GPU count validation
        if config.num_gpus > 8:
            self.warnings.append(
                f"Very high GPU count ({config.num_gpus}). "
                "Ensure your setup supports distributed training."
            )
        
        # Mixed precision validation
        valid_precision = ["no", "fp16", "bf16"]
        if config.mixed_precision not in valid_precision:
            self._add_error(
                "mixed_precision",
                f"Invalid mixed precision option. Must be one of: {valid_precision}",
                config.mixed_precision,
                "invalid_choice"
            )
    
    def _validate_resource_constraints(self, config: TrainingConfig):
        """Validate resource constraints and limits"""
        # Memory estimation
        estimated_memory = self._estimate_memory_usage(config)
        gpu_memory_limit = self.GPU_MEMORY_LIMITS.get(config.gpu_type, 40)
        
        if estimated_memory > gpu_memory_limit * 0.9:  # 90% threshold
            self.warnings.append(
                f"Estimated memory usage ({estimated_memory:.1f}GB) approaches "
                f"{config.gpu_type} limit ({gpu_memory_limit}GB). "
                "Consider reducing batch size or enabling quantization."
            )
        
        # Timeout validation
        if config.timeout < 300:  # 5 minutes
            self.warnings.append(
                f"Very short timeout ({config.timeout}s). "
                "Training might not complete."
            )
        
        if config.timeout > 24 * 3600:  # 24 hours
            self.warnings.append(
                f"Very long timeout ({config.timeout / 3600:.1f}h). "
                "Consider shorter training runs for faster iteration."
            )
    
    def _generate_performance_warnings(self, config: TrainingConfig):
        """Generate performance optimization warnings"""
        # Gradient checkpointing recommendation
        if not config.gradient_checkpointing:
            self.warnings.append(
                "Consider enabling gradient_checkpointing to save memory."
            )
        
        # Dataloader workers
        if config.dataloader_num_workers == 0:
            self.warnings.append(
                "Consider using multiple dataloader workers for better I/O performance."
            )
        
        # Evaluation strategy
        if config.evaluation_strategy == "no":
            self.warnings.append(
                "No evaluation strategy specified. "
                "Consider periodic evaluation to monitor training progress."
            )
    
    def _estimate_memory_usage(self, config: TrainingConfig) -> float:
        """
        Rough estimate of GPU memory usage in GB
        This is a simplified estimation - actual usage depends on model architecture
        """
        base_memory = 2.0  # Base overhead
        
        # Model size estimation (very rough)
        if "7b" in config.model_name.lower():
            model_memory = 14.0
        elif "13b" in config.model_name.lower():
            model_memory = 26.0
        elif "1b" in config.model_name.lower():
            model_memory = 2.0
        else:
            model_memory = 6.0  # Default assumption
        
        # Quantization reduction
        if config.use_4bit_quantization:
            model_memory *= 0.25
        elif config.use_8bit_quantization:
            model_memory *= 0.5
        
        # Batch size impact
        batch_memory = config.per_device_train_batch_size * 0.5
        
        # Gradient and optimizer states
        if config.use_lora:
            optimizer_memory = model_memory * 0.1  # LoRA reduces optimizer memory
        else:
            optimizer_memory = model_memory * 2.0  # Full fine-tuning
        
        total_memory = base_memory + model_memory + batch_memory + optimizer_memory
        
        return total_memory
    
    def _is_valid_model_identifier(self, model_name: str) -> bool:
        """Check if model identifier is valid"""
        # HuggingFace format: organization/model-name
        hf_pattern = r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$'
        
        # Local path format
        path_pattern = r'^(/[^/\s]+)+/?$'
        
        return (
            bool(re.match(hf_pattern, model_name)) or
            bool(re.match(path_pattern, model_name)) or
            model_name.startswith("./") or
            model_name.startswith("../")
        )
    
    def _add_error(self, field: str, message: str, value: Any, error_type: str):
        """Add a validation error"""
        self.validation_errors.append(
            ValidationError(
                field=field,
                message=message,
                value=value,
                error_type=error_type
            )
        )
    
    def get_gpu_recommendations(self, config: TrainingConfig) -> Dict[str, Any]:
        """Get GPU recommendations based on configuration"""
        recommendations = {}
        
        # Estimate required memory
        estimated_memory = self._estimate_memory_usage(config)
        
        # Find suitable GPU types
        suitable_gpus = []
        for gpu_type, memory_limit in self.GPU_MEMORY_LIMITS.items():
            if memory_limit >= estimated_memory * 1.2:  # 20% buffer
                cost_per_hour = self.GPU_COST_PER_HOUR.get(gpu_type, 1.0)
                suitable_gpus.append({
                    "gpu_type": gpu_type,
                    "memory_gb": memory_limit,
                    "cost_per_hour": cost_per_hour,
                    "memory_utilization": (estimated_memory / memory_limit) * 100
                })
        
        # Sort by cost
        suitable_gpus.sort(key=lambda x: x["cost_per_hour"])
        
        recommendations["suitable_gpus"] = suitable_gpus
        recommendations["estimated_memory_gb"] = estimated_memory
        recommendations["recommended_batch_size"] = self.RECOMMENDED_BATCH_SIZES.get(
            config.gpu_type, {"optimal": 8}
        )["optimal"]
        
        return recommendations