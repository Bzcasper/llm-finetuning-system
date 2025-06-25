import modal
import os
from typing import Optional
from minio import Minio
from minio.error import S3Error
import boto3
from botocore.exceptions import ClientError
import tempfile
import shutil

# Define a Modal App
app = modal.App(name="llm-finetuner")

# Define a shared volume for models and datasets
volume = modal.Volume.from_name("llm-finetuning-volume", create_if_missing=True)

# Image for fine-tuning, including all necessary libraries
finetune_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "datasets>=2.14.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "loralib>=0.1.2",
        "scipy>=1.11.0",
        "einops>=0.7.0",
        "wandb>=0.15.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tensorboard>=2.14.0",
        "evaluate>=0.4.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "safetensors>=0.3.0",
        "tokenizers>=0.14.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "nvidia-ml-py3>=7.352.0",
        "minio>=7.2.0",
        "boto3>=1.34.0",
        "s3fs>=2023.12.0"
    ])
    .apt_install(["git", "wget", "curl", "htop"])
    .run_commands([
        "pip install flash-attn --no-build-isolation || echo 'Flash attention install failed, continuing...'",
    ])
)

class StorageManager:
    """Unified storage manager for Modal volumes and MinIO"""
    
    def __init__(self):
        self.minio_client = None
        self.s3_client = None
        self.setup_storage_clients()
    
    def setup_storage_clients(self):
        """Initialize MinIO and S3 clients"""
        try:
            # MinIO configuration
            minio_endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
            minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
            minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
            minio_secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
            
            self.minio_client = Minio(
                minio_endpoint,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=minio_secure
            )
            
            # S3 configuration (for AWS S3 compatibility)
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            
            if aws_access_key and aws_secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
            
            print("✅ Storage clients initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize storage clients: {e}")
    
    def ensure_bucket_exists(self, bucket_name: str):
        """Ensure MinIO bucket exists"""
        try:
            if self.minio_client and not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
                print(f"✅ Created MinIO bucket: {bucket_name}")
        except S3Error as e:
            print(f"⚠️ MinIO bucket error: {e}")
    
    def upload_to_minio(self, local_path: str, bucket_name: str, object_name: str):
        """Upload file to MinIO"""
        try:
            if not self.minio_client:
                return False
                
            self.ensure_bucket_exists(bucket_name)
            self.minio_client.fput_object(bucket_name, object_name, local_path)
            print(f"✅ Uploaded {local_path} to MinIO: {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            print(f"❌ MinIO upload error: {e}")
            return False
    
    def download_from_minio(self, bucket_name: str, object_name: str, local_path: str):
        """Download file from MinIO"""
        try:
            if not self.minio_client:
                return False
                
            self.minio_client.fget_object(bucket_name, object_name, local_path)
            print(f"✅ Downloaded from MinIO: {bucket_name}/{object_name} to {local_path}")
            return True
        except S3Error as e:
            print(f"❌ MinIO download error: {e}")
            return False
    
    def upload_to_s3(self, local_path: str, bucket_name: str, object_name: str):
        """Upload file to AWS S3"""
        try:
            if not self.s3_client:
                return False
                
            self.s3_client.upload_file(local_path, bucket_name, object_name)
            print(f"✅ Uploaded {local_path} to S3: {bucket_name}/{object_name}")
            return True
        except ClientError as e:
            print(f"❌ S3 upload error: {e}")
            return False
    
    def download_from_s3(self, bucket_name: str, object_name: str, local_path: str):
        """Download file from AWS S3"""
        try:
            if not self.s3_client:
                return False
                
            self.s3_client.download_file(bucket_name, object_name, local_path)
            print(f"✅ Downloaded from S3: {bucket_name}/{object_name} to {local_path}")
            return True
        except ClientError as e:
            print(f"❌ S3 download error: {e}")
            return False
    
    def list_minio_objects(self, bucket_name: str, prefix: str = ""):
        """List objects in MinIO bucket"""
        try:
            if not self.minio_client:
                return []
                
            objects = self.minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"❌ MinIO list error: {e}")
            return []

@app.function(
    image=finetune_image,
    gpu="A100", # Configurable GPU type
    volumes={"/vol": volume},
    timeout=3600, # Configurable timeout
    secrets=[modal.Secret.from_name("huggingface-secret")] # For Hugging Face access
)
def fine_tune_llm(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str = "/vol/finetuned_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 0.0002,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    optimizer: str = "adamw_torch",
    gpu_type: str = "A100",
    timeout: int = 3600,
    storage_backend: str = "volume",  # "volume", "minio", or "s3"
    minio_bucket: str = "llm-models",
    s3_bucket: str = "llm-models"
):
    """
    Fine-tune an LLM using LoRA/QLoRA with multiple storage backends
    
    Args:
        model_name_or_path: HuggingFace model name or path to model
        dataset_path: Path to training dataset
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate for training
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        use_4bit: Whether to use 4-bit quantization (QLoRA)
        optimizer: Optimizer to use
        gpu_type: GPU type for training
        timeout: Training timeout in seconds
        storage_backend: Storage backend ("volume", "minio", or "s3")
        minio_bucket: MinIO bucket name
        s3_bucket: S3 bucket name
    """
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    import json
    import time
    import GPUtil
    import psutil
    
    print(f"🚀 Starting fine-tuning job")
    print(f"📊 Configuration:")
    print(f"   Model: {model_name_or_path}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Output: {output_dir}")
    print(f"   Epochs: {num_train_epochs}")
    print(f"   Batch Size: {per_device_train_batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"   4-bit: {use_4bit}")
    print(f"   Optimizer: {optimizer}")
    print(f"   Storage: {storage_backend}")
    
    # Initialize storage manager
    storage = StorageManager()
    
    # Setup quantization config for QLoRA
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("✅ 4-bit quantization enabled")
    else:
        bnb_config = None
        print("✅ Full precision training")
    
    # Load model and tokenizer
    print(f"📥 Loading model: {model_name_or_path}")
    
    # Check if model is in storage backends
    model_loaded_from_storage = False
    local_model_path = f"/tmp/model_{int(time.time())}"
    
    if storage_backend == "minio":
        # Try to load model from MinIO
        model_objects = storage.list_minio_objects(minio_bucket, f"models/{model_name_or_path}/")
        if model_objects:
            print(f"📥 Loading model from MinIO: {minio_bucket}")
            os.makedirs(local_model_path, exist_ok=True)
            for obj in model_objects:
                local_file = os.path.join(local_model_path, os.path.basename(obj))
                if storage.download_from_minio(minio_bucket, obj, local_file):
                    model_loaded_from_storage = True
    
    elif storage_backend == "s3":
        # Try to load model from S3
        try:
            response = storage.s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=f"models/{model_name_or_path}/")
            if 'Contents' in response:
                print(f"📥 Loading model from S3: {s3_bucket}")
                os.makedirs(local_model_path, exist_ok=True)
                for obj in response['Contents']:
                    local_file = os.path.join(local_model_path, os.path.basename(obj['Key']))
                    if storage.download_from_s3(s3_bucket, obj['Key'], local_file):
                        model_loaded_from_storage = True
        except Exception as e:
            print(f"⚠️ Could not check S3 for model: {e}")
    
    # Load model from HuggingFace or local storage
    if model_loaded_from_storage:
        model_path = local_model_path
        print(f"✅ Using model from storage: {model_path}")
    else:
        model_path = model_name_or_path
        print(f"✅ Using HuggingFace model: {model_path}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model and tokenizer loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Setup LoRA configuration with architecture-specific target modules
    def get_target_modules(model_name_or_path):
        """Get appropriate target modules based on model architecture"""
        model_name_lower = model_name_or_path.lower()
        
        if any(arch in model_name_lower for arch in ["llama", "alpaca", "vicuna"]):
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif any(arch in model_name_lower for arch in ["mistral", "mixtral"]):
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif any(arch in model_name_lower for arch in ["codellama", "code-llama"]):
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif any(arch in model_name_lower for arch in ["falcon"]):
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif any(arch in model_name_lower for arch in ["gpt", "bloom"]):
            return ["c_attn", "c_proj", "c_fc"]
        elif any(arch in model_name_lower for arch in ["t5"]):
            return ["q", "v", "k", "o", "wi", "wo"]
        else:
            # Default for most transformer models
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    target_modules = get_target_modules(model_name_or_path)
    print(f"🎯 Using target modules for LoRA: {target_modules}")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        fan_in_fan_out=False
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print(f"📥 Loading dataset: {dataset_path}")
    
    # Check if dataset is in storage
    dataset_loaded_from_storage = False
    local_dataset_path = f"/tmp/dataset_{int(time.time())}.jsonl"
    
    if storage_backend == "minio":
        if storage.download_from_minio(minio_bucket, f"datasets/{dataset_path}", local_dataset_path):
            dataset_path = local_dataset_path
            dataset_loaded_from_storage = True
    elif storage_backend == "s3":
        if storage.download_from_s3(s3_bucket, f"datasets/{dataset_path}", local_dataset_path):
            dataset_path = local_dataset_path
            dataset_loaded_from_storage = True
    
    if not dataset_loaded_from_storage and not os.path.exists(dataset_path):
        # Try to load from Modal volume
        volume_dataset_path = f"/vol/{dataset_path}"
        if os.path.exists(volume_dataset_path):
            dataset_path = volume_dataset_path
            print(f"✅ Using dataset from Modal volume: {dataset_path}")
        else:
            print(f"❌ Dataset not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    try:
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        elif dataset_path.endswith('.json'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            # Try to load as HuggingFace dataset
            dataset = load_dataset(dataset_path, split='train')
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        optim=optimizer,
        save_steps=500,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        save_strategy="steps",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Monitor system resources
    def log_system_stats():
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"📊 GPU: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% utilization")
        
        memory = psutil.virtual_memory()
        print(f"📊 RAM: {memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    
    # Start training
    print("🏋️ Starting training...")
    log_system_stats()
    
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    # Save the model
    print(f"💾 Saving model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Upload to storage backends
    if storage_backend == "minio":
        print(f"📤 Uploading model to MinIO: {minio_bucket}")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, output_dir)
                object_name = f"models/finetuned/{model_name_or_path.replace('/', '_')}/{relative_path}"
                storage.upload_to_minio(local_file, minio_bucket, object_name)
    
    elif storage_backend == "s3":
        print(f"📤 Uploading model to S3: {s3_bucket}")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, output_dir)
                object_name = f"models/finetuned/{model_name_or_path.replace('/', '_')}/{relative_path}"
                storage.upload_to_s3(local_file, s3_bucket, object_name)
    
    # Final system stats
    log_system_stats()
    
    # Cleanup temporary files
    if model_loaded_from_storage and os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
    if dataset_loaded_from_storage and os.path.exists(local_dataset_path):
        os.remove(local_dataset_path)
    
    result = {
        "status": "completed",
        "model_path": output_dir,
        "training_time": training_time,
        "storage_backend": storage_backend,
        "final_loss": trainer.state.log_history[-1].get("train_loss", 0) if trainer.state.log_history else 0,
        "total_steps": trainer.state.global_step,
        "message": "Fine-tuning completed successfully"
    }
    
    print(f"🎉 Fine-tuning job completed successfully!")
    print(f"📊 Final results: {result}")
    
    return result

@app.function(
    image=finetune_image,
    volumes={"/vol": volume},
)
def list_models_and_datasets():
    """List available models and datasets from all storage backends"""
    storage = StorageManager()
    
    models = []
    datasets = []
    
    # List from Modal volume
    volume_models_dir = "/vol/models"
    volume_datasets_dir = "/vol/datasets"
    
    if os.path.exists(volume_models_dir):
        for item in os.listdir(volume_models_dir):
            models.append({
                "name": item,
                "path": f"/vol/models/{item}",
                "source": "modal_volume",
                "size": "unknown"
            })
    
    if os.path.exists(volume_datasets_dir):
        for item in os.listdir(volume_datasets_dir):
            datasets.append({
                "name": item,
                "path": f"/vol/datasets/{item}",
                "source": "modal_volume",
                "size": "unknown"
            })
    
    # List from MinIO
    try:
        minio_models = storage.list_minio_objects("llm-models", "models/")
        for model in minio_models:
            models.append({
                "name": os.path.basename(model),
                "path": model,
                "source": "minio",
                "size": "unknown"
            })
        
        minio_datasets = storage.list_minio_objects("llm-models", "datasets/")
        for dataset in minio_datasets:
            datasets.append({
                "name": os.path.basename(dataset),
                "path": dataset,
                "source": "minio",
                "size": "unknown"
            })
    except Exception as e:
        print(f"⚠️ Could not list MinIO objects: {e}")
    
    # List from S3
    try:
        if storage.s3_client:
            response = storage.s3_client.list_objects_v2(Bucket="llm-models", Prefix="models/")
            if 'Contents' in response:
                for obj in response['Contents']:
                    models.append({
                        "name": os.path.basename(obj['Key']),
                        "path": obj['Key'],
                        "source": "s3",
                        "size": obj['Size']
                    })
            
            response = storage.s3_client.list_objects_v2(Bucket="llm-models", Prefix="datasets/")
            if 'Contents' in response:
                for obj in response['Contents']:
                    datasets.append({
                        "name": os.path.basename(obj['Key']),
                        "path": obj['Key'],
                        "source": "s3",
                        "size": obj['Size']
                    })
    except Exception as e:
        print(f"⚠️ Could not list S3 objects: {e}")
    
    return {
        "models": models,
        "datasets": datasets,
        "storage_backends": ["modal_volume", "minio", "s3"]
    }

@app.function(
    image=finetune_image,
    gpu="T4",
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def model_inference(
    model_name_or_path: str,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    storage_backend: str = "volume",
    use_4bit: bool = True
):
    """
    Run inference on a fine-tuned model
    
    Args:
        model_name_or_path: Path to the fine-tuned model
        prompt: Input text for generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        num_return_sequences: Number of sequences to generate
        storage_backend: Storage backend to load model from
        use_4bit: Whether to use 4-bit quantization for inference
    """
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        BitsAndBytesConfig,
        GenerationConfig
    )
    from peft import PeftModel
    import time
    
    print(f"🔮 Starting inference with model: {model_name_or_path}")
    print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Prompt: {prompt}")
    
    storage = StorageManager()
    
    # Setup quantization for inference
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model from appropriate storage
    model_path = model_name_or_path
    if storage_backend == "volume":
        volume_path = f"/vol/{model_name_or_path}"
        if os.path.exists(volume_path):
            model_path = volume_path
            print(f"✅ Using model from Modal volume: {model_path}")
    
    try:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model and tokenizer loaded for inference")
        
        # Setup generation config
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate text
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        inference_time = time.time() - start_time
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the input prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            generated_texts.append(generated_text.strip())
        
        result = {
            "status": "success",
            "prompt": prompt,
            "generated_texts": generated_texts,
            "inference_time": inference_time,
            "model_path": model_path,
            "config": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences
            }
        }
        
        print(f"✅ Inference completed in {inference_time:.2f} seconds")
        return result
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "prompt": prompt
        }

@app.function(
    image=finetune_image,
    gpu="any",
    volumes={"/vol": volume},
)
def gpu_monitoring():
    """
    Monitor GPU resources and system status
    """
    import GPUtil
    import psutil
    import torch
    import json
    import time
    
    try:
        # GPU information
        gpu_info = []
        gpus = GPUtil.getGPUs()
        
        for i, gpu in enumerate(gpus):
            gpu_info.append({
                "id": i,
                "name": gpu.name,
                "memory_total": gpu.memoryTotal,
                "memory_used": gpu.memoryUsed,
                "memory_free": gpu.memoryFree,
                "memory_util": gpu.memoryUtil * 100,
                "load": gpu.load * 100,
                "temperature": gpu.temperature,
            })
        
        # PyTorch CUDA info
        cuda_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
        
        if torch.cuda.is_available():
            cuda_info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            cuda_info["memory_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
            cuda_info["max_memory_allocated"] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # System memory
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total / 1024**3,  # GB
            "used": memory.used / 1024**3,   # GB
            "available": memory.available / 1024**3,  # GB
            "percent": memory.percent
        }
        
        # CPU information
        cpu_info = {
            "count": psutil.cpu_count(),
            "percent": psutil.cpu_percent(interval=1),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_info = {
            "total": disk.total / 1024**3,  # GB
            "used": disk.used / 1024**3,   # GB
            "free": disk.free / 1024**3,   # GB
            "percent": (disk.used / disk.total) * 100
        }
        
        result = {
            "timestamp": time.time(),
            "status": "success",
            "gpu_info": gpu_info,
            "cuda_info": cuda_info,
            "memory_info": memory_info,
            "cpu_info": cpu_info,
            "disk_info": disk_info,
            "system": {
                "platform": os.uname().sysname if hasattr(os, 'uname') else 'Unknown',
                "hostname": os.uname().nodename if hasattr(os, 'uname') else 'Unknown'
            }
        }
        
        print(f"📊 System monitoring completed")
        return result
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return {
            "timestamp": time.time(),
            "status": "error",
            "error": str(e)
        }

@app.function(
    image=finetune_image,
    volumes={"/vol": volume},
)
def data_preprocessing(
    dataset_path: str,
    output_path: str,
    dataset_format: str = "jsonl",
    text_column: str = "text",
    max_length: int = 512,
    min_length: int = 10,
    remove_duplicates: bool = True,
    shuffle: bool = True,
    train_split: float = 0.8,
    storage_backend: str = "volume",
    preprocessing_steps: list = None
):
    """
    Preprocess datasets for fine-tuning
    
    Args:
        dataset_path: Path to raw dataset
        output_path: Path to save processed dataset
        dataset_format: Format of the dataset ("jsonl", "json", "csv", "txt")
        text_column: Column name containing text data
        max_length: Maximum text length
        min_length: Minimum text length
        remove_duplicates: Whether to remove duplicate entries
        shuffle: Whether to shuffle the dataset
        train_split: Proportion of data for training (rest for validation)
        storage_backend: Storage backend to use
        preprocessing_steps: List of preprocessing steps to apply
    """
    import pandas as pd
    import json
    import re
    from datasets import Dataset
    import random
    from collections import Counter
    
    print(f"🔧 Starting data preprocessing")
    print(f"📁 Input: {dataset_path}")
    print(f"📁 Output: {output_path}")
    print(f"📊 Format: {dataset_format}")
    
    storage = StorageManager()
    
    # Default preprocessing steps
    if preprocessing_steps is None:
        preprocessing_steps = [
            "remove_html",
            "normalize_whitespace",
            "remove_empty",
            "length_filter"
        ]
    
    try:
        # Load dataset based on format
        if dataset_format == "jsonl":
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            df = pd.DataFrame(data)
            
        elif dataset_format == "json":
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
                
        elif dataset_format == "csv":
            df = pd.read_csv(dataset_path)
            
        elif dataset_format == "txt":
            with open(dataset_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            df = pd.DataFrame({text_column: [line.strip() for line in lines]})
            
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
        
        print(f"✅ Loaded {len(df)} samples")
        
        # Ensure text column exists
        if text_column not in df.columns:
            print(f"❌ Text column '{text_column}' not found. Available columns: {list(df.columns)}")
            raise ValueError(f"Text column '{text_column}' not found")
        
        original_count = len(df)
        
        # Apply preprocessing steps
        for step in preprocessing_steps:
            if step == "remove_html":
                print("🧹 Removing HTML tags")
                df[text_column] = df[text_column].astype(str).apply(
                    lambda x: re.sub(r'<[^>]+>', '', x)
                )
                
            elif step == "normalize_whitespace":
                print("🧹 Normalizing whitespace")
                df[text_column] = df[text_column].astype(str).apply(
                    lambda x: re.sub(r'\s+', ' ', x).strip()
                )
                
            elif step == "remove_empty":
                print("🧹 Removing empty entries")
                df = df[df[text_column].astype(str).str.strip() != ""]
                df = df[df[text_column].notna()]
                
            elif step == "length_filter":
                print(f"🧹 Filtering by length ({min_length}-{max_length} chars)")
                df = df[
                    (df[text_column].str.len() >= min_length) & 
                    (df[text_column].str.len() <= max_length)
                ]
                
            elif step == "remove_duplicates" and remove_duplicates:
                print("🧹 Removing duplicates")
                df = df.drop_duplicates(subset=[text_column])
                
            elif step == "lowercase":
                print("🧹 Converting to lowercase")
                df[text_column] = df[text_column].str.lower()
                
            elif step == "remove_special_chars":
                print("🧹 Removing special characters")
                df[text_column] = df[text_column].apply(
                    lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x))
                )
        
        print(f"📊 After preprocessing: {len(df)} samples (removed {original_count - len(df)})")
        
        # Shuffle if requested
        if shuffle:
            print("🔀 Shuffling dataset")
            df = df.sample(frac=1).reset_index(drop=True)
        
        # Split into train/validation
        if train_split < 1.0:
            print(f"✂️ Splitting dataset: {train_split:.1%} train, {1-train_split:.1%} validation")
            split_idx = int(len(df) * train_split)
            train_df = df[:split_idx]
            val_df = df[split_idx:]
            
            # Save split datasets
            train_output = output_path.replace('.jsonl', '_train.jsonl')
            val_output = output_path.replace('.jsonl', '_val.jsonl')
            
            # Save train set
            with open(train_output, 'w', encoding='utf-8') as f:
                for _, row in train_df.iterrows():
                    f.write(json.dumps(row.to_dict()) + '\n')
            
            # Save validation set
            with open(val_output, 'w', encoding='utf-8') as f:
                for _, row in val_df.iterrows():
                    f.write(json.dumps(row.to_dict()) + '\n')
            
            print(f"💾 Saved train set: {train_output} ({len(train_df)} samples)")
            print(f"💾 Saved validation set: {val_output} ({len(val_df)} samples)")
            
            result_files = [train_output, val_output]
        else:
            # Save entire dataset
            with open(output_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict()) + '\n')
            
            print(f"💾 Saved processed dataset: {output_path} ({len(df)} samples)")
            result_files = [output_path]
        
        # Generate statistics
        text_lengths = df[text_column].str.len()
        stats = {
            "total_samples": len(df),
            "original_samples": original_count,
            "removed_samples": original_count - len(df),
            "text_length_stats": {
                "mean": float(text_lengths.mean()),
                "median": float(text_lengths.median()),
                "min": int(text_lengths.min()),
                "max": int(text_lengths.max()),
                "std": float(text_lengths.std())
            },
            "preprocessing_steps": preprocessing_steps,
            "train_split": train_split
        }
        
        # Upload to storage backends if needed
        if storage_backend != "volume":
            for file_path in result_files:
                filename = os.path.basename(file_path)
                if storage_backend == "minio":
                    storage.upload_to_minio(file_path, "llm-models", f"datasets/processed/{filename}")
                elif storage_backend == "s3":
                    storage.upload_to_s3(file_path, "llm-models", f"datasets/processed/{filename}")
        
        result = {
            "status": "success",
            "input_path": dataset_path,
            "output_files": result_files,
            "statistics": stats,
            "storage_backend": storage_backend
        }
        
        print(f"✅ Data preprocessing completed")
        return result
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "input_path": dataset_path
        }

@app.function(
    image=finetune_image,
    gpu="T4",
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def validate_model(
    model_name_or_path: str,
    test_dataset_path: str = None,
    storage_backend: str = "volume",
    validation_prompts: list = None,
    metrics: list = None
):
    """
    Validate a fine-tuned model's performance
    
    Args:
        model_name_or_path: Path to the model to validate
        test_dataset_path: Path to test dataset for evaluation
        storage_backend: Storage backend to load model from
        validation_prompts: List of prompts for qualitative evaluation
        metrics: List of metrics to compute ("perplexity", "bleu", "rouge")
    """
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        BitsAndBytesConfig
    )
    from datasets import load_dataset
    import numpy as np
    import json
    import time
    
    print(f"🔍 Starting model validation")
    print(f"📊 Model: {model_name_or_path}")
    
    if metrics is None:
        metrics = ["perplexity"]
    
    if validation_prompts is None:
        validation_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important aspect of machine learning is"
        ]
    
    storage = StorageManager()
    
    # Setup quantization for validation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        # Load model from appropriate storage
        model_path = model_name_or_path
        if storage_backend == "volume":
            volume_path = f"/vol/{model_name_or_path}"
            if os.path.exists(volume_path):
                model_path = volume_path
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded for validation")
        
        results = {
            "model_path": model_path,
            "validation_timestamp": time.time(),
            "qualitative_results": [],
            "quantitative_metrics": {},
            "status": "success"
        }
        
        # Qualitative evaluation with validation prompts
        print("🎭 Running qualitative evaluation")
        for prompt in validation_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            
            results["qualitative_results"].append({
                "prompt": prompt,
                "generated_text": generated_text.strip(),
                "full_response": tokenizer.decode(outputs[0], skip_special_tokens=True)
            })
        
        # Quantitative evaluation
        if "perplexity" in metrics:
            print("📈 Computing perplexity")
            try:
                if test_dataset_path:
                    # Load test dataset
                    if os.path.exists(test_dataset_path):
                        with open(test_dataset_path, 'r') as f:
                            test_data = [json.loads(line) for line in f]
                        test_texts = [item.get("text", "") for item in test_data[:100]]  # Limit for efficiency
                    else:
                        test_texts = validation_prompts
                else:
                    test_texts = validation_prompts
                
                perplexities = []
                for text in test_texts[:20]:  # Limit to avoid timeout
                    if len(text.strip()) > 0:
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model(**inputs, labels=inputs["input_ids"])
                            loss = outputs.loss
                            perplexity = torch.exp(loss).item()
                            perplexities.append(perplexity)
                
                if perplexities:
                    results["quantitative_metrics"]["perplexity"] = {
                        "mean": float(np.mean(perplexities)),
                        "std": float(np.std(perplexities)),
                        "min": float(np.min(perplexities)),
                        "max": float(np.max(perplexities)),
                        "samples_evaluated": len(perplexities)
                    }
                    print(f"📊 Average perplexity: {np.mean(perplexities):.2f}")
                
            except Exception as e:
                print(f"⚠️ Perplexity computation failed: {e}")
                results["quantitative_metrics"]["perplexity"] = {"error": str(e)}
        
        # Model info
        try:
            model_size = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results["model_info"] = {
                "total_parameters": model_size,
                "trainable_parameters": trainable_params,
                "trainable_percentage": (trainable_params / model_size) * 100 if model_size > 0 else 0,
                "model_type": type(model).__name__,
                "device": str(model.device) if hasattr(model, 'device') else "unknown"
            }
            
            print(f"📊 Model has {model_size:,} parameters ({trainable_params:,} trainable)")
            
        except Exception as e:
            print(f"⚠️ Could not get model info: {e}")
            results["model_info"] = {"error": str(e)}
        
        print("✅ Model validation completed")
        return results
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "model_path": model_name_or_path
        }

@app.function(
    image=finetune_image,
    volumes={"/vol": volume},
)
def upload_file(file_bytes: bytes, file_name: str, storage_backend: str = "volume"):
    """Upload a file to the specified storage backend"""
    storage = StorageManager()
    
    # Save file temporarily
    temp_path = f"/tmp/{file_name}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    success = False
    final_path = ""
    
    if storage_backend == "volume":
        # Save to Modal volume
        volume_path = f"/vol/{file_name}"
        shutil.copy2(temp_path, volume_path)
        success = True
        final_path = volume_path
        
    elif storage_backend == "minio":
        # Upload to MinIO
        bucket_name = "llm-models"
        object_name = f"datasets/{file_name}"
        success = storage.upload_to_minio(temp_path, bucket_name, object_name)
        final_path = f"minio://{bucket_name}/{object_name}"
        
    elif storage_backend == "s3":
        # Upload to S3
        bucket_name = "llm-models"
        object_name = f"datasets/{file_name}"
        success = storage.upload_to_s3(temp_path, bucket_name, object_name)
        final_path = f"s3://{bucket_name}/{object_name}"
    
    # Cleanup
    os.remove(temp_path)
    
    return {
        "success": success,
        "filename": file_name,
        "path": final_path,
        "storage_backend": storage_backend
    }

if __name__ == "__main__":
    # This function can be used to test the fine-tuning locally or via modal run
    print("🧪 Testing fine-tuning function...")
    
    # Example usage
    result = fine_tune_llm.remote(
        model_name_or_path="microsoft/DialoGPT-small",
        dataset_path="dummy_dataset.jsonl",
        output_dir="/vol/test_finetuned_model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        storage_backend="volume"
    )
    
    print(f"✅ Test completed: {result}")

