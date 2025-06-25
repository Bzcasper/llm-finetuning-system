"""
Pytest configuration and fixtures for the LLM Fine-tuning System tests
"""

import pytest
import asyncio
import tempfile
import shutil
import json
import os
from typing import Generator, Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import httpx
from faker import Faker

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def fake() -> Faker:
    """Faker instance for generating test data."""
    return Faker()


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> Generator[httpx.AsyncClient, None, None]:
    """Async HTTP client for testing."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset() -> List[Dict[str, Any]]:
    """Sample dataset for testing."""
    return [
        {"text": "This is a sample training text for fine-tuning."},
        {"text": "Another example sentence for training the model."},
        {"text": "Machine learning is fascinating and powerful."},
        {"text": "Natural language processing enables amazing applications."},
        {"text": "Fine-tuning allows customization of pre-trained models."}
    ]


@pytest.fixture
def sample_jsonl_dataset(sample_dataset: List[Dict[str, Any]], temp_dir: str) -> str:
    """Create a sample JSONL dataset file."""
    dataset_path = os.path.join(temp_dir, "sample_dataset.jsonl")
    with open(dataset_path, "w") as f:
        for item in sample_dataset:
            f.write(json.dumps(item) + "\n")
    return dataset_path


@pytest.fixture
def training_config() -> Dict[str, Any]:
    """Sample training configuration."""
    return {
        "model_name": "microsoft/DialoGPT-small",
        "dataset_path": "test_dataset.jsonl",
        "output_dir": "/vol/test_model",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "learning_rate": 0.0002,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "use_4bit": True,
        "optimizer": "adamw_torch",
        "gpu_type": "T4",
        "timeout": 1800
    }


@pytest.fixture
def mock_modal_app():
    """Mock Modal application."""
    with patch('modal.App') as mock_app:
        mock_instance = Mock()
        mock_app.return_value = mock_instance
        mock_app.lookup.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_modal_function():
    """Mock Modal function."""
    mock_func = AsyncMock()
    mock_func.remote.return_value = AsyncMock()
    mock_func.remote.return_value.return_value = {
        "status": "completed",
        "model_path": "/vol/test_model",
        "metrics": {"loss": 0.5, "accuracy": 0.85}
    }
    return mock_func


@pytest.fixture
def mock_modal_volume():
    """Mock Modal volume."""
    with patch('modal.Volume') as mock_volume:
        mock_instance = Mock()
        mock_volume.return_value = mock_instance
        mock_instance.listdir.return_value = ["dataset1.jsonl", "dataset2.jsonl"]
        mock_instance.read_file.return_value = b'{"text": "sample"}\n{"text": "data"}'
        yield mock_instance


@pytest.fixture
def mock_huggingface_datasets():
    """Mock HuggingFace datasets."""
    with patch('datasets.load_dataset') as mock_load:
        mock_dataset = Mock()
        mock_dataset.train = [
            {"text": "Sample training text 1"},
            {"text": "Sample training text 2"}
        ]
        mock_load.return_value = mock_dataset
        yield mock_load


@pytest.fixture
def mock_transformers():
    """Mock transformers library."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer, \
         patch('transformers.AutoModelForCausalLM') as mock_model:
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        yield {
            'tokenizer': mock_tokenizer_instance,
            'model': mock_model_instance
        }


@pytest.fixture
def mock_training_job():
    """Mock training job data."""
    return {
        "job_id": "test-job-123",
        "status": "running",
        "config": {
            "model_name": "microsoft/DialoGPT-small",
            "dataset_path": "test_dataset.jsonl",
            "num_train_epochs": 1
        },
        "metrics": {
            "epoch": 1,
            "loss": 0.5,
            "learning_rate": 0.0002
        },
        "start_time": "2023-01-01T00:00:00Z",
        "logs": ["Starting training...", "Epoch 1/1 completed"]
    }


@pytest.fixture
def mock_storage_manager():
    """Mock storage manager."""
    with patch('api.datasets.StorageManager') as mock_storage:
        mock_instance = Mock()
        mock_storage.return_value = mock_instance
        mock_instance.list_files.return_value = ["file1.jsonl", "file2.jsonl"]
        mock_instance.upload_file.return_value = {"path": "/vol/uploaded_file.jsonl"}
        mock_instance.download_file.return_value = b'{"text": "sample data"}'
        yield mock_instance


@pytest.fixture
def mock_gpu_info():
    """Mock GPU information."""
    return {
        "gpu_count": 1,
        "gpu_type": "T4",
        "memory_total": 15360,
        "memory_free": 12288,
        "memory_used": 3072,
        "utilization": 25.0
    }


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables."""
    env_vars = {
        "MODAL_PROFILE": "ai-tool-pool",
        "MODAL_TOKEN_ID": "test-token-id",
        "MODAL_TOKEN_SECRET": "test-token-secret",
        "HUGGINGFACE_TOKEN": "test-hf-token"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def disable_modal_imports():
    """Disable Modal imports for testing without Modal dependencies."""
    with patch.dict('sys.modules', {'modal': Mock()}):
        yield


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.headers = {"content-type": "application/json"}
    
    def json(self) -> Dict[str, Any]:
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", 
                request=Mock(), 
                response=self
            )


@pytest.fixture
def mock_http_responses():
    """Factory for creating mock HTTP responses."""
    def _create_mock_response(json_data: Dict[str, Any], status_code: int = 200):
        return MockResponse(json_data, status_code)
    
    return _create_mock_response


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "modal: marks tests that require Modal.com connection"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "modal" in item.nodeid:
            item.add_marker(pytest.mark.modal)
        elif "test_backend" in item.nodeid:
            item.add_marker(pytest.mark.backend)
        elif "test_frontend" in item.nodeid:
            item.add_marker(pytest.mark.frontend)


# Custom assertion helpers
def assert_valid_job_id(job_id: str):
    """Assert that a job ID is valid."""
    assert isinstance(job_id, str)
    assert len(job_id) > 0
    assert job_id.replace("-", "").replace("_", "").isalnum()


def assert_valid_training_config(config: Dict[str, Any]):
    """Assert that a training configuration is valid."""
    required_fields = [
        "model_name", "dataset_path", "output_dir", 
        "num_train_epochs", "learning_rate"
    ]
    for field in required_fields:
        assert field in config, f"Missing required field: {field}"
        assert config[field] is not None, f"Field {field} cannot be None"


def assert_api_response_structure(response_data: Dict[str, Any], expected_keys: List[str]):
    """Assert that API response has expected structure."""
    assert isinstance(response_data, dict)
    for key in expected_keys:
        assert key in response_data, f"Missing key in response: {key}"