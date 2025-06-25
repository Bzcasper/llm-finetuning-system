"""
Comprehensive unit tests for all API endpoints
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from conftest import (
    assert_valid_job_id, 
    assert_valid_training_config,
    assert_api_response_structure
)


@pytest.mark.unit
@pytest.mark.backend
class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, client: TestClient):
        """Test successful health check."""
        response = client.get("/api/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        expected_keys = ["status", "timestamp", "modal_connected"]
        assert_api_response_structure(data, expected_keys)
        assert data["status"] == "healthy"
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["modal_connected"], bool)
    
    def test_health_check_headers(self, client: TestClient):
        """Test health check response headers."""
        response = client.get("/api/health")
        
        assert response.headers["content-type"] == "application/json"
    
    @patch('api.health.check_modal_connection')
    def test_health_check_modal_disconnected(self, mock_modal_check, client: TestClient):
        """Test health check when Modal is disconnected."""
        mock_modal_check.return_value = False
        
        response = client.get("/api/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["modal_connected"] is False


@pytest.mark.unit
@pytest.mark.backend
class TestTrainingEndpoints:
    """Test training-related endpoints."""
    
    def test_training_start_valid_config(self, client: TestClient, training_config, mock_modal_function):
        """Test training start with valid configuration."""
        with patch('api.training.start_training_job') as mock_start:
            mock_start.return_value = {
                "job_id": "test-job-123",
                "status": "started",
                "config": training_config
            }
            
            response = client.post("/api/training/start", json=training_config)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            expected_keys = ["job_id", "status", "config"]
            assert_api_response_structure(data, expected_keys)
            assert_valid_job_id(data["job_id"])
            assert data["status"] == "started"
    
    def test_training_start_invalid_config(self, client: TestClient):
        """Test training start with invalid configuration."""
        invalid_config = {
            "model_name": "",  # Invalid empty model name
            "dataset_path": "",
            "num_train_epochs": -1,  # Invalid negative epochs
        }
        
        response = client.post("/api/training/start", json=invalid_config)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_training_start_missing_fields(self, client: TestClient):
        """Test training start with missing required fields."""
        incomplete_config = {
            "model_name": "test-model"
            # Missing other required fields
        }
        
        response = client.post("/api/training/start", json=incomplete_config)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_training_status_existing_job(self, client: TestClient, mock_training_job):
        """Test training status for existing job."""
        job_id = "test-job-123"
        
        with patch('api.training.get_training_status') as mock_status:
            mock_status.return_value = mock_training_job
            
            response = client.get(f"/api/training/status/{job_id}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            expected_keys = ["job_id", "status", "config", "metrics"]
            assert_api_response_structure(data, expected_keys)
            assert data["job_id"] == job_id
    
    def test_training_status_nonexistent_job(self, client: TestClient):
        """Test training status for non-existent job."""
        job_id = "nonexistent-job"
        
        with patch('api.training.get_training_status') as mock_status:
            mock_status.return_value = None
            
            response = client.get(f"/api/training/status/{job_id}")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_training_stop_existing_job(self, client: TestClient):
        """Test stopping an existing training job."""
        job_id = "test-job-123"
        
        with patch('api.training.stop_training_job') as mock_stop:
            mock_stop.return_value = {"status": "stopped", "message": "Job stopped successfully"}
            
            response = client.post(f"/api/training/stop/{job_id}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "stopped"
    
    def test_training_list_jobs(self, client: TestClient):
        """Test listing all training jobs."""
        mock_jobs = [
            {"job_id": "job-1", "status": "completed"},
            {"job_id": "job-2", "status": "running"}
        ]
        
        with patch('api.training.list_training_jobs') as mock_list:
            mock_list.return_value = mock_jobs
            
            response = client.get("/api/training/jobs")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "jobs" in data
            assert len(data["jobs"]) == 2
            assert data["jobs"][0]["job_id"] == "job-1"
    
    def test_training_logs_existing_job(self, client: TestClient):
        """Test getting logs for existing job."""
        job_id = "test-job-123"
        mock_logs = ["Log line 1", "Log line 2", "Log line 3"]
        
        with patch('api.training.get_training_logs') as mock_logs_func:
            mock_logs_func.return_value = {"logs": mock_logs}
            
            response = client.get(f"/api/training/logs/{job_id}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "logs" in data
            assert len(data["logs"]) == 3
            assert data["logs"][0] == "Log line 1"


@pytest.mark.unit
@pytest.mark.backend
class TestDatasetEndpoints:
    """Test dataset-related endpoints."""
    
    def test_datasets_list(self, client: TestClient, mock_modal_volume):
        """Test listing available datasets."""
        mock_datasets = ["dataset1.jsonl", "dataset2.jsonl", "dataset3.jsonl"]
        
        with patch('api.datasets.list_datasets') as mock_list:
            mock_list.return_value = {"datasets": mock_datasets}
            
            response = client.get("/api/datasets")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "datasets" in data
            assert len(data["datasets"]) == 3
    
    def test_dataset_upload_valid_file(self, client: TestClient, sample_jsonl_dataset, temp_dir):
        """Test uploading a valid dataset file."""
        with open(sample_jsonl_dataset, "rb") as f:
            files = {"file": ("test_dataset.jsonl", f, "application/json")}
            
            with patch('api.datasets.upload_dataset') as mock_upload:
                mock_upload.return_value = {
                    "filename": "test_dataset.jsonl",
                    "path": "/vol/test_dataset.jsonl",
                    "size": 1024
                }
                
                response = client.post("/api/upload", files=files)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                
                expected_keys = ["filename", "path", "size"]
                assert_api_response_structure(data, expected_keys)
    
    def test_dataset_upload_invalid_file_type(self, client: TestClient):
        """Test uploading invalid file type."""
        # Create a non-JSONL file
        files = {"file": ("invalid.txt", "This is not JSONL", "text/plain")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_dataset_download_existing(self, client: TestClient):
        """Test downloading an existing dataset."""
        dataset_name = "test_dataset.jsonl"
        
        with patch('api.datasets.download_dataset') as mock_download:
            mock_download.return_value = b'{"text": "sample data"}\n'
            
            response = client.get(f"/api/datasets/{dataset_name}/download")
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "application/octet-stream"
    
    def test_dataset_preview(self, client: TestClient, sample_dataset):
        """Test previewing dataset contents."""
        dataset_name = "test_dataset.jsonl"
        
        with patch('api.datasets.preview_dataset') as mock_preview:
            mock_preview.return_value = {
                "preview": sample_dataset[:3],
                "total_rows": len(sample_dataset),
                "columns": ["text"]
            }
            
            response = client.get(f"/api/datasets/{dataset_name}/preview")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            expected_keys = ["preview", "total_rows", "columns"]
            assert_api_response_structure(data, expected_keys)
            assert len(data["preview"]) == 3
    
    def test_dataset_delete_existing(self, client: TestClient):
        """Test deleting an existing dataset."""
        dataset_name = "test_dataset.jsonl"
        
        with patch('api.datasets.delete_dataset') as mock_delete:
            mock_delete.return_value = {"message": "Dataset deleted successfully"}
            
            response = client.delete(f"/api/datasets/{dataset_name}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "message" in data


@pytest.mark.unit
@pytest.mark.backend
class TestModalStatusEndpoints:
    """Test Modal.com status endpoints."""
    
    def test_modal_status_connected(self, client: TestClient):
        """Test Modal status when connected."""
        with patch('api.modal_status.get_modal_status') as mock_status:
            mock_status.return_value = {
                "connected": True,
                "environment": "ai-tool-pool",
                "app_name": "llm-finetuner",
                "functions": ["finetune_llm", "preprocess_data"]
            }
            
            response = client.get("/api/modal/status")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            expected_keys = ["connected", "environment", "app_name"]
            assert_api_response_structure(data, expected_keys)
            assert data["connected"] is True
    
    def test_modal_status_disconnected(self, client: TestClient):
        """Test Modal status when disconnected."""
        with patch('api.modal_status.get_modal_status') as mock_status:
            mock_status.return_value = {
                "connected": False,
                "error": "Connection failed"
            }
            
            response = client.get("/api/modal/status")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["connected"] is False
            assert "error" in data
    
    def test_modal_resources(self, client: TestClient, mock_gpu_info):
        """Test Modal resource information."""
        with patch('api.modal_status.get_modal_resources') as mock_resources:
            mock_resources.return_value = {
                "gpu_info": mock_gpu_info,
                "storage_info": {
                    "total_gb": 100,
                    "used_gb": 25,
                    "available_gb": 75
                }
            }
            
            response = client.get("/api/modal/resources")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            expected_keys = ["gpu_info", "storage_info"]
            assert_api_response_structure(data, expected_keys)


@pytest.mark.unit
@pytest.mark.backend
class TestModelEndpoints:
    """Test model-related endpoints."""
    
    def test_models_list_huggingface(self, client: TestClient):
        """Test listing HuggingFace models."""
        mock_models = [
            {"name": "microsoft/DialoGPT-small", "type": "conversational"},
            {"name": "gpt2", "type": "text-generation"},
            {"name": "distilbert-base-uncased", "type": "text-classification"}
        ]
        
        with patch('api.datasets.list_huggingface_models') as mock_list:
            mock_list.return_value = {"models": mock_models}
            
            response = client.get("/api/models")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "models" in data
            assert len(data["models"]) == 3
    
    def test_models_search(self, client: TestClient):
        """Test searching for models."""
        query = "DialoGPT"
        mock_results = [
            {"name": "microsoft/DialoGPT-small", "downloads": 1000000},
            {"name": "microsoft/DialoGPT-medium", "downloads": 500000}
        ]
        
        with patch('api.datasets.search_models') as mock_search:
            mock_search.return_value = {"results": mock_results}
            
            response = client.get(f"/api/models/search?q={query}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "results" in data
            assert len(data["results"]) == 2
    
    def test_model_info(self, client: TestClient):
        """Test getting model information."""
        model_name = "microsoft/DialoGPT-small"
        mock_info = {
            "name": model_name,
            "description": "A conversational AI model",
            "parameters": "117M",
            "license": "MIT",
            "tags": ["conversational", "pytorch"]
        }
        
        with patch('api.datasets.get_model_info') as mock_info_func:
            mock_info_func.return_value = mock_info
            
            response = client.get(f"/api/models/{model_name}/info")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            expected_keys = ["name", "description", "parameters"]
            assert_api_response_structure(data, expected_keys)


@pytest.mark.unit
@pytest.mark.backend
class TestErrorHandling:
    """Test error handling across endpoints."""
    
    def test_404_endpoint(self, client: TestClient):
        """Test non-existent endpoint."""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_method_not_allowed(self, client: TestClient):
        """Test wrong HTTP method."""
        response = client.put("/api/health")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_large_payload(self, client: TestClient):
        """Test extremely large payload."""
        large_payload = {"data": "x" * (10 * 1024 * 1024)}  # 10MB payload
        
        response = client.post("/api/training/start", json=large_payload)
        
        # Should fail with 413 or 422 depending on server config
        assert response.status_code in [
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_malformed_json(self, client: TestClient):
        """Test malformed JSON payload."""
        response = client.post(
            "/api/training/start",
            data='{"malformed": json}',
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('api.training.start_training_job')
    def test_internal_server_error(self, mock_start, client: TestClient, training_config):
        """Test internal server error handling."""
        mock_start.side_effect = Exception("Internal error")
        
        response = client.post("/api/training/start", json=training_config)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.unit
@pytest.mark.backend
class TestCORSHeaders:
    """Test CORS header configuration."""
    
    def test_cors_preflight(self, client: TestClient):
        """Test CORS preflight request."""
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_cors_actual_request(self, client: TestClient):
        """Test CORS headers on actual request."""
        response = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        # Note: TestClient may not include CORS headers, 
        # these would be tested in integration tests


@pytest.mark.unit
@pytest.mark.backend
class TestRateLimiting:
    """Test rate limiting (if implemented)."""
    
    def test_rate_limit_not_exceeded(self, client: TestClient):
        """Test normal request rate."""
        for i in range(5):
            response = client.get("/api/health")
            assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.slow
    def test_rate_limit_exceeded(self, client: TestClient):
        """Test rate limit enforcement."""
        # This test would only be relevant if rate limiting is implemented
        # For now, it's a placeholder
        pass