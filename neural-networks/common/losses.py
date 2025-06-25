import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class Loss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the loss."""
        pass
        
    @abstractmethod
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss with respect to predictions."""
        pass

class MeanSquaredError(Loss):
    """Mean Squared Error loss function."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute MSE loss."""
        return np.mean((predictions - targets) ** 2)
        
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of MSE loss."""
        return 2 * (predictions - targets) / predictions.shape[0]

class CrossEntropyLoss(Loss):
    """Cross-entropy loss function for multi-class classification."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        if targets.ndim == 1:
            # Convert to one-hot encoding
            num_classes = predictions.shape[1]
            one_hot_targets = np.zeros((targets.shape[0], num_classes))
            one_hot_targets[np.arange(targets.shape[0]), targets] = 1
            targets = one_hot_targets
            
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))
        
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of cross-entropy loss."""
        # Clip predictions to prevent division by 0
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        if targets.ndim == 1:
            # Convert to one-hot encoding
            num_classes = predictions.shape[1]
            one_hot_targets = np.zeros((targets.shape[0], num_classes))
            one_hot_targets[np.arange(targets.shape[0]), targets] = 1
            targets = one_hot_targets
            
        return (predictions - targets) / predictions.shape[0]

class BinaryCrossEntropyLoss(Loss):
    """Binary cross-entropy loss function."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of binary cross-entropy loss."""
        # Clip predictions to prevent division by 0
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return (predictions - targets) / (predictions * (1 - predictions)) / predictions.shape[0]

class HuberLoss(Loss):
    """Huber loss function (robust regression)."""
    
    def __init__(self, delta: float = 1.0):
        self.delta = delta
        
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Huber loss."""
        residual = np.abs(predictions - targets)
        condition = residual <= self.delta
        
        squared_loss = 0.5 * (predictions - targets) ** 2
        linear_loss = self.delta * residual - 0.5 * self.delta ** 2
        
        return np.mean(np.where(condition, squared_loss, linear_loss))
        
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of Huber loss."""
        residual = predictions - targets
        condition = np.abs(residual) <= self.delta
        
        return np.where(condition, residual, self.delta * np.sign(residual)) / predictions.shape[0]

class HingeLoss(Loss):
    """Hinge loss function for SVM-style classification."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute hinge loss."""
        return np.mean(np.maximum(0, 1 - targets * predictions))
        
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of hinge loss."""
        condition = targets * predictions < 1
        return -targets * condition / predictions.shape[0]