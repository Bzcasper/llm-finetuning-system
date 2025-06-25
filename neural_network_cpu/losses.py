"""Loss Functions Implementation

CPU-optimized implementations of common loss functions with
stable numerical computations and efficient gradient calculations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class LossFunction(ABC):
    """Abstract base class for loss functions."""
    
    def __init__(self):
        self.predictions_cache = None
        self.targets_cache = None
    
    @abstractmethod
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss value."""
        pass
    
    @abstractmethod
    def backward(self) -> np.ndarray:
        """Compute gradient of loss with respect to predictions."""
        pass
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(predictions, targets)


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss function."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """MSE forward pass: (1/2n) * sum((y_pred - y_true)^2)"""
        self.predictions_cache = predictions.copy()
        self.targets_cache = targets.copy()
        
        diff = predictions - targets
        loss = np.mean(0.5 * diff ** 2)
        return loss
    
    def backward(self) -> np.ndarray:
        """MSE backward pass: (y_pred - y_true) / n"""
        diff = self.predictions_cache - self.targets_cache
        grad = diff / self.predictions_cache.shape[0]
        return grad


class MeanAbsoluteError(LossFunction):
    """Mean Absolute Error loss function."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """MAE forward pass: (1/n) * sum(|y_pred - y_true|)"""
        self.predictions_cache = predictions.copy()
        self.targets_cache = targets.copy()
        
        diff = predictions - targets
        loss = np.mean(np.abs(diff))
        return loss
    
    def backward(self) -> np.ndarray:
        """MAE backward pass: sign(y_pred - y_true) / n"""
        diff = self.predictions_cache - self.targets_cache
        grad = np.sign(diff) / self.predictions_cache.shape[0]
        return grad


class CrossEntropyLoss(LossFunction):
    """Cross Entropy loss function with numerical stability."""
    
    def __init__(self, epsilon: float = 1e-15):
        super().__init__()
        self.epsilon = epsilon  # Small value to prevent log(0)
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Cross entropy forward pass: -sum(y_true * log(y_pred))"""
        self.predictions_cache = predictions.copy()
        self.targets_cache = targets.copy()
        
        # Clip predictions to prevent log(0)
        clipped_preds = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        # Handle both one-hot and integer targets
        if targets.ndim == 1 or targets.shape[1] == 1:
            # Integer targets
            if targets.ndim == 2:
                targets = targets.flatten()
            targets_int = targets.astype(int)
            
            # Extract predicted probabilities for correct classes
            correct_probs = clipped_preds[np.arange(len(targets_int)), targets_int]
            loss = -np.mean(np.log(correct_probs))\n        else:\n            # One-hot encoded targets\n            loss = -np.mean(np.sum(targets * np.log(clipped_preds), axis=1))\n        \n        return loss\n    \n    def backward(self) -> np.ndarray:\n        \"\"\"Cross entropy backward pass\"\"\"\n        predictions = self.predictions_cache\n        targets = self.targets_cache\n        \n        # Handle both one-hot and integer targets\n        if targets.ndim == 1 or targets.shape[1] == 1:\n            # Integer targets - convert to one-hot\n            if targets.ndim == 2:\n                targets = targets.flatten()\n            targets_int = targets.astype(int)\n            \n            one_hot = np.zeros_like(predictions)\n            one_hot[np.arange(len(targets_int)), targets_int] = 1\n            targets = one_hot\n        \n        # Gradient: (y_pred - y_true) / n\n        grad = (predictions - targets) / predictions.shape[0]\n        return grad\n\n\nclass BinaryCrossEntropyLoss(LossFunction):\n    \"\"\"Binary Cross Entropy loss function.\"\"\"\n    \n    def __init__(self, epsilon: float = 1e-15):\n        super().__init__()\n        self.epsilon = epsilon\n    \n    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:\n        \"\"\"Binary cross entropy forward pass\"\"\"\n        self.predictions_cache = predictions.copy()\n        self.targets_cache = targets.copy()\n        \n        # Clip predictions to prevent log(0)\n        clipped_preds = np.clip(predictions, self.epsilon, 1 - self.epsilon)\n        \n        # BCE: -[y*log(p) + (1-y)*log(1-p)]\n        loss = -np.mean(\n            targets * np.log(clipped_preds) + \n            (1 - targets) * np.log(1 - clipped_preds)\n        )\n        \n        return loss\n    \n    def backward(self) -> np.ndarray:\n        \"\"\"Binary cross entropy backward pass\"\"\"\n        predictions = self.predictions_cache\n        targets = self.targets_cache\n        \n        # Clip predictions to prevent division by 0\n        clipped_preds = np.clip(predictions, self.epsilon, 1 - self.epsilon)\n        \n        # Gradient: (y_pred - y_true) / (y_pred * (1 - y_pred)) / n\n        grad = (clipped_preds - targets) / (clipped_preds * (1 - clipped_preds))\n        grad = grad / predictions.shape[0]\n        \n        return grad\n\n\nclass HuberLoss(LossFunction):\n    \"\"\"Huber loss function (smooth L1 loss).\"\"\"\n    \n    def __init__(self, delta: float = 1.0):\n        super().__init__()\n        self.delta = delta\n    \n    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:\n        \"\"\"Huber loss forward pass\"\"\"\n        self.predictions_cache = predictions.copy()\n        self.targets_cache = targets.copy()\n        \n        diff = predictions - targets\n        abs_diff = np.abs(diff)\n        \n        # Huber loss: quadratic for small errors, linear for large errors\n        loss = np.where(\n            abs_diff <= self.delta,\n            0.5 * diff ** 2,\n            self.delta * abs_diff - 0.5 * self.delta ** 2\n        )\n        \n        return np.mean(loss)\n    \n    def backward(self) -> np.ndarray:\n        \"\"\"Huber loss backward pass\"\"\"\n        diff = self.predictions_cache - self.targets_cache\n        abs_diff = np.abs(diff)\n        \n        # Gradient: diff for small errors, delta*sign(diff) for large errors\n        grad = np.where(\n            abs_diff <= self.delta,\n            diff,\n            self.delta * np.sign(diff)\n        )\n        \n        return grad / self.predictions_cache.shape[0]\n\n\nclass FocalLoss(LossFunction):\n    \"\"\"Focal loss for addressing class imbalance.\"\"\"\n    \n    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, epsilon: float = 1e-15):\n        super().__init__()\n        self.alpha = alpha\n        self.gamma = gamma\n        self.epsilon = epsilon\n    \n    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:\n        \"\"\"Focal loss forward pass\"\"\"\n        self.predictions_cache = predictions.copy()\n        self.targets_cache = targets.copy()\n        \n        # Clip predictions\n        clipped_preds = np.clip(predictions, self.epsilon, 1 - self.epsilon)\n        \n        # Compute cross entropy\n        if targets.ndim == 1 or targets.shape[1] == 1:\n            # Integer targets\n            if targets.ndim == 2:\n                targets = targets.flatten()\n            targets_int = targets.astype(int)\n            \n            ce = -np.log(clipped_preds[np.arange(len(targets_int)), targets_int])\n            pt = clipped_preds[np.arange(len(targets_int)), targets_int]\n        else:\n            # One-hot targets\n            ce = -np.sum(targets * np.log(clipped_preds), axis=1)\n            pt = np.sum(targets * clipped_preds, axis=1)\n        \n        # Focal loss: alpha * (1 - pt)^gamma * ce\n        focal_weight = self.alpha * (1 - pt) ** self.gamma\n        loss = np.mean(focal_weight * ce)\n        \n        return loss\n    \n    def backward(self) -> np.ndarray:\n        \"\"\"Focal loss backward pass (simplified)\"\"\"\n        # Simplified gradient - full implementation is complex\n        predictions = self.predictions_cache\n        targets = self.targets_cache\n        \n        if targets.ndim == 1 or targets.shape[1] == 1:\n            if targets.ndim == 2:\n                targets = targets.flatten()\n            targets_int = targets.astype(int)\n            \n            one_hot = np.zeros_like(predictions)\n            one_hot[np.arange(len(targets_int)), targets_int] = 1\n            targets = one_hot\n        \n        # Simplified gradient (approximation)\n        grad = (predictions - targets) / predictions.shape[0]\n        return grad\n\n\nclass KLDivergenceLoss(LossFunction):\n    \"\"\"Kullback-Leibler Divergence loss.\"\"\"\n    \n    def __init__(self, epsilon: float = 1e-15):\n        super().__init__()\n        self.epsilon = epsilon\n    \n    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:\n        \"\"\"KL divergence forward pass: sum(y_true * log(y_true / y_pred))\"\"\"\n        self.predictions_cache = predictions.copy()\n        self.targets_cache = targets.copy()\n        \n        # Clip to prevent log(0)\n        clipped_preds = np.clip(predictions, self.epsilon, 1.0)\n        clipped_targets = np.clip(targets, self.epsilon, 1.0)\n        \n        # KL divergence\n        kl_div = clipped_targets * np.log(clipped_targets / clipped_preds)\n        loss = np.mean(np.sum(kl_div, axis=1))\n        \n        return loss\n    \n    def backward(self) -> np.ndarray:\n        \"\"\"KL divergence backward pass\"\"\"\n        predictions = self.predictions_cache\n        targets = self.targets_cache\n        \n        # Clip to prevent division by 0\n        clipped_preds = np.clip(predictions, self.epsilon, 1.0)\n        clipped_targets = np.clip(targets, self.epsilon, 1.0)\n        \n        # Gradient: -y_true / y_pred / n\n        grad = -clipped_targets / clipped_preds / predictions.shape[0]\n        \n        return grad\n\n\n# Convenience function to get loss by name\ndef get_loss(name: str, **kwargs) -> LossFunction:\n    \"\"\"Factory function to get loss function by name.\"\"\"\n    losses = {\n        'mse': MeanSquaredError,\n        'mae': MeanAbsoluteError,\n        'crossentropy': CrossEntropyLoss,\n        'binary_crossentropy': BinaryCrossEntropyLoss,\n        'huber': HuberLoss,\n        'focal': FocalLoss,\n        'kl_divergence': KLDivergenceLoss\n    }\n    \n    if name.lower() not in losses:\n        raise ValueError(f\"Unknown loss function: {name}\")\n    \n    return losses[name.lower()](**kwargs)