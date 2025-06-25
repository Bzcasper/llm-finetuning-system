import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    @abstractmethod
    def update(self, layer, gradients: Dict[str, np.ndarray]):
        """Update layer parameters using gradients."""
        pass
        
    @abstractmethod
    def reset(self):
        """Reset optimizer state."""
        pass

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
        
    def update(self, layer, gradients: Dict[str, np.ndarray]):
        layer_id = id(layer)
        
        if layer_id not in self.velocity:
            self.velocity[layer_id] = {}
            for param_name in gradients:
                self.velocity[layer_id][param_name] = np.zeros_like(gradients[param_name])
        
        for param_name, grad in gradients.items():
            # Update velocity
            self.velocity[layer_id][param_name] = (self.momentum * self.velocity[layer_id][param_name] - 
                                                   self.learning_rate * grad)
            
            # Update parameters
            layer.parameters[param_name] += self.velocity[layer_id][param_name]
            
    def reset(self):
        self.velocity = {}

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
        
    def update(self, layer, gradients: Dict[str, np.ndarray]):
        layer_id = id(layer)
        self.t += 1
        
        if layer_id not in self.m:
            self.m[layer_id] = {}
            self.v[layer_id] = {}
            for param_name in gradients:
                self.m[layer_id][param_name] = np.zeros_like(gradients[param_name])
                self.v[layer_id][param_name] = np.zeros_like(gradients[param_name])
        
        for param_name, grad in gradients.items():
            # Update biased first moment estimate
            self.m[layer_id][param_name] = self.beta1 * self.m[layer_id][param_name] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[layer_id][param_name] = self.beta2 * self.v[layer_id][param_name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[layer_id][param_name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[layer_id][param_name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            layer.parameters[param_name] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            
    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0

class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
        
    def update(self, layer, gradients: Dict[str, np.ndarray]):
        layer_id = id(layer)
        
        if layer_id not in self.cache:
            self.cache[layer_id] = {}
            for param_name in gradients:
                self.cache[layer_id][param_name] = np.zeros_like(gradients[param_name])
        
        for param_name, grad in gradients.items():
            # Update cache
            self.cache[layer_id][param_name] = (self.decay_rate * self.cache[layer_id][param_name] + 
                                                (1 - self.decay_rate) * grad ** 2)
            
            # Update parameters
            layer.parameters[param_name] -= (self.learning_rate * grad / 
                                             (np.sqrt(self.cache[layer_id][param_name]) + self.epsilon))
            
    def reset(self):
        self.cache = {}

class AdaGrad(Optimizer):
    """AdaGrad optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}
        
    def update(self, layer, gradients: Dict[str, np.ndarray]):
        layer_id = id(layer)
        
        if layer_id not in self.cache:
            self.cache[layer_id] = {}
            for param_name in gradients:
                self.cache[layer_id][param_name] = np.zeros_like(gradients[param_name])
        
        for param_name, grad in gradients.items():
            # Accumulate squared gradients
            self.cache[layer_id][param_name] += grad ** 2
            
            # Update parameters
            layer.parameters[param_name] -= (self.learning_rate * grad / 
                                             (np.sqrt(self.cache[layer_id][param_name]) + self.epsilon))
            
    def reset(self):
        self.cache = {}