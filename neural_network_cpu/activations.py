"""Activation Functions Implementation

CPU-optimized implementations of common activation functions with
stable numerical computations and efficient gradient calculations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    def __init__(self):
        self.input_cache = None
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through activation function."""
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through activation function."""
        pass
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)


class ReLU(ActivationFunction):
    """Rectified Linear Unit activation function."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """ReLU forward pass: max(0, x)"""
        self.input_cache = inputs.copy()
        return np.maximum(0, inputs)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """ReLU backward pass: gradient is 1 where input > 0, else 0"""
        grad_input = grad_output.copy()
        grad_input[self.input_cache <= 0] = 0
        return grad_input


class LeakyReLU(ActivationFunction):
    """Leaky Rectified Linear Unit activation function."""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Leaky ReLU forward pass: max(alpha*x, x)"""
        self.input_cache = inputs.copy()
        return np.where(inputs > 0, inputs, self.alpha * inputs)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Leaky ReLU backward pass"""
        grad_input = grad_output.copy()
        grad_input[self.input_cache <= 0] *= self.alpha
        return grad_input


class Sigmoid(ActivationFunction):
    """Sigmoid activation function with numerical stability."""
    
    def _stable_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        # Prevent overflow by clamping extreme values
        x = np.clip(x, -500, 500)
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Sigmoid forward pass: 1 / (1 + exp(-x))"""
        self.input_cache = inputs.copy()
        output = self._stable_sigmoid(inputs)
        self.output_cache = output  # Cache output for efficient backward pass
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Sigmoid backward pass: sigmoid(x) * (1 - sigmoid(x))"""
        sigmoid_grad = self.output_cache * (1 - self.output_cache)
        return grad_output * sigmoid_grad


class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Tanh forward pass"""
        self.input_cache = inputs.copy()
        output = np.tanh(inputs)
        self.output_cache = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Tanh backward pass: 1 - tanh^2(x)"""
        tanh_grad = 1 - self.output_cache ** 2
        return grad_output * tanh_grad


class Softmax(ActivationFunction):
    """Softmax activation function with numerical stability."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Stable softmax forward pass"""
        self.input_cache = inputs.copy()
        
        # Subtract max for numerical stability
        shifted_inputs = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        
        self.output_cache = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Softmax backward pass"""
        # For softmax, the gradient computation depends on the loss function
        # This is a simplified version
        softmax_output = self.output_cache
        
        # Compute Jacobian matrix for each sample
        batch_size = softmax_output.shape[0]
        grad_input = np.zeros_like(self.input_cache)
        
        for i in range(batch_size):
            s = softmax_output[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            grad_input[i] = np.dot(jacobian, grad_output[i])
        
        return grad_input


class ELU(ActivationFunction):
    """Exponential Linear Unit activation function."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """ELU forward pass: x if x > 0 else alpha * (exp(x) - 1)"""
        self.input_cache = inputs.copy()
        output = np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))
        self.output_cache = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """ELU backward pass"""
        grad_input = np.where(self.input_cache > 0, 
                             grad_output, 
                             grad_output * (self.output_cache + self.alpha))
        return grad_input


class Swish(ActivationFunction):
    """Swish activation function: x * sigmoid(x)"""
    
    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Swish forward pass: x * sigmoid(x)"""
        self.input_cache = inputs.copy()
        sigmoid_output = self.sigmoid.forward(inputs)
        output = inputs * sigmoid_output
        self.sigmoid_cache = sigmoid_output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Swish backward pass"""
        sigmoid_val = self.sigmoid_cache
        swish_grad = sigmoid_val + self.input_cache * sigmoid_val * (1 - sigmoid_val)
        return grad_output * swish_grad


class GELU(ActivationFunction):
    """Gaussian Error Linear Unit activation function."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """GELU forward pass (approximation): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))"""
        self.input_cache = inputs.copy()
        
        # GELU approximation
        coefficient = np.sqrt(2.0 / np.pi)
        inner = coefficient * (inputs + 0.044715 * inputs**3)
        output = 0.5 * inputs * (1 + np.tanh(inner))
        
        # Cache intermediate values for backward pass
        self.tanh_inner = np.tanh(inner)
        self.inner_cache = inner
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """GELU backward pass"""
        coefficient = np.sqrt(2.0 / np.pi)
        x = self.input_cache
        
        # Derivative computation
        sech2 = 1 - self.tanh_inner**2  # sech^2(inner)
        inner_grad = coefficient * (1 + 3 * 0.044715 * x**2)
        
        gelu_grad = 0.5 * (1 + self.tanh_inner) + 0.5 * x * sech2 * inner_grad
        
        return grad_output * gelu_grad


# Convenience function to get activation by name
def get_activation(name: str, **kwargs) -> ActivationFunction:
    """Factory function to get activation function by name."""
    activations = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax,
        'elu': ELU,
        'swish': Swish,
        'gelu': GELU
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name.lower()](**kwargs)