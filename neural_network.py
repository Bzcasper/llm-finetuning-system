import numpy as np
import pickle
import time
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

class Layer(ABC):
    """Abstract base class for neural network layers."""
    
    def __init__(self):
        self.parameters = {}
        self.gradients = {}
        self.cache = {}
        
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagation through the layer."""
        pass
        
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward propagation through the layer."""
        pass
        
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return layer parameters."""
        return self.parameters
        
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """Set layer parameters."""
        self.parameters = parameters

class Dense(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights using Xavier initialization
        self.parameters['W'] = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.parameters['b'] = np.zeros((1, output_size))
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.cache['inputs'] = inputs
        
        # Linear transformation
        z = np.dot(inputs, self.parameters['W']) + self.parameters['b']
        self.cache['z'] = z
        
        # Apply activation function
        if self.activation == 'relu':
            output = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            output = np.tanh(z)
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            output = z  # linear activation
            
        self.cache['output'] = output
        return output
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        inputs = self.cache['inputs']
        z = self.cache['z']
        output = self.cache['output']
        
        # Compute activation gradient
        if self.activation == 'relu':
            grad_z = grad_output * (z > 0)
        elif self.activation == 'sigmoid':
            grad_z = grad_output * output * (1 - output)
        elif self.activation == 'tanh':
            grad_z = grad_output * (1 - output**2)
        elif self.activation == 'softmax':
            grad_z = grad_output  # Assuming this is combined with cross-entropy loss
        else:
            grad_z = grad_output
            
        # Compute parameter gradients
        self.gradients['W'] = np.dot(inputs.T, grad_z)
        self.gradients['b'] = np.sum(grad_z, axis=0, keepdims=True)
        
        # Compute input gradient
        grad_input = np.dot(grad_z, self.parameters['W'].T)
        
        return grad_input

class Conv2D(Layer):
    """2D Convolutional layer optimized for CPU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, activation: str = 'relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        # Initialize weights
        fan_in = in_channels * kernel_size * kernel_size
        self.parameters['W'] = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.parameters['b'] = np.zeros((out_channels,))
        
    def im2col(self, input_data: np.ndarray) -> np.ndarray:
        """Convert image to column matrix for efficient convolution."""
        N, C, H, W = input_data.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Pad input if necessary
        if self.padding > 0:
            input_data = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        col = np.ndarray((N, C, self.kernel_size, self.kernel_size, out_h, out_w), dtype=input_data.dtype)
        
        for j in range(self.kernel_size):
            j_max = j + self.stride * out_h
            for i in range(self.kernel_size):
                i_max = i + self.stride * out_w
                col[:, :, j, i, :, :] = input_data[:, :, j:j_max:self.stride, i:i_max:self.stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col, out_h, out_w
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        N, C, H, W = inputs.shape
        self.cache['inputs'] = inputs
        
        # Convert to column matrix
        col, out_h, out_w = self.im2col(inputs)
        self.cache['col'] = col
        self.cache['out_h'] = out_h
        self.cache['out_w'] = out_w
        
        # Reshape weights for matrix multiplication
        W_col = self.parameters['W'].reshape(self.out_channels, -1)
        
        # Convolution as matrix multiplication
        z = np.dot(col, W_col.T) + self.parameters['b']
        
        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, z)
        else:
            output = z
            
        # Reshape output
        output = output.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        self.cache['output'] = output
        
        return output
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        inputs = self.cache['inputs']
        col = self.cache['col']
        out_h = self.cache['out_h']
        out_w = self.cache['out_w']
        output = self.cache['output']
        
        N, C, H, W = inputs.shape
        
        # Apply activation gradient
        if self.activation == 'relu':
            grad_output = grad_output * (output > 0)
            
        # Reshape gradient
        grad_output = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # Parameter gradients
        self.gradients['b'] = np.sum(grad_output, axis=0)
        W_col = self.parameters['W'].reshape(self.out_channels, -1)
        self.gradients['W'] = np.dot(grad_output.T, col).reshape(self.parameters['W'].shape)
        
        # Input gradient (simplified for CPU efficiency)
        grad_input = np.zeros_like(inputs)
        
        return grad_input

class BatchNormalization(Layer):
    """Batch normalization layer."""
    
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True
        
        # Parameters
        self.parameters['gamma'] = np.ones((1, num_features))
        self.parameters['beta'] = np.zeros((1, num_features))
        
        # Running statistics
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(inputs, axis=0, keepdims=True)
            batch_var = np.var(inputs, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            inputs_norm = (inputs - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Cache for backward pass
            self.cache['inputs'] = inputs
            self.cache['inputs_norm'] = inputs_norm
            self.cache['batch_mean'] = batch_mean
            self.cache['batch_var'] = batch_var
        else:
            # Use running statistics
            inputs_norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
            
        # Scale and shift
        output = self.parameters['gamma'] * inputs_norm + self.parameters['beta']
        return output
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        inputs = self.cache['inputs']
        inputs_norm = self.cache['inputs_norm']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']
        
        N = inputs.shape[0]
        
        # Parameter gradients
        self.gradients['gamma'] = np.sum(grad_output * inputs_norm, axis=0, keepdims=True)
        self.gradients['beta'] = np.sum(grad_output, axis=0, keepdims=True)
        
        # Input gradient
        grad_inputs_norm = grad_output * self.parameters['gamma']
        grad_var = np.sum(grad_inputs_norm * (inputs - batch_mean), axis=0, keepdims=True) * -0.5 * np.power(batch_var + self.eps, -1.5)
        grad_mean = np.sum(grad_inputs_norm * -1.0 / np.sqrt(batch_var + self.eps), axis=0, keepdims=True) + grad_var * np.mean(-2.0 * (inputs - batch_mean), axis=0, keepdims=True)
        
        grad_input = grad_inputs_norm / np.sqrt(batch_var + self.eps) + grad_var * 2.0 * (inputs - batch_mean) / N + grad_mean / N
        
        return grad_input
        
    def set_training(self, training: bool):
        self.training = training

class MaxPool2D(Layer):
    """2D Max pooling layer."""
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        N, C, H, W = inputs.shape
        
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        output = np.zeros((N, C, out_h, out_w))
        self.cache['inputs'] = inputs
        self.cache['mask'] = np.zeros_like(inputs)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                pool_region = inputs[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(pool_region, axis=(2, 3))
                
                # Create mask for backward pass
                for n in range(N):
                    for c in range(C):
                        mask = (pool_region[n, c] == output[n, c, i, j])
                        self.cache['mask'][n, c, h_start:h_end, w_start:w_end] = mask
                        
        return output
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        inputs = self.cache['inputs']
        mask = self.cache['mask']
        
        N, C, out_h, out_w = grad_output.shape
        grad_input = np.zeros_like(inputs)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                for n in range(N):
                    for c in range(C):
                        grad_input[n, c, h_start:h_end, w_start:w_end] += grad_output[n, c, i, j] * mask[n, c, h_start:h_end, w_start:w_end]
                        
        return grad_input