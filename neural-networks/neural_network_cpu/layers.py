"""Neural Network Layers

Implements various layer types with forward and backward passes
optimized for CPU performance using NumPy.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from .activations import ActivationFunction, ReLU


class Layer:
    """Base class for all neural network layers."""
    
    def __init__(self):
        self.trainable = True
        self.built = False
        self.input_shape = None
        self.output_shape = None
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer (initialize weights and biases)."""
        self.input_shape = input_shape
        self.built = True
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the layer."""
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get layer parameters (weights and biases)."""
        return {}
    
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set layer parameters."""
        pass
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape given input shape."""
        raise NotImplementedError


class DenseLayer(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, units: int, activation: Optional[ActivationFunction] = None,
                 use_bias: bool = True, kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros'):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        # Parameters
        self.weights = None
        self.bias = None
        
        # Gradients (for training)
        self.grad_weights = None
        self.grad_bias = None
        
        # Cache for backward pass
        self.cache = {}
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the dense layer."""
        super().build(input_shape)
        
        # Assuming input_shape is (batch_size, input_dim)
        input_dim = input_shape[-1]
        
        # Initialize weights
        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6.0 / (input_dim + self.units))
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.units))
        elif self.kernel_initializer == 'he_uniform':
            limit = np.sqrt(6.0 / input_dim)
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.units))
        elif self.kernel_initializer == 'normal':
            self.weights = np.random.normal(0, 0.01, (input_dim, self.units))
        else:  # zeros
            self.weights = np.zeros((input_dim, self.units))
        
        # Initialize bias
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                self.bias = np.zeros(self.units)
            else:  # random
                self.bias = np.random.normal(0, 0.01, self.units)
        
        self.output_shape = input_shape[:-1] + (self.units,)
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through dense layer."""
        if not self.built:
            self.build(inputs.shape)
        
        # Linear transformation: X @ W + b
        output = np.dot(inputs, self.weights)
        if self.use_bias:
            output += self.bias
        
        # Cache inputs for backward pass
        if training:
            self.cache['inputs'] = inputs
            self.cache['linear_output'] = output
        
        # Apply activation if specified
        if self.activation is not None:
            if training:
                self.cache['pre_activation'] = output
            output = self.activation.forward(output)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through dense layer."""
        # Apply activation gradient if activation exists
        if self.activation is not None:
            grad_output = grad_output * self.activation.backward(self.cache['pre_activation'])
        
        # Gradients w.r.t. weights and bias
        inputs = self.cache['inputs']
        batch_size = inputs.shape[0]
        
        self.grad_weights = np.dot(inputs.T, grad_output) / batch_size
        if self.use_bias:
            self.grad_bias = np.mean(grad_output, axis=0)
        
        # Gradient w.r.t. inputs
        grad_inputs = np.dot(grad_output, self.weights.T)
        
        return grad_inputs
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get layer parameters."""
        params = {'weights': self.weights}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set layer parameters."""
        if 'weights' in params:
            self.weights = params['weights']
        if 'bias' in params and self.use_bias:
            self.bias = params['bias']
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        return input_shape[:-1] + (self.units,)


class ConvolutionalLayer(Layer):
    """2D Convolutional layer (simplified implementation)."""
    
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: str = 'valid', activation: Optional[ActivationFunction] = None,
                 use_bias: bool = True):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        
        # Parameters
        self.weights = None  # (kernel_h, kernel_w, input_channels, filters)
        self.bias = None
        
        # Gradients
        self.grad_weights = None
        self.grad_bias = None
        
        # Cache
        self.cache = {}
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the convolutional layer."""
        super().build(input_shape)
        
        # Assuming input_shape is (batch_size, height, width, channels)
        _, input_h, input_w, input_channels = input_shape
        kernel_h, kernel_w = self.kernel_size
        
        # Initialize weights using He initialization
        fan_in = kernel_h * kernel_w * input_channels
        std = np.sqrt(2.0 / fan_in)
        self.weights = np.random.normal(0, std, (kernel_h, kernel_w, input_channels, self.filters))
        
        if self.use_bias:
            self.bias = np.zeros(self.filters)
        
        # Calculate output shape
        if self.padding == 'valid':
            output_h = (input_h - kernel_h) // self.stride[0] + 1
            output_w = (input_w - kernel_w) // self.stride[1] + 1
        else:  # 'same'
            output_h = input_h // self.stride[0]
            output_w = input_w // self.stride[1]
        
        self.output_shape = (input_shape[0], output_h, output_w, self.filters)
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through convolutional layer."""
        if not self.built:
            self.build(inputs.shape)
        
        batch_size, input_h, input_w, input_channels = inputs.shape
        kernel_h, kernel_w = self.kernel_size
        output_h, output_w = self.output_shape[1], self.output_shape[2]
        
        # Apply padding if needed
        if self.padding == 'same':
            pad_h = max(0, (output_h - 1) * self.stride[0] + kernel_h - input_h)
            pad_w = max(0, (output_w - 1) * self.stride[1] + kernel_w - input_w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            inputs_padded = np.pad(inputs, ((0, 0), (pad_top, pad_bottom), 
                                          (pad_left, pad_right), (0, 0)), mode='constant')
        else:
            inputs_padded = inputs
        
        # Convolution operation (naive implementation)
        output = np.zeros((batch_size, output_h, output_w, self.filters))
        
        for i in range(output_h):
            for j in range(output_w):
                start_i = i * self.stride[0]
                start_j = j * self.stride[1]
                end_i = start_i + kernel_h
                end_j = start_j + kernel_w
                
                # Extract patch
                patch = inputs_padded[:, start_i:end_i, start_j:end_j, :]
                
                # Convolve patch with all filters
                for f in range(self.filters):
                    output[:, i, j, f] = np.sum(patch * self.weights[:, :, :, f], axis=(1, 2, 3))
        
        # Add bias
        if self.use_bias:
            output += self.bias
        
        # Cache for backward pass
        if training:
            self.cache['inputs'] = inputs_padded
            self.cache['linear_output'] = output
        
        # Apply activation
        if self.activation is not None:
            if training:
                self.cache['pre_activation'] = output
            output = self.activation.forward(output)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through convolutional layer (simplified)."""
        # This is a simplified implementation
        # Full convolution backward pass is quite complex
        
        if self.activation is not None:
            grad_output = grad_output * self.activation.backward(self.cache['pre_activation'])
        
        # For now, return identity gradient (placeholder)
        # In a real implementation, this would compute proper convolution gradients
        return grad_output
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        _, input_h, input_w, input_channels = input_shape
        kernel_h, kernel_w = self.kernel_size
        
        if self.padding == 'valid':
            output_h = (input_h - kernel_h) // self.stride[0] + 1
            output_w = (input_w - kernel_w) // self.stride[1] + 1
        else:  # 'same'
            output_h = input_h // self.stride[0]
            output_w = input_w // self.stride[1]
        
        return (input_shape[0], output_h, output_w, self.filters)


class RecurrentLayer(Layer):
    """Simple Recurrent Neural Network layer."""
    
    def __init__(self, units: int, activation: Optional[ActivationFunction] = None,
                 return_sequences: bool = False, return_state: bool = False):
        super().__init__()
        self.units = units
        self.activation = activation if activation is not None else ReLU()
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Parameters
        self.W_input = None  # Input to hidden weights
        self.W_recurrent = None  # Hidden to hidden weights
        self.bias = None
        
        # Gradients
        self.grad_W_input = None
        self.grad_W_recurrent = None
        self.grad_bias = None
        
        # Cache
        self.cache = {}
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the recurrent layer."""
        super().build(input_shape)
        
        # Assuming input_shape is (batch_size, sequence_length, input_dim)
        batch_size, sequence_length, input_dim = input_shape
        
        # Initialize weights
        # Input to hidden weights
        limit = np.sqrt(1.0 / input_dim)
        self.W_input = np.random.uniform(-limit, limit, (input_dim, self.units))
        
        # Hidden to hidden weights
        limit = np.sqrt(1.0 / self.units)
        self.W_recurrent = np.random.uniform(-limit, limit, (self.units, self.units))
        
        # Bias
        self.bias = np.zeros(self.units)
        
        # Output shape
        if self.return_sequences:
            self.output_shape = (batch_size, sequence_length, self.units)
        else:
            self.output_shape = (batch_size, self.units)
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through recurrent layer."""
        if not self.built:
            self.build(inputs.shape)
        
        batch_size, sequence_length, input_dim = inputs.shape
        
        # Initialize hidden state
        hidden_state = np.zeros((batch_size, self.units))
        
        outputs = []
        hidden_states = [hidden_state]
        
        for t in range(sequence_length):
            # RNN step: h_t = activation(x_t @ W_input + h_{t-1} @ W_recurrent + bias)
            input_contribution = np.dot(inputs[:, t, :], self.W_input)
            recurrent_contribution = np.dot(hidden_state, self.W_recurrent)
            
            hidden_state = self.activation.forward(
                input_contribution + recurrent_contribution + self.bias
            )
            
            outputs.append(hidden_state)
            hidden_states.append(hidden_state)
        
        # Cache for backward pass
        if training:
            self.cache['inputs'] = inputs
            self.cache['hidden_states'] = hidden_states
            self.cache['outputs'] = outputs
        
        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through recurrent layer (simplified)."""
        # This is a simplified implementation
        # Full BPTT (Backpropagation Through Time) is quite complex
        
        # For now, return identity gradient (placeholder)
        batch_size, sequence_length, input_dim = self.cache['inputs'].shape
        return np.ones((batch_size, sequence_length, input_dim))
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        batch_size, sequence_length, input_dim = input_shape
        
        if self.return_sequences:
            return (batch_size, sequence_length, self.units)
        else:
            return (batch_size, self.units)


class LSTMLayer(Layer):
    """Long Short-Term Memory layer (simplified implementation)."""
    
    def __init__(self, units: int, return_sequences: bool = False):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        
        # LSTM has 4 gates: forget, input, candidate, output
        # Each gate has input weights, recurrent weights, and bias
        self.W_f = None  # Forget gate weights
        self.W_i = None  # Input gate weights
        self.W_c = None  # Candidate gate weights
        self.W_o = None  # Output gate weights
        
        self.U_f = None  # Forget gate recurrent weights
        self.U_i = None  # Input gate recurrent weights
        self.U_c = None  # Candidate gate recurrent weights
        self.U_o = None  # Output gate recurrent weights
        
        self.b_f = None  # Forget gate bias
        self.b_i = None  # Input gate bias
        self.b_c = None  # Candidate gate bias
        self.b_o = None  # Output gate bias
        
        # Cache
        self.cache = {}
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the LSTM layer."""
        super().build(input_shape)
        
        batch_size, sequence_length, input_dim = input_shape
        
        # Initialize all weights
        limit_input = np.sqrt(1.0 / input_dim)
        limit_recurrent = np.sqrt(1.0 / self.units)
        
        # Input weights
        self.W_f = np.random.uniform(-limit_input, limit_input, (input_dim, self.units))
        self.W_i = np.random.uniform(-limit_input, limit_input, (input_dim, self.units))
        self.W_c = np.random.uniform(-limit_input, limit_input, (input_dim, self.units))
        self.W_o = np.random.uniform(-limit_input, limit_input, (input_dim, self.units))
        
        # Recurrent weights
        self.U_f = np.random.uniform(-limit_recurrent, limit_recurrent, (self.units, self.units))
        self.U_i = np.random.uniform(-limit_recurrent, limit_recurrent, (self.units, self.units))
        self.U_c = np.random.uniform(-limit_recurrent, limit_recurrent, (self.units, self.units))
        self.U_o = np.random.uniform(-limit_recurrent, limit_recurrent, (self.units, self.units))
        
        # Biases (initialize forget gate bias to 1 for better gradient flow)
        self.b_f = np.ones(self.units)
        self.b_i = np.zeros(self.units)
        self.b_c = np.zeros(self.units)
        self.b_o = np.zeros(self.units)
        
        # Output shape
        if self.return_sequences:
            self.output_shape = (batch_size, sequence_length, self.units)
        else:
            self.output_shape = (batch_size, self.units)
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through LSTM layer."""
        if not self.built:
            self.build(inputs.shape)
        
        batch_size, sequence_length, input_dim = inputs.shape
        
        # Initialize hidden state and cell state
        h_t = np.zeros((batch_size, self.units))
        c_t = np.zeros((batch_size, self.units))
        
        outputs = []
        
        for t in range(sequence_length):
            x_t = inputs[:, t, :]
            
            # Forget gate
            f_t = self._sigmoid(np.dot(x_t, self.W_f) + np.dot(h_t, self.U_f) + self.b_f)
            
            # Input gate
            i_t = self._sigmoid(np.dot(x_t, self.W_i) + np.dot(h_t, self.U_i) + self.b_i)
            
            # Candidate values
            c_tilde = np.tanh(np.dot(x_t, self.W_c) + np.dot(h_t, self.U_c) + self.b_c)
            
            # Update cell state
            c_t = f_t * c_t + i_t * c_tilde
            
            # Output gate
            o_t = self._sigmoid(np.dot(x_t, self.W_o) + np.dot(h_t, self.U_o) + self.b_o)
            
            # Update hidden state
            h_t = o_t * np.tanh(c_t)
            
            outputs.append(h_t)
        
        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through LSTM layer (placeholder)."""
        # LSTM backward pass is very complex - this is a placeholder
        batch_size, sequence_length, input_dim = self.cache.get('inputs', (1, 1, 1))
        return np.ones((batch_size, sequence_length, input_dim))
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        batch_size, sequence_length, input_dim = input_shape
        
        if self.return_sequences:
            return (batch_size, sequence_length, self.units)
        else:
            return (batch_size, self.units)


class DropoutLayer(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.rate = rate
        self.cache = {}
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through dropout layer."""
        if training and self.rate > 0:
            # Generate dropout mask
            mask = np.random.binomial(1, 1 - self.rate, inputs.shape) / (1 - self.rate)
            self.cache['mask'] = mask
            return inputs * mask
        else:
            return inputs
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through dropout layer."""
        if 'mask' in self.cache:
            return grad_output * self.cache['mask']
        else:
            return grad_output
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Output shape is same as input shape."""
        return input_shape


class BatchNormalizationLayer(Layer):
    """Batch Normalization layer."""
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.99):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = None  # Scale parameter
        self.beta = None   # Shift parameter
        
        # Running statistics
        self.running_mean = None
        self.running_var = None
        
        # Cache
        self.cache = {}
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the batch normalization layer."""
        super().build(input_shape)
        
        # Initialize parameters
        feature_size = input_shape[-1]
        self.gamma = np.ones(feature_size)
        self.beta = np.zeros(feature_size)
        
        # Initialize running statistics
        self.running_mean = np.zeros(feature_size)
        self.running_var = np.ones(feature_size)
        
        self.output_shape = input_shape
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through batch normalization layer."""
        if not self.built:
            self.build(inputs.shape)
        
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            
            # Normalize
            x_normalized = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Cache for backward pass
            self.cache['inputs'] = inputs
            self.cache['x_normalized'] = x_normalized
            self.cache['batch_mean'] = batch_mean
            self.cache['batch_var'] = batch_var
        else:
            # Use running statistics
            x_normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through batch normalization layer."""
        # This is a simplified implementation
        # Full batch norm backward pass involves complex gradient calculations
        return grad_output
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Output shape is same as input shape."""
        return input_shape