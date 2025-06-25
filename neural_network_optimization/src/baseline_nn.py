"""
Baseline Neural Network Implementation
=====================================

This module contains a basic neural network implementation using standard NumPy operations.
It serves as the baseline for performance comparisons with optimized versions.

Features:
- Multi-layer perceptron with configurable architecture
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Backpropagation with gradient descent
- Basic regularization (L2)
- Training utilities and metrics
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import pickle


@dataclass
class TrainingMetrics:
    """Container for training metrics and timing information."""
    epoch: int
    loss: float
    accuracy: float
    forward_time: float
    backward_time: float
    total_time: float


class ActivationFunction:
    """Container for activation functions and their derivatives."""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Layer:
    """Individual neural network layer with weights, biases, and activation."""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights using Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        
        # Cache for forward and backward passes
        self.last_input = None
        self.last_output = None
        self.last_activation_input = None
        
        # Get activation functions
        self.activation_fn, self.activation_derivative = self._get_activation_functions()
    
    def _get_activation_functions(self) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative."""
        if self.activation == 'relu':
            return ActivationFunction.relu, ActivationFunction.relu_derivative
        elif self.activation == 'sigmoid':
            return ActivationFunction.sigmoid, ActivationFunction.sigmoid_derivative
        elif self.activation == 'tanh':
            return ActivationFunction.tanh, ActivationFunction.tanh_derivative
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        self.last_input = x.copy()
        
        # Linear transformation: z = x @ W + b
        z = np.dot(x, self.weights) + self.biases
        self.last_activation_input = z.copy()
        
        # Apply activation function
        output = self.activation_fn(z)
        self.last_output = output.copy()
        
        return output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""
        # Apply activation derivative
        activation_grad = self.activation_derivative(self.last_activation_input)
        delta = gradient * activation_grad
        
        # Compute gradients
        weight_gradient = np.dot(self.last_input.T, delta)
        bias_gradient = np.sum(delta, axis=0, keepdims=True)
        input_gradient = np.dot(delta, self.weights.T)
        
        return input_gradient, weight_gradient, bias_gradient


class BaselineNeuralNetwork:
    """
    Baseline Neural Network Implementation
    
    A multi-layer perceptron with configurable architecture, designed to be
    simple and unoptimized for use as a performance baseline.
    """
    
    def __init__(self, layer_sizes: List[int], activations: List[str] = None,
                 learning_rate: float = 0.001, l2_regularization: float = 0.01):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions for each layer
            learning_rate: Learning rate for gradient descent
            l2_regularization: L2 regularization coefficient
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        
        # Default activations
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['sigmoid']
        
        self.activations = activations
        
        # Create layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            self.layers.append(layer)
        
        # Training history
        self.training_history = []
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the entire network."""
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Backward pass through the entire network."""
        batch_size = x.shape[0]
        
        # Compute initial gradient (assuming mean squared error)
        gradient = 2 * (y_pred - y_true) / batch_size
        
        # Backpropagate through layers in reverse order
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            gradient, weight_grad, bias_grad = layer.backward(gradient)
            
            # Apply L2 regularization to weights
            weight_grad += self.l2_regularization * layer.weights
            
            # Update weights and biases
            layer.weights -= self.learning_rate * weight_grad
            layer.biases -= self.learning_rate * bias_grad
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error loss with L2 regularization."""
        mse_loss = np.mean((y_pred - y_true) ** 2)
        
        # Add L2 regularization
        l2_loss = 0
        for layer in self.layers:
            l2_loss += np.sum(layer.weights ** 2)
        l2_loss *= self.l2_regularization / 2
        
        return mse_loss + l2_loss
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        if y_true.shape[1] == 1:  # Binary classification
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions == y_true)
        else:  # Multi-class classification
            true_labels = np.argmax(y_true, axis=1)
            pred_labels = np.argmax(y_pred, axis=1)
            return np.mean(true_labels == pred_labels)
    
    def train_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """Train on a single batch and return timing information."""
        # Forward pass
        start_time = time.time()
        y_pred = self.forward(x)
        forward_time = time.time() - start_time
        
        # Compute loss and accuracy
        loss = self.compute_loss(y, y_pred)
        accuracy = self.compute_accuracy(y, y_pred)
        
        # Backward pass
        start_time = time.time()
        self.backward(x, y, y_pred)
        backward_time = time.time() - start_time
        
        return loss, accuracy, forward_time, backward_time
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, verbose: bool = True) -> List[TrainingMetrics]:
        """
        Train the neural network.
        
        Args:
            x_train: Training input data
            y_train: Training target data
            x_val: Validation input data (optional)
            y_val: Validation target data (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print progress
            
        Returns:
            List of training metrics for each epoch
        """
        n_samples = x_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        epoch_metrics = []
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            # Training metrics
            total_loss = 0
            total_accuracy = 0
            total_forward_time = 0
            total_backward_time = 0
            
            # Process batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss, accuracy, forward_time, backward_time = self.train_batch(x_batch, y_batch)
                
                total_loss += loss
                total_accuracy += accuracy
                total_forward_time += forward_time
                total_backward_time += backward_time
            
            # Average metrics over batches
            avg_loss = total_loss / n_batches
            avg_accuracy = total_accuracy / n_batches
            epoch_time = time.time() - epoch_start_time
            
            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                loss=avg_loss,
                accuracy=avg_accuracy,
                forward_time=total_forward_time,
                backward_time=total_backward_time,
                total_time=epoch_time
            )
            
            epoch_metrics.append(metrics)
            self.training_history.append(metrics)
            
            # Validation evaluation
            val_loss, val_accuracy = None, None
            if x_val is not None and y_val is not None:
                y_val_pred = self.forward(x_val)
                val_loss = self.compute_loss(y_val, y_val_pred)
                val_accuracy = self.compute_accuracy(y_val, y_val_pred)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
                if val_loss is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                print(f"  Forward Time: {total_forward_time:.4f}s, Backward Time: {total_backward_time:.4f}s")
                print(f"  Total Epoch Time: {epoch_time:.4f}s")
                print()
        
        return epoch_metrics
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        return self.forward(x)
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'learning_rate': self.learning_rate,
            'l2_regularization': self.l2_regularization,
            'layers': []
        }
        
        for layer in self.layers:
            layer_data = {
                'weights': layer.weights,
                'biases': layer.biases,
                'activation': layer.activation
            }
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.layer_sizes = model_data['layer_sizes']
        self.activations = model_data['activations']
        self.learning_rate = model_data['learning_rate']
        self.l2_regularization = model_data['l2_regularization']
        
        # Recreate layers with loaded weights
        self.layers = []
        for i, layer_data in enumerate(model_data['layers']):
            layer = Layer(self.layer_sizes[i], self.layer_sizes[i + 1], layer_data['activation'])
            layer.weights = layer_data['weights']
            layer.biases = layer_data['biases']
            self.layers.append(layer)


def generate_sample_data(n_samples: int = 1000, n_features: int = 20, n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample classification data for testing."""
    np.random.seed(42)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels based on a linear combination of features with noise
    weights = np.random.randn(n_features)
    linear_combination = X @ weights
    
    if n_classes == 2:
        # Binary classification
        y = (linear_combination + np.random.randn(n_samples) * 0.1 > 0).astype(float).reshape(-1, 1)
    else:
        # Multi-class classification
        y = np.zeros((n_samples, n_classes))
        class_indices = np.abs(linear_combination + np.random.randn(n_samples) * 0.1) % n_classes
        y[np.arange(n_samples), class_indices.astype(int)] = 1
    
    return X, y


def main():
    """Demonstration of the baseline neural network."""
    print("Baseline Neural Network Performance Test")
    print("=" * 50)
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=5000, n_features=50, n_classes=2)
    
    # Split into train/validation sets
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print()
    
    # Create and train model
    print("Creating baseline neural network...")
    model = BaselineNeuralNetwork(
        layer_sizes=[50, 128, 64, 32, 1],
        activations=['relu', 'relu', 'relu', 'sigmoid'],
        learning_rate=0.001,
        l2_regularization=0.01
    )
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    metrics = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=64,
        verbose=True
    )
    total_training_time = time.time() - start_time
    
    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time / len(metrics):.2f} seconds")
    
    # Final evaluation
    print("\nFinal Performance:")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_loss = model.compute_loss(y_train, y_train_pred)
    train_accuracy = model.compute_accuracy(y_train, y_train_pred)
    val_loss = model.compute_loss(y_val, y_val_pred)
    val_accuracy = model.compute_accuracy(y_val, y_val_pred)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Save training history
    print("\nSaving results...")
    model.save_model('/home/bc/neural_network_optimization/data/baseline_model.pkl')
    
    # Save performance metrics
    performance_data = {
        'metrics': metrics,
        'total_training_time': total_training_time,
        'final_train_loss': train_loss,
        'final_train_accuracy': train_accuracy,
        'final_val_loss': val_loss,
        'final_val_accuracy': val_accuracy
    }
    
    with open('/home/bc/neural_network_optimization/data/baseline_performance.pkl', 'wb') as f:
        pickle.dump(performance_data, f)
    
    print("Baseline implementation complete!")


if __name__ == "__main__":
    main()