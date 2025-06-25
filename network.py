import numpy as np
import pickle
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from neural_network import Layer, Dense, Conv2D, BatchNormalization, MaxPool2D
from optimizers import Optimizer, Adam, SGD
from losses import Loss, CrossEntropyLoss, MeanSquaredError

class NeuralNetwork:
    """CPU-optimized neural network implementation."""
    
    def __init__(self, layers: List[Layer], loss: Loss, optimizer: Optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward propagation through the network."""
        output = X
        
        for layer in self.layers:
            # Set training mode for BatchNormalization layers
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
            output = layer.forward(output)
            
        return output
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward propagation through the network."""
        grad = grad_output
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        return grad
        
    def update_parameters(self):
        """Update parameters using the optimizer."""
        for layer in self.layers:
            if hasattr(layer, 'parameters') and layer.parameters:
                self.optimizer.update(layer, layer.gradients)
                
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the loss."""
        return self.loss.forward(predictions, targets)
        
    def compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute classification accuracy."""
        if predictions.shape[1] > 1:  # Multi-class
            pred_classes = np.argmax(predictions, axis=1)
            if targets.ndim > 1:
                target_classes = np.argmax(targets, axis=1)
            else:
                target_classes = targets
            return np.mean(pred_classes == target_classes)
        else:  # Binary classification
            pred_classes = (predictions > 0.5).astype(int).flatten()
            return np.mean(pred_classes == targets.flatten())
            
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Perform one training step."""
        # Forward pass
        predictions = self.forward(X, training=True)
        
        # Compute loss
        loss = self.compute_loss(predictions, y)
        
        # Compute accuracy
        accuracy = self.compute_accuracy(predictions, y)
        
        # Backward pass
        grad_loss = self.loss.backward(predictions, y)
        self.backward(grad_loss)
        
        # Update parameters
        self.update_parameters()
        
        return loss, accuracy
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on validation/test data."""
        predictions = self.forward(X, training=False)
        loss = self.compute_loss(predictions, y)
        accuracy = self.compute_accuracy(predictions, y)
        return loss, accuracy
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, verbose: bool = True,
            early_stopping: bool = False, patience: int = 10) -> Dict[str, List[float]]:
        """Train the neural network."""
        
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                batch_loss, batch_accuracy = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                
            # Average metrics over batches
            epoch_loss /= n_batches
            epoch_accuracy /= n_batches
            
            self.training_history['loss'].append(epoch_loss)
            self.training_history['accuracy'].append(epoch_accuracy)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self.evaluate(X_val, y_val)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break
            
            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - "
                      f"loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}", end="")
                
                if X_val is not None and y_val is not None:
                    print(f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
                else:
                    print()
                    
        return self.training_history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.forward(X, training=False)
        
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for classification tasks."""
        predictions = self.predict(X)
        
        if predictions.shape[1] > 1:  # Multi-class
            return np.argmax(predictions, axis=1)
        else:  # Binary classification
            return (predictions > 0.5).astype(int).flatten()
            
    def save(self, filepath: str):
        """Save the model to a file."""
        model_data = {
            'layers': self.layers,
            'loss': self.loss,
            'optimizer': self.optimizer,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load(cls, filepath: str):
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        model = cls(model_data['layers'], model_data['loss'], model_data['optimizer'])
        model.training_history = model_data['training_history']
        
        return model
        
    def summary(self):
        """Print model summary."""
        print("Neural Network Summary")
        print("=" * 50)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_params = 0
            layer_name = layer.__class__.__name__
            
            if hasattr(layer, 'parameters'):
                for param_name, param in layer.parameters.items():
                    layer_params += param.size
                    
            total_params += layer_params
            
            print(f"Layer {i + 1}: {layer_name} - Parameters: {layer_params:,}")
            
        print("=" * 50)
        print(f"Total Parameters: {total_params:,}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Loss Function: {self.loss.__class__.__name__}")

# Utility functions for building common architectures
def create_mlp(input_size: int, hidden_sizes: List[int], output_size: int, 
               activation: str = 'relu', output_activation: str = 'softmax') -> List[Layer]:
    """Create a Multi-Layer Perceptron architecture."""
    layers = []
    
    # Input to first hidden layer
    if hidden_sizes:
        layers.append(Dense(input_size, hidden_sizes[0], activation))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(Dense(hidden_sizes[i-1], hidden_sizes[i], activation))
            
        # Output layer
        layers.append(Dense(hidden_sizes[-1], output_size, output_activation))
    else:
        # Direct input to output
        layers.append(Dense(input_size, output_size, output_activation))
        
    return layers

def create_cnn(input_channels: int, num_classes: int) -> List[Layer]:
    """Create a simple CNN architecture."""
    layers = [
        Conv2D(input_channels, 32, 3, padding=1, activation='relu'),
        BatchNormalization(32),
        MaxPool2D(2, 2),
        
        Conv2D(32, 64, 3, padding=1, activation='relu'),
        BatchNormalization(64),
        MaxPool2D(2, 2),
        
        Conv2D(64, 128, 3, padding=1, activation='relu'),
        BatchNormalization(128),
        MaxPool2D(2, 2),
        
        # Flatten and dense layers would be added here
        # This is a simplified version for demonstration
    ]
    
    return layers