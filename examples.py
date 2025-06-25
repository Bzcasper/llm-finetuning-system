#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

from network import NeuralNetwork, create_mlp
from neural_network import Dense
from optimizers import Adam, SGD, RMSprop
from losses import CrossEntropyLoss, MeanSquaredError

def example_classification():
    """Example: Multi-class classification with synthetic data."""
    print("=" * 60)
    print("EXAMPLE 1: Multi-class Classification")
    print("=" * 60)
    
    # Generate synthetic classification data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=3, 
                              n_informative=15, n_redundant=5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Create neural network
    layers = create_mlp(input_size=20, hidden_sizes=[64, 32, 16], output_size=3)
    optimizer = Adam(learning_rate=0.001)
    loss = CrossEntropyLoss()
    
    model = NeuralNetwork(layers, loss, optimizer)
    model.summary()
    
    # Train the model
    print("\nTraining started...")
    start_time = time.time()
    
    history = model.fit(X_train, y_train, X_val, y_val, 
                       epochs=100, batch_size=32, verbose=True,
                       early_stopping=True, patience=10)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    predictions = model.predict_classes(X_test[:10])
    print(f"\nSample Predictions: {predictions}")
    print(f"True Labels: {y_test[:10]}")
    
    return model, history

def example_regression():
    """Example: Regression with synthetic data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Regression")
    print("=" * 60)
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    y = y.reshape(-1, 1)  # Reshape for neural network
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Standardize features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Create neural network for regression
    layers = [
        Dense(10, 64, activation='relu'),
        Dense(64, 32, activation='relu'),
        Dense(32, 16, activation='relu'),
        Dense(16, 1, activation='linear')  # Linear output for regression
    ]
    
    optimizer = Adam(learning_rate=0.001)
    loss = MeanSquaredError()
    
    model = NeuralNetwork(layers, loss, optimizer)
    model.summary()
    
    # Train the model
    print("\nTraining started...")
    start_time = time.time()
    
    history = model.fit(X_train, y_train, X_val, y_val, 
                       epochs=100, batch_size=32, verbose=True)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    test_loss, _ = model.evaluate(X_test, y_test)  # Accuracy not meaningful for regression
    print(f"\nTest Results:")
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test RMSE: {np.sqrt(test_loss):.4f}")
    
    # Make predictions
    predictions = model.predict(X_test[:5])
    predictions = scaler_y.inverse_transform(predictions)
    true_values = scaler_y.inverse_transform(y_test[:5])
    
    print(f"\nSample Predictions vs True Values:")
    for i in range(5):
        print(f"Pred: {predictions[i][0]:.2f}, True: {true_values[i][0]:.2f}")
    
    return model, history

def example_real_dataset():
    """Example: Real dataset - Digits classification."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Real Dataset - Digits Classification")
    print("=" * 60)
    
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Image shape: {digits.images[0].shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Normalize pixel values
    X_train = X_train / 16.0  # Pixels are in range 0-16
    X_val = X_val / 16.0
    X_test = X_test / 16.0
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of training samples: {X_train.shape[0]}")
    
    # Create neural network
    layers = create_mlp(input_size=64, hidden_sizes=[128, 64, 32], output_size=10)
    optimizer = Adam(learning_rate=0.001)
    loss = CrossEntropyLoss()
    
    model = NeuralNetwork(layers, loss, optimizer)
    model.summary()
    
    # Train the model
    print("\nTraining started...")
    start_time = time.time()
    
    history = model.fit(X_train, y_train, X_val, y_val, 
                       epochs=150, batch_size=32, verbose=True,
                       early_stopping=True, patience=15)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Analyze some misclassified examples
    predictions = model.predict_classes(X_test)
    misclassified = np.where(predictions != y_test)[0]
    
    print(f"\nMisclassified examples: {len(misclassified)}/{len(y_test)}")
    if len(misclassified) > 0:
        print("First few misclassified examples:")
        for i in misclassified[:5]:
            print(f"Index {i}: Predicted {predictions[i]}, True {y_test[i]}")
    
    return model, history

def benchmark_optimizers():
    """Compare different optimizers on the same task."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Optimizer Comparison")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=1500, n_features=20, n_classes=3, 
                              n_informative=15, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    optimizers = {
        'Adam': Adam(learning_rate=0.001),
        'SGD': SGD(learning_rate=0.01, momentum=0.9),
        'RMSprop': RMSprop(learning_rate=0.001)
    }
    
    results = {}
    
    for opt_name, optimizer in optimizers.items():
        print(f"\nTesting {opt_name}...")
        
        # Create fresh model for each optimizer
        layers = create_mlp(input_size=20, hidden_sizes=[64, 32], output_size=3)
        loss = CrossEntropyLoss()
        model = NeuralNetwork(layers, loss, optimizer)
        
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
        training_time = time.time() - start_time
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        
        results[opt_name] = {
            'final_loss': history['loss'][-1],
            'test_accuracy': test_accuracy,
            'training_time': training_time
        }
        
        print(f"{opt_name} - Final Loss: {history['loss'][-1]:.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}, Time: {training_time:.2f}s")
    
    print(f"\nOptimizer Comparison Summary:")
    print("-" * 50)
    for opt_name, metrics in results.items():
        print(f"{opt_name:10s} | Loss: {metrics['final_loss']:.4f} | "
              f"Accuracy: {metrics['test_accuracy']:.4f} | "
              f"Time: {metrics['training_time']:.2f}s")
    
    return results

def performance_benchmark():
    """Benchmark CPU performance with different network sizes."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Generate a moderately sized dataset
    X, y = make_classification(n_samples=5000, n_features=50, n_classes=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    network_configs = [
        {"name": "Small", "hidden_sizes": [32, 16]},
        {"name": "Medium", "hidden_sizes": [128, 64, 32]},
        {"name": "Large", "hidden_sizes": [256, 128, 64, 32]},
    ]
    
    results = {}
    
    for config in network_configs:
        print(f"\nTesting {config['name']} network...")
        
        layers = create_mlp(input_size=50, hidden_sizes=config['hidden_sizes'], output_size=5)
        optimizer = Adam(learning_rate=0.001)
        loss = CrossEntropyLoss()
        
        model = NeuralNetwork(layers, loss, optimizer)
        
        # Count parameters
        total_params = sum(param.size for layer in layers for param in layer.parameters.values())
        
        # Time training
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=False)
        training_time = time.time() - start_time
        
        # Time inference
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        
        results[config['name']] = {
            'parameters': total_params,
            'training_time': training_time,
            'inference_time': inference_time,
            'test_accuracy': test_accuracy,
            'throughput': len(X_test) / inference_time  # samples per second
        }
        
        print(f"Parameters: {total_params:,}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Inference time: {inference_time:.4f}s")
        print(f"Throughput: {results[config['name']]['throughput']:.0f} samples/sec")
        print(f"Test accuracy: {test_accuracy:.4f}")
    
    print(f"\nPerformance Summary:")
    print("-" * 80)
    print(f"{'Network':<10} | {'Params':<10} | {'Train(s)':<10} | {'Infer(s)':<10} | {'Samples/s':<10} | {'Accuracy':<10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<10} | {metrics['parameters']:<10,} | {metrics['training_time']:<10.2f} | "
              f"{metrics['inference_time']:<10.4f} | {metrics['throughput']:<10.0f} | {metrics['test_accuracy']:<10.4f}")
    
    return results

if __name__ == "__main__":
    print("CPU-Optimized Neural Network Examples")
    print("=====================================")
    
    # Run examples
    try:
        model1, history1 = example_classification()
        model2, history2 = example_regression()
        model3, history3 = example_real_dataset()
        
        # Benchmarks
        optimizer_results = benchmark_optimizers()
        performance_results = performance_benchmark()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Summary
        print("\nSUMMARY:")
        print("- Implemented complete CPU-optimized neural network framework")
        print("- Demonstrated multi-class classification, regression, and real dataset tasks")
        print("- Compared different optimizers (Adam, SGD, RMSprop)")
        print("- Benchmarked performance across different network sizes")
        print("- All implementations use only NumPy for CPU optimization")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()