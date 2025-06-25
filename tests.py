#!/usr/bin/env python3
import numpy as np
import unittest
from unittest.mock import patch
import warnings
warnings.filterwarnings('ignore')

from neural_network import Dense, Conv2D, BatchNormalization, MaxPool2D
from network import NeuralNetwork, create_mlp
from optimizers import Adam, SGD, RMSprop
from losses import CrossEntropyLoss, MeanSquaredError, BinaryCrossEntropyLoss

class TestLayers(unittest.TestCase):
    """Test individual layer implementations."""
    
    def setUp(self):
        np.random.seed(42)
        
    def test_dense_layer_forward(self):
        """Test Dense layer forward pass."""
        layer = Dense(3, 2, activation='relu')
        inputs = np.array([[1, 2, 3], [4, 5, 6]])
        
        output = layer.forward(inputs)
        
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(np.all(output >= 0))  # ReLU should give non-negative outputs
        
    def test_dense_layer_backward(self):
        """Test Dense layer backward pass."""
        layer = Dense(3, 2, activation='relu')
        inputs = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Forward pass
        output = layer.forward(inputs)
        
        # Backward pass
        grad_output = np.ones_like(output)
        grad_input = layer.backward(grad_output)
        
        self.assertEqual(grad_input.shape, inputs.shape)
        self.assertIn('W', layer.gradients)
        self.assertIn('b', layer.gradients)
        
    def test_dense_layer_activations(self):
        """Test different activation functions in Dense layer."""
        layer_relu = Dense(2, 2, activation='relu')
        layer_sigmoid = Dense(2, 2, activation='sigmoid')
        layer_tanh = Dense(2, 2, activation='tanh')
        
        inputs = np.array([[-1, 1], [2, -2]])
        
        # Test ReLU
        output_relu = layer_relu.forward(inputs)
        self.assertTrue(np.all(output_relu >= 0))
        
        # Test Sigmoid
        output_sigmoid = layer_sigmoid.forward(inputs)
        self.assertTrue(np.all((output_sigmoid >= 0) & (output_sigmoid <= 1)))
        
        # Test Tanh
        output_tanh = layer_tanh.forward(inputs)
        self.assertTrue(np.all((output_tanh >= -1) & (output_tanh <= 1)))
        
    def test_batch_normalization(self):
        """Test Batch Normalization layer."""
        layer = BatchNormalization(3)
        inputs = np.random.randn(10, 3) * 5 + 10  # Mean=10, std=5
        
        # Training mode
        layer.set_training(True)
        output = layer.forward(inputs)
        
        # Check normalization
        output_mean = np.mean(output, axis=0)
        output_std = np.std(output, axis=0)
        
        np.testing.assert_array_almost_equal(output_mean, 0, decimal=5)
        np.testing.assert_array_almost_equal(output_std, 1, decimal=5)
        
    def test_conv2d_layer(self):
        """Test Conv2D layer."""
        layer = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        inputs = np.random.randn(2, 3, 32, 32)  # Batch of 2, 3 channels, 32x32 images
        
        output = layer.forward(inputs)
        
        # Check output shape
        expected_shape = (2, 16, 32, 32)  # Same spatial size due to padding=1
        self.assertEqual(output.shape, expected_shape)
        
    def test_maxpool2d_layer(self):
        """Test MaxPool2D layer."""
        layer = MaxPool2D(pool_size=2, stride=2)
        inputs = np.random.randn(2, 3, 8, 8)
        
        output = layer.forward(inputs)
        
        # Check output shape
        expected_shape = (2, 3, 4, 4)  # Halved spatial dimensions
        self.assertEqual(output.shape, expected_shape)

class TestOptimizers(unittest.TestCase):
    """Test optimizer implementations."""
    
    def setUp(self):
        np.random.seed(42)
        self.layer = Dense(3, 2)
        self.gradients = {
            'W': np.ones((3, 2)),
            'b': np.ones((1, 2))
        }
        
    def test_sgd_optimizer(self):
        """Test SGD optimizer."""
        optimizer = SGD(learning_rate=0.1)
        initial_weights = self.layer.parameters['W'].copy()
        
        optimizer.update(self.layer, self.gradients)
        
        # Check if weights were updated
        self.assertFalse(np.array_equal(initial_weights, self.layer.parameters['W']))
        
    def test_adam_optimizer(self):
        """Test Adam optimizer."""
        optimizer = Adam(learning_rate=0.001)
        initial_weights = self.layer.parameters['W'].copy()
        
        optimizer.update(self.layer, self.gradients)
        
        # Check if weights were updated
        self.assertFalse(np.array_equal(initial_weights, self.layer.parameters['W']))
        
    def test_rmsprop_optimizer(self):
        """Test RMSprop optimizer."""
        optimizer = RMSprop(learning_rate=0.001)
        initial_weights = self.layer.parameters['W'].copy()
        
        optimizer.update(self.layer, self.gradients)
        
        # Check if weights were updated
        self.assertFalse(np.array_equal(initial_weights, self.layer.parameters['W']))

class TestLossFunctions(unittest.TestCase):
    """Test loss function implementations."""
    
    def test_mean_squared_error(self):
        """Test MSE loss function."""
        loss_fn = MeanSquaredError()
        predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([[1.5, 2.5], [2.5, 3.5]])
        
        loss = loss_fn.forward(predictions, targets)
        grad = loss_fn.backward(predictions, targets)
        
        self.assertIsInstance(loss, float)
        self.assertEqual(grad.shape, predictions.shape)
        
    def test_cross_entropy_loss(self):
        """Test Cross-entropy loss function."""
        loss_fn = CrossEntropyLoss()
        predictions = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        targets = np.array([0, 2])  # Class indices
        
        loss = loss_fn.forward(predictions, targets)
        grad = loss_fn.backward(predictions, targets)
        
        self.assertIsInstance(loss, float)
        self.assertEqual(grad.shape, predictions.shape)
        
    def test_binary_cross_entropy_loss(self):
        """Test Binary cross-entropy loss function."""
        loss_fn = BinaryCrossEntropyLoss()
        predictions = np.array([[0.8], [0.3], [0.9]])
        targets = np.array([[1], [0], [1]])
        
        loss = loss_fn.forward(predictions, targets)
        grad = loss_fn.backward(predictions, targets)
        
        self.assertIsInstance(loss, float)
        self.assertEqual(grad.shape, predictions.shape)

class TestNeuralNetwork(unittest.TestCase):
    """Test the main NeuralNetwork class."""
    
    def setUp(self):
        np.random.seed(42)
        
    def test_network_creation(self):
        """Test neural network creation."""
        layers = create_mlp(input_size=4, hidden_sizes=[8, 4], output_size=2)
        optimizer = Adam()
        loss = CrossEntropyLoss()
        
        network = NeuralNetwork(layers, loss, optimizer)
        
        self.assertEqual(len(network.layers), 3)  # 2 hidden + 1 output
        
    def test_network_forward_pass(self):
        """Test forward pass through network."""
        layers = create_mlp(input_size=4, hidden_sizes=[8], output_size=2)
        optimizer = Adam()
        loss = CrossEntropyLoss()
        
        network = NeuralNetwork(layers, loss, optimizer)
        inputs = np.random.randn(5, 4)
        
        output = network.forward(inputs)
        
        self.assertEqual(output.shape, (5, 2))
        
    def test_network_training_step(self):
        """Test a single training step."""
        layers = create_mlp(input_size=4, hidden_sizes=[8], output_size=2)
        optimizer = Adam()
        loss = CrossEntropyLoss()
        
        network = NeuralNetwork(layers, loss, optimizer)
        X = np.random.randn(10, 4)
        y = np.random.randint(0, 2, 10)
        
        initial_loss, initial_accuracy = network.evaluate(X, y)
        train_loss, train_accuracy = network.train_step(X, y)
        
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_accuracy, float)
        
    def test_network_fit(self):
        """Test network training with fit method."""
        layers = create_mlp(input_size=4, hidden_sizes=[8], output_size=2)
        optimizer = Adam()
        loss = CrossEntropyLoss()
        
        network = NeuralNetwork(layers, loss, optimizer)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        
        history = network.fit(X, y, epochs=5, batch_size=16, verbose=False)
        
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
        self.assertEqual(len(history['loss']), 5)
        
    def test_network_predict(self):
        """Test network prediction."""
        layers = create_mlp(input_size=4, hidden_sizes=[8], output_size=3)
        optimizer = Adam()
        loss = CrossEntropyLoss()
        
        network = NeuralNetwork(layers, loss, optimizer)
        X = np.random.randn(10, 4)
        
        predictions = network.predict(X)
        classes = network.predict_classes(X)
        
        self.assertEqual(predictions.shape, (10, 3))
        self.assertEqual(classes.shape, (10,))
        self.assertTrue(np.all(classes >= 0) and np.all(classes <= 2))

class TestGradientChecking(unittest.TestCase):
    """Test gradient computation accuracy using numerical gradients."""
    
    def numerical_gradient(self, f, x, h=1e-5):
        """Compute numerical gradient using finite differences."""
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            old_value = x[idx]
            
            x[idx] = old_value + h
            fxh_pos = f()
            
            x[idx] = old_value - h
            fxh_neg = f()
            
            grad[idx] = (fxh_pos - fxh_neg) / (2 * h)
            x[idx] = old_value
            it.iternext()
            
        return grad
        
    def test_dense_layer_gradients(self):
        """Test Dense layer gradient computation."""
        np.random.seed(42)
        layer = Dense(3, 2, activation='linear')  # Use linear for simpler gradient checking
        inputs = np.random.randn(2, 3)
        grad_output = np.random.randn(2, 2)
        
        # Forward pass
        output = layer.forward(inputs)
        
        # Analytical gradients
        analytical_grad = layer.backward(grad_output)
        
        # Numerical gradient for input
        def f():
            return np.sum(layer.forward(inputs) * grad_output)
            
        numerical_grad = self.numerical_gradient(f, inputs)
        
        # Compare gradients (allow some numerical error)
        np.testing.assert_array_almost_equal(analytical_grad, numerical_grad, decimal=4)

class TestComplexScenarios(unittest.TestCase):
    """Test complex scenarios and edge cases."""
    
    def test_xor_problem(self):
        """Test learning the XOR problem (classic non-linear problem)."""
        # XOR dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0])  # XOR outputs
        
        # Create network capable of learning XOR
        layers = create_mlp(input_size=2, hidden_sizes=[4, 4], output_size=2)
        optimizer = Adam(learning_rate=0.01)
        loss = CrossEntropyLoss()
        
        network = NeuralNetwork(layers, loss, optimizer)
        
        # Train for many epochs
        history = network.fit(X, y, epochs=500, batch_size=4, verbose=False)
        
        # Test predictions
        predictions = network.predict_classes(X)
        accuracy = np.mean(predictions == y)
        
        # Should achieve high accuracy on this simple problem
        self.assertGreater(accuracy, 0.9, f"XOR accuracy was {accuracy}, expected > 0.9")
        
    def test_overfitting_detection(self):
        """Test that the network can overfit to small datasets."""
        # Small dataset with complex pattern
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, 20)
        
        # Large network relative to data
        layers = create_mlp(input_size=10, hidden_sizes=[50, 50], output_size=3)
        optimizer = Adam(learning_rate=0.001)
        loss = CrossEntropyLoss()
        
        network = NeuralNetwork(layers, loss, optimizer)
        
        # Train for many epochs
        history = network.fit(X, y, epochs=100, batch_size=20, verbose=False)
        
        # Should achieve very low training loss (overfitting)
        final_loss = history['loss'][-1]
        self.assertLess(final_loss, 0.1, f"Expected overfitting with loss < 0.1, got {final_loss}")
        
    def test_batch_size_variations(self):
        """Test different batch sizes."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        batch_sizes = [1, 10, 32, 100]
        results = {}
        
        for batch_size in batch_sizes:
            layers = create_mlp(input_size=5, hidden_sizes=[16], output_size=2)
            optimizer = Adam()
            loss = CrossEntropyLoss()
            
            network = NeuralNetwork(layers, loss, optimizer)
            history = network.fit(X, y, epochs=10, batch_size=batch_size, verbose=False)
            
            results[batch_size] = history['loss'][-1]
            
        # All batch sizes should work without errors
        self.assertEqual(len(results), len(batch_sizes))
        
    def test_network_save_load(self):
        """Test saving and loading networks."""
        layers = create_mlp(input_size=4, hidden_sizes=[8], output_size=2)
        optimizer = Adam()
        loss = CrossEntropyLoss()
        
        network = NeuralNetwork(layers, loss, optimizer)
        
        # Train briefly
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50)
        network.fit(X, y, epochs=5, verbose=False)
        
        # Make predictions before saving
        test_X = np.random.randn(10, 4)
        predictions_before = network.predict(test_X)
        
        # Save and load
        network.save('test_model.pkl')
        loaded_network = NeuralNetwork.load('test_model.pkl')
        
        # Make predictions after loading
        predictions_after = loaded_network.predict(test_X)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(predictions_before, predictions_after)
        
        # Clean up
        import os
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')

def run_performance_test():
    """Run performance benchmarks to ensure reasonable speed."""
    print("Running performance tests...")
    
    # Create a moderately sized problem
    X = np.random.randn(1000, 50)
    y = np.random.randint(0, 5, 1000)
    
    layers = create_mlp(input_size=50, hidden_sizes=[128, 64], output_size=5)
    optimizer = Adam()
    loss = CrossEntropyLoss()
    
    network = NeuralNetwork(layers, loss, optimizer)
    
    import time
    
    # Time training
    start_time = time.time()
    network.fit(X, y, epochs=10, batch_size=32, verbose=False)
    training_time = time.time() - start_time
    
    # Time inference
    start_time = time.time()
    predictions = network.predict(X)
    inference_time = time.time() - start_time
    
    print(f"Training time (10 epochs): {training_time:.2f}s")
    print(f"Inference time (1000 samples): {inference_time:.4f}s")
    print(f"Inference throughput: {len(X) / inference_time:.0f} samples/sec")
    
    # Basic performance expectations for CPU
    assert training_time < 60, f"Training too slow: {training_time:.2f}s"
    assert inference_time < 1, f"Inference too slow: {inference_time:.4f}s"
    
    print("Performance tests passed!")

if __name__ == '__main__':
    print("Running Neural Network Tests")
    print("=" * 50)
    
    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    if result.wasSuccessful():
        print("\n" + "=" * 50)
        print("ALL UNIT TESTS PASSED!")
        
        # Run performance test
        print("\n" + "=" * 50)
        run_performance_test()
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Neural network implementation is working correctly.")
    else:
        print("\n" + "=" * 50)
        print("SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")