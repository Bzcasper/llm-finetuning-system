"""CPU-Optimized Neural Network Library

A comprehensive neural network implementation optimized for CPU performance
using NumPy vectorization and efficient algorithms.
"""

__version__ = "1.0.0"
__author__ = "Neural Network Implementation Specialist"

from .layers import DenseLayer, ConvolutionalLayer, RecurrentLayer, LSTMLayer
from .activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
from .losses import MeanSquaredError, CrossEntropyLoss, BinaryCrossEntropyLoss
from .optimizers import SGD, Adam, RMSprop
from .network import NeuralNetwork
from .data_utils import DataLoader, MinMaxScaler, StandardScaler
from .trainer import Trainer

__all__ = [
    'DenseLayer', 'ConvolutionalLayer', 'RecurrentLayer', 'LSTMLayer',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'LeakyReLU',
    'MeanSquaredError', 'CrossEntropyLoss', 'BinaryCrossEntropyLoss',
    'SGD', 'Adam', 'RMSprop',
    'NeuralNetwork', 'DataLoader', 'MinMaxScaler', 'StandardScaler',
    'Trainer'
]