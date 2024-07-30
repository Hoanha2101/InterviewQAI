import numpy as np
import pickle
from utils import *

class SimpleNN:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W1 = np.random.randn(input_dim, 128) * 0.01
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2
            }, f)
            
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            model = SimpleNN(data['input_dim'], data['output_dim'])
            model.W1 = data['W1']
            model.b1 = data['b1']
            model.W2 = data['W2']
            model.b2 = data['b2']
            return model