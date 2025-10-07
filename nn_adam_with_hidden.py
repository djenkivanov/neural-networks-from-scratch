import numpy as np
from adam import Adam as Adam
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, n_in, n_out, n_hidden, learning_rate):
        self.a = learning_rate
        self.W1 = np.random.randn(n_in, n_hidden) * 0.1
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_out) * 0.1
        self.b2 = np.zeros(n_out)
        self.opt1 = Adam(self.W1, self.b1, self.a)
        self.opt2 = Adam(self.W2, self.b2, self.a)
        self.loss_history = []
        
    def predict(self, x):
        z1, a1, z2, a2 = self._forward_pass(x)
        return a2
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int) -> None:
        n_samples = x.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_vec, y_true in zip(x, y):
                dW1, db1, dW2, db2 = self._backprop(x_vec, y_true)
                self.opt1.update(dW1, db1)
                self.opt2.update(dW2, db2)
                y_pred = self.predict(x_vec)
                epoch_loss += self._mse(y_pred, y_true)

            self.loss_history.append(epoch_loss / n_samples)
        
    
    def _forward_pass(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)
        return z1, a1, z2, a2
    
    def _backprop(self, x, y_true):
        z1, a1, z2, a2 = self._forward_pass(x)
        dz2 = (a2 - y_true) * (a2 * (1 - a2))
        db2 = dz2
        dW2 = np.outer(a1, dz2)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (a1 * (1 - a1))
        db1 = dz1
        dW1 = np.outer(x, dz1)
        return dW1, db1, dW2, db2
        
    
    def _update_parameters(self, dE_dw, dE_db):
        self.weights -= self.a * dE_dw
        self.biases -= self.a * dE_db

    def _mse(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true)**2)   

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

