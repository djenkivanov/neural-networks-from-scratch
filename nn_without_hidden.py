import numpy as np
from adam import Adam as Adam
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, n_in, n_out, learning_rate, optimizer):
        self.lr = learning_rate
        self.weights = np.random.randn(n_in, n_out)
        self.biases = np.random.randn(n_out,)
        self.optimizer = optimizer
        self.loss_history = []
        
    def predict(self, x):
        z = np.dot(x, self.weights) + self.biases
        return self._sigmoid(z)
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int) -> None:
        n_samples = x.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_vec, y_true in zip(x, y):
                dE_dw, dE_db = self._compute_gradients(x_vec, y_true)
                self.optimizer.update(dE_dw, dE_db)
                y_pred = self.predict(x_vec)
                epoch_loss += self._mse(y_pred, y_true)

            self.loss_history.append(epoch_loss / n_samples)
    
    def gradient_check(self, x, y_true, epsilon=0.00001):
        og_weights = self.weights.copy()
        og_biases = self.biases.copy()
        numeric_grads = []
        for i, _ in np.ndenumerate(self.weights):
            orig_w = self.weights[i]    
                     
            self.weights[i] = orig_w + epsilon
            e_pred_plus = self.predict(x)
            loss_plus = self._mse(e_pred_plus, y_true)
            # print("1: ", loss_plus)
            
            self.weights[i] = orig_w - epsilon
            e_pred_minus = self.predict(x)
            loss_minus = self._mse(e_pred_minus, y_true)
            # print("2: ", loss_minus)
            
            numeric_grads.append((loss_plus - loss_minus) / (2*epsilon))
            
            self.weights[i] = orig_w       
            

        for i, _ in np.ndenumerate(self.biases):
            orig_b = self.biases[i]    
                     
            self.biases[i] = orig_b + epsilon
            e_pred_plus = self.predict(x)
            loss_plus = self._mse(e_pred_plus, y_true)
            # print("1: ", loss_plus)
            
            self.biases[i] = orig_b - epsilon
            e_pred_minus = self.predict(x)
            loss_minus = self._mse(e_pred_minus, y_true)
            # print("2: ", loss_minus)
            
            numeric_grads.append((loss_plus - loss_minus) / (2*epsilon))

            self.biases[i] = orig_b 
            
        dW, dB = self._compute_gradients(x, y_true)
        analytic_grads = list(dW.flatten()) + list(dB.flatten())
            
        for i, (num, ana) in enumerate(zip(numeric_grads, analytic_grads)):
            rel_err = abs(num - ana) / (abs(num) + abs(ana))
            print(f"param {i:03d}: num={num:.6e}, ana={ana:.6e}, rel_error={rel_err:.2e}")
            
        self.weights = og_weights
        self.biases = og_biases
 
        
    
    def _compute_gradients(self, x, y_true):
        y_pred = self.predict(x)
        z = np.dot(x, self.weights) + self.biases
        dE_dz = (y_pred - y_true) * (self._sigmoid(z) * (1 - self._sigmoid(z))) 
        der_b = dE_dz
        der_w = np.outer(x, dE_dz)
        # print(self._mse(y_pred, y_true))
        return der_w, der_b
    
    def _update_parameters(self, dE_dw, dE_db):
        self.weights -= self.lr * dE_dw
        self.biases -= self.lr * dE_db

    def _mse(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true)**2)   

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

if __name__ == "__main__":   
    inputs_ex = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
    ])
    predict_true = np.array([
        [0,0],
        [1,0],
        [1,0],
        [0,1],
    ])
    
    nn = NeuralNetwork(2, 2, 0.1)
    # nn.gradient_check(inputs_ex[2], predict_true[2])
    predict = nn.predict(inputs_ex)
    print("First predict: ", predict)
    print("Training started\n")
    nn.train(inputs_ex, predict_true, 1000)
    print("Training ended\n")
    predict = nn.predict(inputs_ex)
    print("After train predict: ", predict)
    # print("Loss history: ", nn.loss_history)
    
    plt.plot(nn.loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss over Epochs")
    plt.show()