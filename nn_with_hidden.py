import numpy as np

class NeuralNetwork:
    def __init__(self, n_hidden, n_in, n_out, learning_rate):
        self.lr = learning_rate
        self.W1 = np.random.randn(n_in, n_hidden)
        self.b1 = np.random.randn(n_hidden,)
        self.W2 = np.random.randn(n_hidden, n_out)
        self.b2 = np.random.randn(n_out)
        
    def predict(self, x):
        z1, a1, z2, a2 = self._forward_pass(x)
        return a2
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int) -> None:
        n_samples = x.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_vec, y_true in zip(x, y):
                dE_dW1, dE_db1, dE_dW2, dE_db2 = self._compute_gradients(x_vec, y_true)
                self._update_parameters(dE_dW1, dE_db1, dE_dW2, dE_db2)
                y_pred = self.predict(x_vec)
                epoch_loss += self._binary_cross_entropy(y_pred, y_true)

            # print(f"Epoch {epoch} Loss: {epoch_loss / n_samples}")
    
    def relu(self, z):
        return np.max(0, z)

    def relu_d(self, z):
        return (z > 0).astype(float)
    
    def leaky_relu(self, z, a=0.01):
        return np.where(z > 0, z, a*z)
    
    def leaky_relu_derivative(self, z, a=0.01):
        grad = np.ones_like(z)
                
    
    def _compute_gradients(self, x, y_true):
        z1, a1, z2, a2 = self._forward_pass(x)
        dE_dz2 = a2 - y_true
        dE_db2 = dE_dz2
        dE_dW2 = np.outer(a1, dE_dz2)
        
        dE_da1 = np.dot(dE_dz2, self.W2.T)
        dE_dz1 = dE_da1 * (self._sigmoid(z1) * (1 - self._sigmoid(z1)))
        dE_db1 = dE_dz1
        dE_dW1 = np.outer(x, dE_dz1)
        # print(self._mse(y_pred, y_true))
        return dE_dW1, dE_db1, dE_dW2 , dE_db2
    
    def _update_parameters(self, dE_dW1, dE_db1, dE_dW2, dE_db2):
        self.W1 -= self.lr * dE_dW1
        self.b1 -= self.lr * dE_db1
        self.W2 -= self.lr * dE_dW2
        self.b2 -= self.lr * dE_db2

    def _mse(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true)**2)   

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _forward_pass(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)
        return z1, a1, z2, a2

    def _binary_cross_entropy(self, y_pred, y_true):
        return -np.sum(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))   
        return exp_z / exp_z.sum(axis=-1, keepdims=True)

    def _cross_entropy_loss(self, probs, y_true):
        return -np.sum(y_true * np.log(probs))
    

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
    
    nn = NeuralNetwork(4, 2, 2, 0.1)
    predict = nn.predict(inputs_ex)
    print("First predict: \n", predict)
    nn.train(inputs_ex, predict_true, 1000)
    predict = nn.predict(inputs_ex)
    print("After train predict: \n", predict)

