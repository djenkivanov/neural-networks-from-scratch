import numpy as np


class Attention:
    def __init__(self):
        pass
    
    def attention_out(self, Q:np.ndarray, K:np.ndarray, V):
        QKT = np.dot(Q, K.T)
        return np.dot(self.softmax(QKT/np.sqrt(K.shape[-1])), V)
        
    def softmax(self, x):
        x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return x_exp/np.sum(x_exp, axis=-1, keepdims=True)