import numpy as np


class RNNcell:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(1. / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1. / hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(1. / hidden_size)
        
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
    def RNN_forward(self, x_t, h_prev):
        self.x_t = x_t
        self.h_prev = h_prev
        self.h_t = np.tanh(np.dot(self.Wxh, x_t) + np.dot(self.Whh, h_prev) + self.bh)
        self.y_t = np.dot(self.Why, self.h_t) + self.by
        
        return self.h_t, self.y_t 
    
    def RNN_backward(self, dy, dh_next):
        dWhy = np.dot(dy, self.h_t.T)
        dby = dy
        
        dh = np.dot(self.Why.T, dy) + dh_next
        dtanh = (1 - self.h_t ** 2) * dh

        dWxh = np.dot(dtanh, self.x_t.T)
        dWhh = np.dot(dtanh, self.h_prev.T)
        dbh = dtanh
        
        dh_prev = np.dot(self.Whh.T, dtanh)
        
        return dWxh, dWhh, dWhy, dbh, dby, dh_prev
        