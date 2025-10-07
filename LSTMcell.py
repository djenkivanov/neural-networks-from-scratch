import numpy as np


class LSTM:
    def __init__(self, ht, ct, xt, h_prev, c_prev):
        self.ht = ht
        self.ct = ct
        self.xt = xt
        self.h_prev = h_prev
        self.c_prev = c_prev
        
    def forget_gate(self, Wf, Uf, bf):
        self.ft = self._sigmoid(np.dot(Wf, self.xt) + np.dot(Uf, self.h_prev) + bf)
    
    def input_gate(self, Wi, Ui, bi):
        self.it = self._sigmoid(np.dot(Wi, self.xt) + np.dot(Ui, self.h_prev) + bi)
    
    def calc_candidate(self, Wc, Uc, bc):
        self.candidate = np.tanh(np.dot(Wc, self.xt) + np.dot(Uc, self.h_prev) + bc)
    
    def update_ct(self):
        self.ct = (self.ft * self.c_prev) + (self.it * self.candidate)
        
    def output_gate(self, Wo, Uo, bo):
        self.ot = self._sigmoid(np.dot(Wo, self.xt) + np.dot(Uo, self.h_prev) + bo)
        
    def hidden_update(self):
        return self.ot * np.tanh(self.ct)
            
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))