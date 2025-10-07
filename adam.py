import numpy as np


class Adam:
    def __init__(self, weights, biases, a=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.weights = weights
        self.biases = biases
        self.a = a
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0
        
    
    def update(self, dW, dB):
        # print("Before update: \n", self.weights)
        self.t += 1
        self.m_w = self.beta1*self.m_w + (1 - self.beta1) * dW
        self.v_w = self.beta2*self.v_w + (1 - self.beta2) * (dW**2)
        self.m_b = self.beta1*self.m_b + (1 - self.beta1) * dB
        self.v_b = self.beta2*self.v_b + (1 - self.beta2) * (dB**2)
        
        m_hat_denom = 1 - self.beta1**self.t
        v_hat_denom = 1 - self.beta2**self.t
        
        m_w_hat = self.m_w / m_hat_denom
        v_w_hat = self.v_w / v_hat_denom
        m_b_hat = self.m_b / m_hat_denom
        v_b_hat = self.v_b / v_hat_denom
        
        step_w = self.a * (m_w_hat / (np.sqrt(v_w_hat) + self.eps))
        step_b = self.a * (m_b_hat / (np.sqrt(v_b_hat) + self.eps))
        
        self.weights -= step_w
        self.biases -= step_b
        
        # print("After update: \n", self.weights)
        
