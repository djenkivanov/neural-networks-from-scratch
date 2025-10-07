import numpy as np


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        self.W_q = np.random.randn(embed_dim, num_heads*self.head_dim)
        self.W_k = np.random.randn(embed_dim, num_heads*self.head_dim)
        self.W_v = np.random.randn(embed_dim, num_heads*self.head_dim)
        
        self.W_o = np.random.randn(num_heads * self.head_dim, embed_dim)
        
    def forward(self, X:np.ndarray):
        seq_len = X.shape[0]
        
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        # Q =         [num_heads, seq_len, head_dim]
        # we want K = [num_heads, head_dim, seq_len]
        attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(K.shape[-1])
        attention_weights = self.softmax(attention_scores)
        attention_output = np.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 0, 2).reshape(seq_len, self.num_heads*self.head_dim)
        
        return np.dot(attention_output, self.W_o) 
        
    def softmax(self, X):
        x_exp = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return x_exp/np.sum(x_exp, axis=-1, keepdims=True)