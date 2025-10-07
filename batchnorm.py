import numpy as np


class Batch:    
    def batchnorm_forward(self, X: np.ndarray, gamma, beta, eps=1e-5):
        batch_size = X.shape[0]
        batch_mean = np.mean(X, axis=0)
        variance = np.var(X, axis=0)
            
        batch_norm = (X - batch_mean) / (np.sqrt(variance + eps))
        
        out = gamma * batch_norm + beta
        
        cache = {
            "mean": batch_mean,
            "var": variance,
            "Xhat": batch_norm,
            "bsize": batch_size,
            "gamma": gamma,
            "beta": beta,
            "eps": eps
        }
        
        return out, cache
    
    def dropout_forward(self, X: np.ndarray, p: float):
        mask = (np.random.rand(*X.shape) > p)
        return X * mask / (1 - p)
    
    def batchnorm_backprop(self, dY, cache):
        mean, var, Xhat, m, gamma, beta, eps = cache['mean'], cache['var'], cache['Xhat'], cache['bsize'], cache['gamma'], cache['beta'], cache['eps']
        
        dbeta = np.sum(dY, axis=0)
        dgamma = np.sum(dY * Xhat, axis=0)

        dX = (1 / m) * gamma * (1 / np.sqrt(var + eps)) * (
            m * dY - 
            np.sum(dY, axis=0) - 
            Xhat * np.sum(dY * Xhat, axis=0)
        )

        return dX, dgamma, dbeta