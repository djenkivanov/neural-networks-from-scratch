import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


class Conv2D:
    def __init__(self, x: np.ndarray, w: np.ndarray, b, stride=1, padding=0):
        self.x = x
        self.w = w
        self.b = b
        self.stride = stride
        self.padding = padding
        
    def conv2d_forward(self):
        self.x_pad = np.pad(self.x, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        C_in, H_in, W_in = self.x.shape
        C_out, C_in2, k_h, k_w = self.w.shape
        
        H_out = (H_in + 2*self.padding - k_h) // self.stride + 1
        W_out = (W_in + 2*self.padding - k_w) // self.stride + 1
        
        Y = np.zeros((C_out, H_out, W_out))

        for c in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    vert = h * self.stride
                    hor = w * self.stride
                    patch = self.x_pad[:, vert:vert+k_h, hor:hor+k_w]
                    Y[c, h, w] = np.sum(patch * self.w[c]) + self.b[c]
                    # print(Y[c, :, :])
                    
        return Y
    
    def conv2d_backward(self, dY):
        dx_pad = np.zeros_like(self.x_pad)
        dW = np.zeros_like(self.w)
        db = np.zeros_like(self.b)
        
        C_in, H_in, W_in = self.x.shape
        C_out, C_in2, k_h, k_w = self.w.shape
        H_out = (H_in + 2*self.padding - k_h) // self.stride + 1
        W_out = (W_in + 2*self.padding - k_w) // self.stride + 1
        
        for c in range(C_out):
            db[c] = np.sum(dY[c, :, :])
            for h in range(H_out):
                for w in range(W_out):
                    vert = h * self.stride
                    hor = w * self.stride
                    patch = self.x_pad[:, vert:vert+k_h, hor:hor+k_w]
                    dW[c] += patch * dY[c, h, w]
                    dx_pad[:, vert:vert+k_h, hor:hor+k_w] += self.w[c] * dY[c, h, w]
                    
        if self.padding > 0:
            dx = dx_pad[:, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_pad
            
        return dx, dW, db
    
    
if __name__ ==  "__main__":
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target']
    X = X.astype(np.float32) / 255.0      # Normalize to [0,1]
    y = y.astype(np.int64)
    
    mask = (y == 0) | (y == 1)
    X_small = X[mask][:100].reshape(-1, 1, 28, 28)  # shape: (100, 1, 28, 28)
    y_small = y[mask][:100]
    
    img = X_small[0]  # shape: (1, 28, 28)
    W = np.random.randn(1, 1, 3, 3) * 0.1  # 1 output channel, 1 input channel, 3x3 kernel
    b = np.zeros(1)

    conv = Conv2D(img, W, b)
    Y = conv.conv2d_forward()   # shape: (1, 26, 26)
    print("Conv output shape:", Y.shape)
    
    plt.imshow(img[0], cmap="gray")
    plt.title(f"MNIST digit {y_small[0]}")
    plt.show()
    
    # X = np.arange(16, dtype=np.float64).reshape(1, 4, 4)
    # W = np.ones((1, 1, 3, 3))
    # b = np.zeros(1)
    # conv2d_opt = Conv2D(X, W, b)
    # Y = conv2d_opt.conv2d_forward()
    # print("Y shape:", Y.shape)
    # print(Y[0])
    # dY = np.ones_like(Y)
    # dx, dW, db = conv2d_opt.conv2d_backward(dY)
    # print("dx: \n", dx)
    # print("dW: \n", dW)
    # print("db: \n", db)