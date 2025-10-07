from matplotlib import pyplot as plt
import numpy as np
from nn_adam_with_hidden import NeuralNetwork


if __name__ == "__main__":
    inputs_ex = np.array([
        [0,0],
        [0,1],
        [1,1],
        [1,0],
    ])
    predict_true = np.array([
        [0,0],
        [1,0],
        [0,1],
        [1,1],
    ])
    
    nn = NeuralNetwork(2, 2, 4, 0.1)
    # nn.gradient_check(inputs_ex[2], predict_true[2])
    predict = nn.predict(inputs_ex)
    print("First predict: \n", predict)
    print("Training started")
    nn.train(inputs_ex, predict_true, 500)
    print("Training ended")
    predict = nn.predict(inputs_ex)
    print("After train predict: \n", predict)
    # print("Loss history: ", nn.loss_history)
    
    plt.plot(nn.loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss over Epochs")
    plt.show()