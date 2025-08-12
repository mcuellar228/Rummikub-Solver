import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import os
DIRECTORY = os.environ.get('DIRECTORY')

os.environ['OMP_NUM_THREADS'] = '2'

data = pd.read_csv(f"{DIRECTORY}/Model/Data/TrainingData/train20.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    W1 = np.random.randn(128, 1024) * np.sqrt(1 / 1024)
    b1 = np.random.randn(128, 1) * 0.01

    W2 = np.random.randn(64, 128) * np.sqrt(1 / 128)
    b2 = np.random.randn(64, 1) * 0.01

    W3 = np.random.randn(32, 64) * np.sqrt(1 / 64)
    b3 = np.random.randn(32, 1) * 0.01

    W4 = np.random.randn(14, 32) * np.sqrt(1 / 32)
    b4 = np.random.randn(14, 1) * 0.01

    return W1, b1, W2, b2, W3, b3, W4, b4

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)

    Z3 = W3.dot(A2) + b3
    A3 = ReLU(Z3)

    Z4 = W4.dot(A3) + b4
    A4 = softmax(Z4)

    return Z1, A1, Z2, A2, Z3, A3, Z4, A4

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, A4, W2, W3, W4, X, Y):
    one_hot_Y = one_hot(Y)
    m = X.shape[1]

    dZ4 = A4 - one_hot_Y
    dW4 = 1 / m * dZ4.dot(A3.T)
    db4 = 1 / m * np.sum(dZ4, axis=1, keepdims=True)

    dZ3 = W4.T.dot(dZ4) * ReLU_deriv(Z3)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4


def update_params(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    W4 -= alpha * dW4
    b4 -= alpha * db4
    return W1, b1, W2, b2, W3, b3, W4, b4

def get_predictions(A4):
    return np.argmax(A4, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3, W4, b4 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
        dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backward_prop(Z1, A1, Z2, A2, Z3, A3, A4, W2, W3, W4, X, Y)
        W1, b1, W2, b2, W3, b3, W4, b4 = update_params(W1, b1, W2, b2, W3, b3, W4, b4,
                                                       dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha)
        time.sleep(0.05)
        if i % 50 == 0:
            predictions = get_predictions(A4)
            print(f"Iteration {i} - Accuracy: {get_accuracy(predictions, Y):.4f}")
    return W1, b1, W2, b2, W3, b3, W4, b4

W1, b1, W2, b2, W3, b3, W4, b4 = gradient_descent(X_train, Y_train, 0.01, 2000)

def make_predictions(X, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4):
    _, _, _, _, _, _, _, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
    predictions = get_predictions(A4)
    return predictions, A4

def test_prediction(index, W1, b1, W2, b2, W3, b3, W4, b4):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3, W4, b4)
    label = Y_train[index]
    # if label in [6, 13]:
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((32, 32)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

np.save('W1.npy', W1)
np.save('b1.npy', b1)
np.save('W2.npy', W2)
np.save('b2.npy', b2)
np.save('W3.npy', W3)
np.save('b3.npy', b3)
np.save('W4.npy', W4)
np.save('b4.npy', b4)

# uncomment below to see predictions
# for i in range (30):
#     test_prediction(i, W1, b1, W2, b2, W3, b3, W4, b4)