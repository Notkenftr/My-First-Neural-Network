from layers.layer import Layer
from layers.activation_layer import ActivationLayer
from layers.full_connected_layer import FCLayer
from network.network import Network
from itertools import product
import numpy as np
def relu(z):
    return np.maximum(0,z)

def relu_prime(z):
    z[z<0] = 0
    z[z>0] = 1
    return z

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1-s)

def loss_func(y_true,y_pred):
    return 0.5 * (y_pred - y_true) ** 2

def loss_func_prime(y_true,y_pred):
    return y_pred - y_true

if __name__ == '__main__':
    all_patterns = list(product([0, 1], repeat=5))

    x_train = []
    y_train = []

    for pattern in all_patterns:
        x_train.append([list(pattern)])  # giữ đúng shape (1,5)

        # rule: tổng bit >= 3 thì class 1
        label = 1 if sum(pattern) >= 3 else 0
        y_train.append([[label]])  # giữ đúng shape (1,1)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    net = Network()

    net.add(FCLayer(
        input_shape=(1,5),
        output_shape=(1,3)
    ))
    net.add(ActivationLayer(
        input_shape=(1,3),
        output_shape=(1,3),
        activation=relu,
        activation_prime=relu_prime
    ))
    net.add(FCLayer(
        input_shape=(1,3),
        output_shape=(1,2)
    ))
    net.add(ActivationLayer(
        input_shape=(1,2),
        output_shape=(1,2),
        activation=sigmoid,
        activation_prime=sigmoid_prime
    ))
    net.setup_loss(loss_func,loss_func_prime)
    net.load("weight_2.npz")
    net.fit(x_train,y_train,epochs=1,learning_rate=1,debug=True)
    #net.save("weight_2")

    print(net.predict([[1,1,1,0,0]]))