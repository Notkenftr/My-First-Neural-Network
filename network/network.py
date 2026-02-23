import numpy as np

from layers.layer import Layer
from layers.full_connected_layer import FCLayer
from layers.activation_layer import ActivationLayer
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self,layer):
        self.layers.append(layer)

    def save(self,filename):
        params = []
        for layer in self.layers:
            if hasattr(layer,"weights"):
                params.append(layer.weights)
                params.append(layer.bias)
        np.savez(filename,*params)

    def load(self,filename):
        data = np.load(filename)
        param_index = 0
        for layer in self.layers:
            if hasattr(layer, "weights"):
                layer.weights = data[f'arr_{param_index}']
                layer.bias = data[f'arr_{param_index + 1}']
                param_index += 2

    def setup_loss(self,loss,loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self,input):
        result = []
        n = len(input)

        for i in range(n):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self,x_train,y_train,epochs,learning_rate,debug=False):
        n = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(n):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j],output)
                error = self.loss_prime(y_train[j],output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error,learning_rate)
            err = err / n
            if debug:
                print(f"epoch {i+1}/{epochs}   error={err}")