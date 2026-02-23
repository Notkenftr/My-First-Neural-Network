from .layer import Layer
import numpy as np


class FCLayer(Layer):
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(self.input_shape[1],self.output_shape[1]) - 0.5
        self.bias = np.random.rand(1,self.output_shape[1]) - 0.5

    def forward_propagation(self,input):
        """
        [0,1,0]
           x
        [1,0,1,1]
        ma trận 1x3 có thể nhân với ma trận 1x4 để cho ra ma trận 1x4
        """
        self.input = input
        self.output = np.dot(self.input,self.weights) + self.bias

        return self.output

    def backward_propagation(self,output_error,learning_rate):
        current_layer_err = np.dot(output_error,self.weights.T) # lỗi của lớp trc
        dweight = np.dot(self.input.T,output_error) # đạo hàm của trọng số

        self.weights -= dweight * learning_rate
        self.bias -= learning_rate * output_error

        return current_layer_err