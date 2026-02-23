from abc import abstractmethod
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None
        raise NotImplementedError
    @abstractmethod
    def input(self,input):
        return self.input
    @abstractmethod
    def output(self,output):
        return self.output
    @abstractmethod
    def input_shape(self,input_shape):
        return self.input_shape
    @abstractmethod
    def output_shape(self,output_shape):
        return self.output_shape
    @abstractmethod
    def forward_propagation(self,input):
        raise NotImplementedError
    @abstractmethod
    def backward_propagation(self,output_error,learning_rate):
        raise NotImplementedError