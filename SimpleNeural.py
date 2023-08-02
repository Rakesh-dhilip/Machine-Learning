import numpy 

def mean_square_error(y,y_hat):
    return numpy.mean(numpy.power(y - y_hat, 2))

def mean_square_error_prime(y_true,y_pred):
    # J_hat = 2/n (y-y_hat)
    # n = numpy.size(y_hat)
    return 2 * (y_pred - y_true) / numpy.size(y_true)

class Tanh():
    def tanh(self,x):
        return numpy.tanh(x)

    def tanh_prime(self,x):
        return 1 - numpy.tanh(x) ** 2

    def __init__(self):  
        pass

    def forward(self,inputs):
        self.inputs = inputs
        return self.tanh(self.inputs)
    
    def backward(self,output,lr):
        # print(output,self.inputs.T)
        dZ = numpy.multiply(output,self.tanh_prime(self.inputs))
        return dZ


class Relu:
    def __init__(self):  
        pass
    
    def _prime_function(self,inputs):
        return numpy.where(inputs>0,1,0)

    def forward(self,inputs):
        self.inputs = inputs
        return numpy.maximum(0,inputs)
    
    def backward(self,output,lr):
        # print(output,self.inputs.T)
        dZ = numpy.multiply(output,self._prime_function(self.inputs))
        return dZ
    
class Sigmoid():

    def __init__(self):
        pass
    
    def _sigmoid(self,x):
        return 1 / (1 + numpy.exp(-x))

    def _sigmoid_prime(self,x):
        s = self._sigmoid(x)
        return s * (1 - s)

    def forward(self,inputs):
        self.inputs = inputs
        return self._sigmoid(self.inputs)
    
    def backward(self,output,lr):
        dZ = numpy.multiply(output,self._sigmoid_prime(self.inputs))
        return dZ

    
class Layer:                   

    def __init__(self,input,output):
        self.weights = 0.1*numpy.random.randn(output, input)
        self.bias = numpy.random.randn(output,1)
    
    def forward(self,inputs):
        self.inputs = inputs
        # print(self.weights.shape,inputs.shape,self.bias.shape)
        return numpy.dot(self.weights,self.inputs) + self.bias
        
    def backward(self,output,lr):
        # print("--> W",self.weights)
        # print("--> B",self.bias)
        dZ = output
        # print(dZ,self.inputs.T)
        dW = numpy.dot(dZ ,self.inputs.T)
        # print("dW",dW)
        db = dZ
        # weights update
        self.weights = self.weights - lr*dW
        self.bias = self.bias - lr*db
        dA = numpy.dot(self.weights.T,dZ)
        return dA


    
network = [
    Layer(2,2),
]

X = numpy.reshape([[1, 1]], (1, 2, 1))
Y = numpy.reshape([[1,0]], (1, 2, 1))
lr=0.1
 



for i in range(1,3):
    error = 0
    for x , y in zip(X,Y): 
        input = x
        # print(input)
        # exit()
        for layer in network:
            input = layer.forward(input)
        y_hat = input
        # print(mean_square_error(y,y_hat))
        error = error + mean_square_error(y,y_hat)
        # print(mean_square_error(y,y_hat))
        dA = mean_square_error_prime(y,y_hat)
        for layer in reversed(network):
            dA = layer.backward(dA,lr)
    # exit(error/4)
    print(error/len(X),i)

input = [[1],[2]]
for layer in network:
    input = layer.forward(input)

print("-->", input)





# import numpy
# print(numpy.asarray([1,2,3]).shape)
# # print(numpy.dot(,numpy.asarray(1) ))
# 
