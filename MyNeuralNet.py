import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math

# -------------------------------------Auxilliary functions--------------------------------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1-sigmoid(x))
        
def relu(x):
    result =  x if x > 0 else 0
    return result

def relu_d(x):
    result = 1 if x >= 0 else 0
    return result

# Takes ints and np arrays from all dimensions
def callOnArray(x, function):    
    if (isinstance(x, np.ndarray)):
        if (x.ndim == 1):
            return np.array(list(map(function,x)))
        else:
            return np.array([callOnArray(ai, function) for ai in x]) # if it has multiple dimensions
    if isinstance(x, int):
        return function(x)


    
# -------------------------------------Class for one layer neural network--------------------------------------
class Layer:
    def __init__(self, input_size, neurons, learning_rate = 0.2, activationFunction = sigmoid, activationFunctionD = sigmoid_d):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.neurons = neurons
        self.weights = np.random.rand(neurons, input_size)*2 -1 # assign random values between -1 and 1
        self.biases = np.array([np.random.rand(neurons)]).T *2 -1 # make the array in an array a column vector
        
        self.activationFunction = activationFunction
        self.activationFunctionD = activationFunctionD        
        
    def calc(self, x):
        assert(len(x[0]) == self.weights.shape[1])         
        result = callOnArray(np.dot(self.weights, x.T) + self.biases, self.activationFunction).T
        return result
    
    def cost(self, x, y):
        y_pred = self.calc(x)
        return (y_pred - y) * ( y_pred - y)
    
    # This is the normal learning function for a one layer network
    def learnLastLayer(self,x, y):
        y_pred = self.calc(x)
        Dcost_Da = y_pred.T - y.T
        self.learn(x, Dcost_Da)
        return
    
    # The learning algorithm is expressed with the following terms:
    #    a: output of a neuron, z: input of a neuron
    #    Dcost_Da: partial derivative of the cost with respect to the output of the neurons
    #    Da_Dz   : partial derivative of the activationfunction with respect to the input of the neurons
    #    Dcost_Dz: partial derivative of the cost with respect to the input of the neurons
    #              -> also referred to as "Deltas" when used in backpropagation
    def learn(self, x, Dcost_Da):
        assert(len(x)>0)       
        Da_Dz = callOnArray(np.dot(self.weights, x.T) + self.biases, self.activationFunctionD)      
        self.deltas = Dcost_Da * Da_Dz  # save the deltas for the next layer
        Dcost_Dz = -self.deltas
        
        # Calculate the delta weights for every neuron, for every datapoint
        delta_w = np.array( [delta*p for n in Dcost_Dz for delta, p in zip(n,x) ])
        # Sum up the delta_w for all the datapoints to get the total delta_weight per neuron
        delta_weights = np.sum(delta_w.reshape(self.neurons,len(x),len(x[0])), axis = 1) * self.learning_rate # aantal neurons (len(self.weights, aantal datapoints, aantal dimensies 
        
        # similar for the biases
        delta_biases = np.sum(Dcost_Dz.reshape(self.neurons,len(x),1), axis = 1) * self.learning_rate
     
        self.weights += delta_weights
        self.biases  += delta_biases        
        return
    
    
    # The below functions are necessary for a multilayer network 
    def learnLayer(self, x, delta_next_layer, weights_next_layer):
        Dcost_Da = np.dot(delta_next_layer.T, weights_next_layer).T
        self.learn(x, Dcost_Da)
        return
    
    # Should only be called if the layer has learnt and calculated the deltas for the 
    def getDeltas(self):
        return self.deltas
    
    def getWeights(self):
        return self.weights
        
        
    # For debugging purposes, prints weights and biases
    def show(self):
        string = ""
        biases = " "        
        for i in range(self.neurons):
            string +="[ "            
            for w in self.weights[i]:
                string+= "{0:4f} ".format(w)
                biases += "     "
            string += "]    "
            biases += "{0:4f}       ".format(self.biases[i][0])
        print(string)
        print(biases)  
        return
        
        
        
# -------------------------------------Multiple layers are combined to make a deeper network--------------------------
class NeuralNet:
    def __init__(self, n_input, n_output, hidden = [5], learning_rate = 0.1, activationFunction = sigmoid, activationFunctionD = sigmoid_d):
        self.activationFunction = activationFunction
        self.activationFunctionD = activationFunctionD
        self.learning_rate = learning_rate
        
        # Initiate layers
        self.layers = []
        for n in hidden:
            self.layers.append(Layer(n_input,n, learning_rate = self.learning_rate, activationFunction = self.activationFunction, activationFunctionD = self.activationFunctionD))
            n_input = n
        self.layers.append(Layer(n_input, n_output, learning_rate = self.learning_rate, activationFunction = self.activationFunction, activationFunctionD = self.activationFunctionD))
        
    def calc(self, x):
        for layer in self.layers:
            x = layer.calc(x)
        return x
    
    def cost(self, x, y):
        y_pred = self.calc(x)
        return (y_pred - y)*(y_pred - y)
    
    
    def learn(self, x, y):
        x_layers = [x]
        
        # Calculate the output for every layer and pass it to the next layer
        for layer in self.layers:
            x_layers.append(layer.calc(x_layers[-1]))
        
        # The last layer can be learned simply by comparing the output to the desired output
        last_layer = self.layers[-1]
        last_layer.learnLastLayer(x_layers[-2], y)
        
        # The other layers use the deltas and weights of the next layer
        deltas = last_layer.getDeltas()
        weights = last_layer.getWeights()
        for layer, xj in zip(reversed(self.layers[0:-1]), reversed(x_layers[0:-2])):
            layer.learnLayer(xj, deltas, weights)
            deltas = layer.getDeltas()
            weights = layer.getWeights()
        return
    
    def show(self):
        print("---------------------------------------------------------------------------")
        for layer in self.layers:
            layer.show()
        return