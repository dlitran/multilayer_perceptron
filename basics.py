import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import json
import sys

epsilon=1e-7


#Activation functions
def ReLU(weightSum: float) -> float:
    return(max(0.0, weightSum))

def logistic(weightSum: float) -> float:
    clipped = np.clip(weightSum, -500, 500)
    p = 1 /(1 + np.exp(-clipped))
    return p

def softmax(values):
    shifted_values = values - max(values) #avoid overflow
    result = np.exp(shifted_values) / np.exp(shifted_values).sum()
    return result

#Derivatives dA_dz (for delta calculus)
def derivative_ReLU(x: float) -> float:
    if x > 0:
        return 1
    else:
        return 0
    
def derivative_Logistic(p: float) -> float:
    return p * (1 - p)

def derivative_Softmax(values):
    result = np.ndarray(shape=[len(values), len(values)])
    for (i, j), value in np.ndenumerate(result):
        if i == j:
            result[i, j] = values[i] * (1 - values[i])
        else:
            result[i, j] = - values[i] * values[j]
    return result

#layer initializations
def initialize_ReLU_layer(numberInputs: int, numberOutputs: int) -> np.ndarray:
    # For ReLU
    var = np.sqrt(2.0/numberInputs) #Its multiplied by 2 for a reason. Because ReLU kills half the signal?
    return np.random.normal(loc=0.0, scale=var, size=(numberInputs, numberOutputs))

def xavier_normal_initialization(numberInputs: int, numberOutputs: int) -> np.ndarray:
    var = np.sqrt(2.0/(numberInputs + numberOutputs)) #Xavier initialization (16 del input y 1 del output.)
    weightsFinalNeuron = np.random.normal(loc=0.0, scale=var, size=(numberInputs, numberOutputs))
    return weightsFinalNeuron

def initialize_bias(numberOutputs: int) -> np.ndarray: #I believe that instead of number inputs I shouldput the number of datapoints use to train the model
    # Create a single random row and tile it
    bias = np.zeros(numberOutputs)
    # bias = np.tile(base_row, (numberDataPoints, 1))
    return bias

#utils
def multiplication(input: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.dot(input, weights)

#Loss functions
def MSE(value, prediction):
    return (value - prediction)**2

def binaryCrossEntropy(value, prediction):
    return -( value * np.log(prediction) + (1 - value) * np.log(1 - prediction))

def categoricalCrossEntropy(values, prediction):
    result  = 0
    
    result += -( values * np.log(prediction[:, 0:1]))
    opposite_values = 1 - values
    result += -( opposite_values * np.log(prediction[:, 1:2]))
    return result


#Loss function derivatives
def MSE_derivative(value, prediction):
    return prediction - value

def binaryCrossEntropyDerivative(value, prediction):
    return (prediction - value)/(prediction * (1 - prediction) + epsilon)

def categoricalCrossEntropyDerivative(value, prediction):
    opposite_value = 1 - value
    result0 = - value / (prediction[:, 0:1] + epsilon)
    result1 = - opposite_value / (prediction[:, 1:2]  + epsilon)
    result = np.hstack((result0, result1))
    return  result #TODO look up the actual formula

def evaluatePrediction(realValue, prediction):
    if abs(prediction - realValue) < 0.5:
        return True
    else:
        return False
    
def evaluatePredictionSoftmax(realValue, prediction):
    # print(prediction.shape)
    firstElementPrediction = prediction[:, 0:1]
    result = firstElementPrediction - realValue < 0.5
    # if len(result) == sum(1 for x in result if x == True):
        # print("100% accuracy.")
    
    return result

class Layer:
    initializeWeightsFunctions = {"ReLU" : initialize_ReLU_layer, "Logistic": xavier_normal_initialization, "Softmax": xavier_normal_initialization}
    activationFunctions = {"ReLU" : np.vectorize(ReLU), "Logistic" : np.vectorize(logistic), "Softmax": np.vectorize(softmax, signature='(n)->(n)')}
    deltaCalculus = {"ReLU" : np.vectorize(derivative_ReLU), "Logistic" : np.vectorize(derivative_Logistic), "Softmax": np.vectorize(derivative_Softmax, signature='(n)->(n, n)')}
    learningRate = None

    def __init__(self, activationFunction, numberInputs, numberOutputs):
        self.numberInputs = numberInputs
        self.numberOutputs = numberOutputs

        self.weights = self.initializeWeightsFunctions[activationFunction](numberInputs, numberOutputs)
        self.bias = initialize_bias(numberOutputs)

        self.activationFunctionName = activationFunction
        self.activationFunction = self.activationFunctions[activationFunction]
        self.deltaCalculus = self.deltaCalculus[activationFunction]

        self.activationFunctionOutput = None
        self.inputCurrentLayer = None
    
    def actualizeWeights(self, lastDelta, batchSize) -> np.ndarray:

        dA_dz = self.deltaCalculus(self.activationFunctionOutput)
        print("lastDelta:", lastDelta.shape)
        print("dA_dz:", dA_dz.shape)
        #print(dA_dz)
        #print(len(lastDelta.shape))
        if len(dA_dz.shape) == 3:
            delta = np.einsum('ij, ijk->ik', lastDelta, dA_dz)
        else:
            delta = (lastDelta * dA_dz) 
        print("\033[1;31mdelta = lastDelta * dA_dz =\033[0m", delta.shape)
        dz_dw = self.inputCurrentLayer #X
        print("dz_dw:", dz_dw.shape)

        dL_dw = np.dot((delta).T, dz_dw).T
        print("\033[1;31mdL_dw = lastDelta * dA_dz * dz_dw =\033[0m", dL_dw.shape)
        tmp_dz_dX = self.weights #X is the previous A.
        # print("dz_dX:", dz_dX.shape)

        self.weights = self.weights - self.learningRate * (dL_dw /batchSize)
        self.bias = self.bias - self.learningRate * delta.mean(axis=0)
        delta = np.dot(delta, tmp_dz_dX.T) #dz_da
        return delta
    
class neuronalNetwork:
    """docstring"""
    lossFunctions_dict = {"MSE" : MSE, "binaryCrossEntropy": binaryCrossEntropy, "categoricalCrossEntropy": categoricalCrossEntropy}
    lossFunctinoDerivative_dict = {"MSE" : MSE_derivative, "binaryCrossEntropy": binaryCrossEntropyDerivative, "categoricalCrossEntropy": categoricalCrossEntropyDerivative}
    def __init__(self, df_training, df_val, layers, lossFunction, learningRate, batchSize):
        """constructor"""
        self.X, self.Y, self.numberDataPointsTrain = cleanData(df_training)
        self.X_val, self.Y_val, self.numberDataPointsVal = cleanData(df_val)
        self.numberInputs = self.X.shape[1]
        self.layers = layers
        Layer.learningRate = learningRate
        self.batchSize = batchSize

        self.lossFunctionName = lossFunction
        self.lossFunction = self.lossFunctions_dict[lossFunction]
        self.lossFunctionDerivative = self.lossFunctinoDerivative_dict[lossFunction]

    def actualForwardPass(self, input):
        for i, layer in enumerate(self.layers):
            self.layers[i].inputCurrentLayer = input
            multiplied = multiplication(input, self.layers[i].weights)
            added = multiplied + self.layers[i].bias
            signal = self.layers[i].activationFunction(added)
            # if layer.activationFunctionName == "Softmax":
            #     print("DONE")
            self.layers[i].activationFunctionOutput = signal
            input = signal
        prediction = self.layers[-1].activationFunctionOutput
        return prediction
        
    def forwardPass(self, validation=True):
        #validation dataset
        if validation is True:
            prediction_val = self.actualForwardPass(self.X_val)
        #training dataset
        #randomized dataset
        batchSize = len(self.Y)
        indices = np.random.permutation(len(self.Y))
        shuffled_X = self.X[indices]
        shuffled_Y = self.Y[indices]
        if self.lossFunctionName == "categoricalCrossEntropy":
            prediction = np.empty((0,2)) #HARDCODED
        else:
            prediction = np.empty((0, 1))
        # print(self.numberDataPointsTrain)
        # print(self.numberDataPointsVal)
        for batch in range(0, self.numberDataPointsTrain, self.batchSize):
            #input = batch
            batch_x = shuffled_X[batch : batch + self.batchSize]
            batch_y = shuffled_Y[batch: batch + self.batchSize]
            # print("iteration:", batch)
            for i, layer in enumerate(self.layers):
                self.layers[i].inputCurrentLayer = batch_x
                multiplied = multiplication(batch_x, self.layers[i].weights)
                added = multiplied + self.layers[i].bias

                signal = self.layers[i].activationFunction(added)
                self.layers[i].activationFunctionOutput = signal
                batch_x = signal
            batchPrediction = self.layers[-1].activationFunctionOutput
            #calculate loss too
            # print("Inputs loss functino:", batch_y, batchPrediction)
            delta = self.lossFunctionDerivative(batch_y, batchPrediction)
            print("The shape: ", delta.shape)
            #print(delta)
            prediction = np.vstack((prediction, batchPrediction))
            self.backpropagation(delta, batchSize)
            #batch += self.batchSize
        #average loss and accuracy for batch
        # print(prediction.shape)
        reverse_mapping = np.argsort(indices)
        prediction = prediction[reverse_mapping]
        return prediction, prediction_val

    def backpropagation(self, delta, batchSize):
        for currentLayer in reversed(self.layers):
            delta = currentLayer.actualizeWeights(delta, batchSize)
            # return
            # dz_dX = currentLayer.weights #X is the previous A.
            # # print("dz_dX:", dz_dX.shape)
            # delta = np.dot(delta, dz_dX.T) #dz_da
            # print("\033[1;33mlastDelta = dz_dX * delta =\033[0m", delta.shape)

def numeralize_diagnosis(diagnosis: str) -> float:
    if diagnosis == 'M':
        return 1.
    elif diagnosis == 'B':
        return 0.

def cleanData(df):
    Y = np.array(df["diagnosis"])
    vectorized_num = np.vectorize(numeralize_diagnosis)
    Y = vectorized_num(Y)
    Y = np.reshape(Y, (-1, 1))

    X = df.drop(["diagnosis"], axis=1)
    #numberVariablesInput = len(X.columns)
    X = X.to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0) #normalization, we assume that this is the training dataset.

    NumberDataPoints = Y.shape[0]
    return X, Y, NumberDataPoints