"""Main program for multilayer perceptron"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data() -> pd.DataFrame:
    """docstring for function: loads data"""
    columnNames = ["id", "diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compacteness", 
    "concavity", "concave points", "symmetry", "fractal dimension", "radius SE", "texture SE", "perimeter SE", "area SE", "smoothness SE", "compacteness SE", 
    "concavity SE", "concave points SE", "symmetry SE", "fractal dimension SE", "radius WORST", "texture WORST", "perimeter WORST", "area WORST", "smoothness WORST", "compacteness WORST", 
    "concavity WORST", "concave points WORST", "symmetry WORST", "fractal dimension WORST"]
    df = pd.read_csv("./data.csv", header=None, names=columnNames)
    # df.set_index("id")
    df = df[["diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compacteness", 
    "concavity", "concave points", "symmetry", "fractal dimension"]]
    return (df)

def numeralize_diagnosis(diagnosis: str) -> float:
    if diagnosis == 'M':
        return 1.
    elif diagnosis == 'B':
        return 0.

#Derivatives dA_dz (for delta calculus)
def derivative_ReLU(x: float) -> float:
    if x > 0:
        return 1
    else:
        return 0
    
def derivative_Logistic(p: float) -> float:
    return p * (1 - p)

#Activation functions
def ReLU(weightSum: float) -> float:
    return(max(0.0, weightSum))

def logistic(weightSum: float) -> float:
    clipped = np.clip(weightSum, -500, 500)
    p = 1 /(1 + np.exp(-clipped))
    return p

#layer initializations
def initialize_ReLU_layer(numberInputs: int, numberOutputs: int) -> np.ndarray:
    # For ReLU
    var = np.sqrt(2.0/numberInputs) #Its multiplied by 2 for a reason. Because ReLU kills half the signal?
    return np.random.normal(loc=0.0, scale=var, size=(numberInputs, numberOutputs))

def initialize_logistic_layer(numberInputs: int, numberOutputs: int) -> np.ndarray:
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

class Layer:
    initializeWeightsFunctions = {"ReLU" : initialize_ReLU_layer, "Logistic": initialize_logistic_layer}
    activationFunctions = {"ReLU" : np.vectorize(ReLU), "Logistic" : np.vectorize(logistic)}
    deltaCalculus = {"ReLU" : np.vectorize(derivative_ReLU), "Logistic" : np.vectorize(derivative_Logistic)}
    learningRate = 0.005

    def __init__(self, activationFunction, numberInputs, numberOutputs):
        self.weights = self.initializeWeightsFunctions[activationFunction](numberInputs, numberOutputs)
        self.bias = initialize_bias(numberOutputs)

        self.activationFunction = self.activationFunctions[activationFunction]
        self.deltaCalculus = self.deltaCalculus[activationFunction]

        self.activationFunctionOutput = None
        self.inputLayer = None
    
    def actualizeWeights(self, lastDelta) -> np.ndarray:
        delta = lastDelta * self.deltaCalculus(self.activationFunctionOutput)
        
        dz_dw = self.inputLayer
        dz_dw = dz_dw.T

        dL_dw = np.dot(dz_dw, delta)
        self.weights = layer - self.learningRate * dL_dw
        self.bias = self.bias - self.learningRate * delta.mean()
        return delta
    
class neuronalNetwork:
    """docstring"""
    def __init__(self, numberVariablesInput, X, Y, numberDataPoints, layers):
        """constructor"""
        self.numberVariablesInput = numberVariablesInput
        self.X = X
        self.Y = Y
        self.numberDataPoints = numberDataPoints
        self.layers = layers

    def forwardPass(self):
        vectorized_ReLU = np.vectorize(ReLU)
        input = self.X
        for i, layer in enumerate(self.layers):
            self.layers[i].inputLayer = input
            multiplied = multiplication(input, self.layers[i].weights)
            added = multiplied + self.layers[i].bias

            signal = self.layers[i].activationFunction(added)
            self.layers[i].activationFunctionOutput = signal
            input = signal

    def backpropagation(self):
        delta = self.Y - self.layers[-1].activationFunctionOutput
        for i, layer in enumerate(reversed(self.layers)):
            self.layers[i].deltaCalculus(delta)


# def forward_pass(network: neuronalNetwork) -> [np.ndarray, np.ndarray, np.ndarray]:
#     vectorized_ReLU = np.vectorize(ReLU)
#     for layer, bias in zip(network.layers, network.biases):
#         network.inputArray.append(network.X)
#         multiplied = multiplication(network.X, layer)
#         added = multiplied + bias

#         signal = vectorized_ReLU(added)
#         network.activationFunctionOutput.append(signal)
#         network.X = signal
    

#     weightSum = np.dot(network.X, network.weightsFinalNeuron)
#     weightSum = weightSum + network.biasFinalNeuron
#     # print(weightSum)
#     clipped = np.clip(weightSum, -500, 500)
#     p = 1 /(1 + np.exp(-clipped))
#     return p

def main():
    df = load_data()
    Y = np.array(df["diagnosis"])
    vectorized_num = np.vectorize(numeralize_diagnosis)
    Y = vectorized_num(Y)

    X = df.drop(["diagnosis"], axis=1)
    numberVariablesInput = len(X.columns)
    X = X.to_numpy()
    X = (X - X.mean()) / X.var()


    layers = [Layer("ReLU", 10, 64), Layer("ReLU", 64, 32), Layer("ReLU", 32, 16), Layer("Logistic", 16, 1)]
    
    network = neuronalNetwork(numberVariablesInput, X, X, 569, layers) #add the epochs and maybe learning rate.
    network.forwardPass()
    network.backpropagation()
    print(network.layers[-1].activationFunctionOutput)
    return
    
if __name__ == "__main__":
    main()