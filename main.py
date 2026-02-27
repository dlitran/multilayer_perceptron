"""Main program for multilayer perceptron"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

epsilon=1e-7

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

def MSE(value, prediction):
    return (value - prediction)**2

def binaryCrossEntropy(value, prediction):
    return -( value * np.log(prediction) + (1 - value) * np.log(1 - prediction))

def MSE_derivative(value, prediction):
    return prediction - value

def binaryCrossEntropyDerivative(value, prediction):
    return (prediction - value)/(prediction * (1 - prediction) + epsilon)

class Layer:
    initializeWeightsFunctions = {"ReLU" : initialize_ReLU_layer, "Logistic": initialize_logistic_layer}
    activationFunctions = {"ReLU" : np.vectorize(ReLU), "Logistic" : np.vectorize(logistic)}
    deltaCalculus = {"ReLU" : np.vectorize(derivative_ReLU), "Logistic" : np.vectorize(derivative_Logistic)}
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
        self.inputLayer = None
    
    def actualizeWeights(self, lastDelta) -> np.ndarray:

        dA_dz = self.deltaCalculus(self.activationFunctionOutput)
        # print("lastDelta:", lastDelta.shape)
        # print("dA_dz:", dA_dz.shape)
   
        delta = (lastDelta * dA_dz) 
        # print("\033[1;31mdelta = lastDelta * dA_dz =\033[0m", delta.shape)

        dz_dw = self.inputLayer #X
        # print("dz_dw:", dz_dw.shape)

        dL_dw = np.dot((lastDelta * dA_dz).T, dz_dw).T
        # print("\033[1;31mdL_dw = lastDelta * dA_dz * dz_dw =\033[0m", dL_dw.shape)
        tmp_dz_dX = self.weights #X is the previous A.
        # print("dz_dX:", dz_dX.shape)
        self.weights = self.weights - self.learningRate * (dL_dw) #TODO for now it's hardcoded. It's the numberDataPoints.
        self.bias = self.bias - self.learningRate * delta.mean(axis=0)
        delta = np.dot(delta, tmp_dz_dX.T) #dz_da
        return delta
    
class neuronalNetwork:
    """docstring"""
    lossFunctions_dict = {"MSE" : MSE, "binaryCrossEntropy": binaryCrossEntropy}
    lossFunctinoDerivative_dict = {"MSE" : MSE_derivative, "binaryCrossEntropy": binaryCrossEntropyDerivative}
    def __init__(self, X, Y, numberDataPoints, layers, lossFunction, learningRate=0.01):
        """constructor"""
        self.X = X
        self.Y = Y
        self.numberDataPoints = numberDataPoints
        self.layers = layers
        Layer.learningRate = learningRate

        self.lossFunctionName = lossFunction
        self.lossFunction = self.lossFunctions_dict[lossFunction]
        self.lossFunctionDerivative = self.lossFunctinoDerivative_dict[lossFunction]

    def forwardPass(self, iteration=0):
        vectorized_ReLU = np.vectorize(ReLU)
        input = self.X
        for i, layer in enumerate(self.layers):
            self.layers[i].inputLayer = input
            multiplied = multiplication(input, self.layers[i].weights)
            added = multiplied + self.layers[i].bias

            signal = self.layers[i].activationFunction(added)
            self.layers[i].activationFunctionOutput = signal
            input = signal
        if iteration % 50 == 0:
            print(f"{iteration}-Error:", ((self.Y - self.layers[-1].activationFunctionOutput)**2).mean())

    def backpropagation(self):
        #TODO add more loss functions
        #TODO I should use another derivative for the
        delta = self.lossFunctionDerivative(self.Y, self.layers[-1].activationFunctionOutput) #vector of size numberDatapoints. MSE

        # delta = self.layers[-1].activationFunctionOutput * (1 - self.layers[-1].activationFunctionOutput)
        for currentLayer in reversed(self.layers):
            delta = currentLayer.actualizeWeights(delta)
            # dz_dX = currentLayer.weights #X is the previous A.
            # # print("dz_dX:", dz_dX.shape)
            # delta = np.dot(delta, dz_dX.T) #dz_da
            # print("\033[1;33mlastDelta = dz_dX * delta =\033[0m", delta.shape)

def saveModel(network: neuronalNetwork):
    architecture_dict = {}
    architecture_dict["lossFunction"] = network.lossFunctionName
    for i, layer in enumerate(network.layers):
        architecture_dict["layer" + str(i)] = {"activationFunction" : layer.activationFunctionName, "numberInputs" : layer.numberInputs, "numberOutputs" : layer.numberOutputs}

    model_dict = {}
    model_dict["architecture"] = np.array(architecture_dict, dtype=object)
    for i, layer in enumerate(network.layers):
        model_dict["layer_" + str(i) +"_weights"] = layer.weights
        model_dict["layer_" + str(i) +"_bias"] = layer.bias
    np.savez("./parameters.npz", **model_dict)
    print("Model saved!")


def load_model(X, Y, NumberDataPoints):
    model = np.load("./parameters.npz", allow_pickle=True)
    architecture = model["architecture"].item()
    lossFunctionName = architecture["lossFunction"]
    numberLayers = len(architecture) -1
    layers = []
    for i in range(numberLayers):
        layers.append(Layer(architecture["layer" + str(i)]["activationFunction"], architecture["layer" + str(i)]["numberInputs"], architecture["layer" + str(i)]["numberOutputs"]))
        bias = model["layer_" + str(i) + "_bias"]
        weights = model["layer_" + str(i) + "_weights"]
        layers[i].weights = weights
        layers[i].bias = bias
    network = neuronalNetwork(X, Y, NumberDataPoints, layers, lossFunctionName)
    return network

def predictValue(network: neuronalNetwork):
    network.forwardPass()
    result = network.layers[-1].activationFunctionOutput
    correct = 0
    incorrect = 0
    for prediction, realValue in zip(result, network.Y):
        if abs(prediction - realValue) < 0.5:
            correct += 1
        else:
            incorrect += 1
    print(f"correct: {correct}, incorrect: {incorrect}, accuracy: {((correct * 100)/(correct + incorrect))}%")

def main():
    np.random.seed(0)
    df = load_data()
    Y = np.array(df["diagnosis"])
    vectorized_num = np.vectorize(numeralize_diagnosis)
    Y = vectorized_num(Y)
    Y = np.reshape(Y, (-1, 1))

    X = df.drop(["diagnosis"], axis=1)
    #numberVariablesInput = len(X.columns)
    X = X.to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0) #normalization, we assume that this is the training dataset.

    learningRate = 0.005
    NumberDataPoints = 569
    lossFunction = "binaryCrossEntropy"
    if len(sys.argv) == 2:
        network = load_model(X, Y, NumberDataPoints)
        predictValue(network)
        return
    layers = [Layer("ReLU", 10, 16), Layer("ReLU", 16, 16), Layer("ReLU", 16, 16), Layer("Logistic", 16, 1)]
    network = neuronalNetwork(X, Y, NumberDataPoints, layers, lossFunction, learningRate) #add the epochs
    for i in range(700):
        network.forwardPass(i)
        network.backpropagation()
    #print(network.layers[-1].activationFunctionOutput)
    #Save weights
    saveModel(network)
    return
    
if __name__ == "__main__":
    main()