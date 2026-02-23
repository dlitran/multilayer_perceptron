"""Main program for multilayer perceptron"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class neuronalNetwork:
    """docstring"""
    def __init__(self, numberVariablesInput, X, Y, numberDataPoints, learningRate=0.005):
        """constructor"""
        numberVariablesInput = numberVariablesInput
        X = X
        Y = Y
        learningRate = learningRate
        numberDataPoints = numberDataPoints
        layers = [initialize_ReLU_layer(numberVariablesInput, 64), initialize_ReLU_layer(64, 32), initialize_ReLU_layer(32, 16)] #initialize randomly following a specific method for ReLU.
        biases = [initialize_bias(numberDataPoints, 64), initialize_bias(numberDataPoints, 32), initialize_bias(numberDataPoints, 16)] 
        activationFunctionOutput = []
        inputArray = []
    
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
    
def multiplication(input: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.dot(input, weights)

def initialize_ReLU_layer(numberInputs: int, numberOutputs: int) -> np.ndarray:
    # For ReLU
    var = np.sqrt(2.0/numberInputs) #Its multiplied by 2 for a reason. Because ReLU kills half the signal?
    return np.random.normal(loc=0.0, scale=var, size=(numberInputs, numberOutputs))

def initialize_logistic_layer(numberInputs: int, numberOutputs: int) -> np.ndarray:
    var = np.sqrt(2.0/(numberInputs + numberOutputs)) #Xavier initialization (16 del input y 1 del output.)
    weightsFinalNeuron = np.random.normal(loc=0.0, scale=var, size=(numberInputs, numberOutputs))
    return weightsFinalNeuron

def initialize_bias(numberDataPoints: int, numberOutputs: int) -> np.ndarray: #I believe that instead of number inputs I shouldput the number of datapoints use to train the model
    # Create a single random row and tile it
    bias = np.zeros(numberOutputs)
    # bias = np.tile(base_row, (numberDataPoints, 1))
    return bias

def ReLU(num: float) -> float:
    return(max(0.0, num))

def numeralize_diagnosis(diagnosis: str) -> float:
    if diagnosis == 'M':
        return 1.
    elif diagnosis == 'B':
        return 0.

def derivative_ReLU(x: float) -> float:
    if x > 0:
        return 1
    else:
        return 0

def forward_pass(layers, biases, inputArray, numberDataPoints, activationFunctionOutput, input) -> [np.ndarray, np.ndarray, np.ndarray]:
    vectorized_ReLU = np.vectorize(ReLU)
    for layer, bias in zip(layers, biases):
        inputArray.append(input)
        multiplied = multiplication(input, layer)
        added = multiplied + bias

        # Apply activation function.
        # if len(added[0]) == 16:
        #     print(added[0])
        signal = vectorized_ReLU(added)
        activationFunctionOutput.append(signal)
        input = signal
    #logistic regression:
    weightsFinalNeuron = initialize_logistic_layer(16, 1) 
    biasFinalNeuron = initialize_bias(numberDataPoints, 1)

    weightSum = np.dot(input, weightsFinalNeuron)
    weightSum = weightSum + biasFinalNeuron
    # print(weightSum)
    clipped = np.clip(weightSum, -500, 500)
    p = 1 /(1 + np.exp(-clipped))
    return p, weightsFinalNeuron, input, biasFinalNeuron

def main():
    df = load_data()

    Y = np.array(df["diagnosis"])
    vectorized_num = np.vectorize(numeralize_diagnosis)
    Y = vectorized_num(Y)

    X = df.drop(["diagnosis"], axis=1)
    numberVariablesInput = len(X.columns)
    X = X.to_numpy()
    input = (X - X.mean()) / X.var()

    learningRate = 0.005
    numberDataPoints = 569 #Batch gradient descent.
    layer_size = [numberVariablesInput, 64, 32, 1]

    activationFunctionOutput = []
    inputArray = []

    layers = [initialize_ReLU_layer(numberVariablesInput, 64), initialize_ReLU_layer(64, 32), initialize_ReLU_layer(32, 16)] #initialize randomly following a specific method for ReLU.
    biases = [initialize_bias(numberDataPoints, 64), initialize_bias(numberDataPoints, 32), initialize_bias(numberDataPoints, 16)] 

    network = neuronalNetwork(numberVariablesInput, X, Y, numberDataPoints)

    # for i in range(500):
    p, weightsFinalNeuron, input, biasFinalNeuron = forward_pass(layers, biases, inputArray, numberDataPoints, activationFunctionOutput, input)
    p = p.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    dL_dA = Y - p
    dA_dz = p * (1 - p)
    delta = (dL_dA * dA_dz)
    delta = delta.reshape(1, -1)
    dz_dw = input
    dL_dw = np.dot(delta, dz_dw) / N
    dL_dw = dL_dw.reshape(-1, 1)
    dz_db = 1

    weightsFinalNeuron = weightsFinalNeuron - learningRate * dL_dw
    biasFinalNeuron = biasFinalNeuron - learningRate * delta.mean() * dz_db 

    vectorized_derivative = np.vectorize(derivative_ReLU)
    for layer, bias, x, input in zip(layers, biases, activationFunctionOutput, inputArray):
        dA_dz = vectorized_derivative(x)
        dz_dw = input
        dz_dw = dz_dw.T
        delta = dL_dA * dA_dz
        dL_dw = np.dot(dz_dw, delta)
        print("dL_dw:", dz_dw.shape)
        print("delta:", delta.shape)
        layer = layer - learningRate * dL_dw
        bias = bias - learningRate * delta.mean()
    
if __name__ == "__main__":
    main()