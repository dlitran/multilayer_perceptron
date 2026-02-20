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
    
def multiplication(input: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.dot(input, weights)

def initialize_ReLU_layer(numberInputs: int, numberOutputs: int) -> np.ndarray:
    # For ReLU
    var = np.sqrt(2.0/numberInputs) #Its multiplied by 2 for a reason. Because ReLU kills half the signal?
    return np.random.normal(loc=0.0, scale=var, size=(numberInputs, numberOutputs))

def initialize_logistic_layer(numberInputs: int, numberOutputs: int) -> np.ndarray:
    var = np.sqrt(2.0/(numberInputs + numberOutputs)) #Xavier initialization (16 del input y 1 del output.)
    weightsFinalNeuron = np.random.normal(loc=0.0, scale=var, size=numberInputs)
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

def main():
    learningRate = 0.005
    df = load_data()
    N = len(df.drop(["diagnosis"], axis=1).columns)
    Y = np.array(df["diagnosis"])
    vectorized_num = np.vectorize(numeralize_diagnosis)
    Y = vectorized_num(Y)
    # X = pd.DataFrame(df.drop(["diagnosis"], axis=1))
    layer_size = [N, 64, 32, 1]
    layers = [initialize_ReLU_layer(N, 64), initialize_ReLU_layer(64, 32), initialize_ReLU_layer(32, 16)] #initialize randomly following a specific method for ReLU.

    numberDataPoints = 569 #Batch gradient descent.
    biases = [initialize_bias(numberDataPoints, 64), initialize_bias(numberDataPoints, 32), initialize_bias(numberDataPoints, 16)] 


    vectorized_ReLU = np.vectorize(ReLU)
    X = df.drop(["diagnosis"], axis=1).to_numpy()
    input = (X - X.mean()) / X.var()
    for layer, bias in zip(layers, biases):
        multiplied = multiplication(input, layer)
        added = multiplied + bias

        # Apply activation function.
        # if len(added[0]) == 16:
        #     print(added[0])
        signal = vectorized_ReLU(added)

        input = signal
    #logistic regression:
    weightsFinalNeuron = initialize_logistic_layer(16, 1) 
    biasFinalNeuron = initialize_bias(numberDataPoints, 1)

    weightSum = np.dot(input, weightsFinalNeuron)
    weightSum = weightSum + biasFinalNeuron
    # print(weightSum)
    clipped = np.clip(weightSum, -500, 500)
    p = 1 /(1 + np.exp(-clipped))
    print(p)
    #input here is the last weight (or weights)
    #We want the derivative of the weighted sum with respect to the weights
    #derivative of the activation functionn with respect the the weighted sum
    #derivative of the Loss function respect the activation function
    #This gives us the gradient.
    # print(Y)
    # print(input)
    dL_dA = Y - p
    dA_dz = p * (1 - p)
    dz_dw = weightsFinalNeuron
    dz_db = 1
    weightsFinalNeuron = weightsFinalNeuron + learningRate * 
    # print(dL)
    # print(Error.mean())
    # print(Error)
    # print(input[0])

    #Number of iteratrions
    # for iter in range(500):
        #caluclate derivatives
        # gradient = Y
        #Calculate the error (difference between )
        #calculate gradient
        #check
    #backpropagation
    # print(weights)
    


if __name__ == "__main__":
    main()

    #Neurons are just placeholders for the outputs of activation functions -> each column is a neuron.
    #When training with just a datapoint we just have a value in each column. When we have more than one datapoint,
    #we have more than one value in each column.

    #Resum del més important:
    #Imaginem que tenim el resultat de la predicció: a = 0.8
    #El resultat real es y = 1.
    #Volem calcular el gradient, vol dir -> direcció que disminueix L en funció de w.
    #dL/dw = dL/da * da/dz * dz/dw
    # L = 1/2(y - a)2
    # dL/da = delta? = -(y - a)
    #da/dz ->aquí ja anem enrere. hi ha una z per cada neurona de la capa anterior.
    #La derivada depèn de la funció d'activació que estiguem utilitzant. -> obtenim un resultat per cada neurona de la capa anterior.
    #dz/dw = wx -> x (on x son les a de la capa anterior' o el input.)
    #En cas de que hi hagi una capa anterior ':
    #multipliquem per da'/dz'
    #multipiliquem per dz'/dw' -> x' on x' es l'input o prové de la capa anterior''.