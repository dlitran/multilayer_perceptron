from basics import *

def load_model_human_readable(df_training, df_val):
    #TODO I need to read two files, the model.json and the paramters.json
    X, Y, NumberDataPoints = cleanData(df_training)
    with open("./model/architecture.json", "r") as file:
        architecture = json.load(file)
    numberLayers = architecture["numberLayers"]
    lossFunctionName = architecture["lossFunction"]

    with open("./model/parameters.json", "r") as file:
        parameters = json.load(file)
    layers = []
    numberInputs = X.shape[1]
    for i in range(numberLayers):
        layers.append(Layer(architecture["layer" + str(i)]["activationFunction"], numberInputs, architecture["layer" + str(i)]["numberNeurons"]))
        numberInputs = architecture["layer" + str(i)]["numberNeurons"]
        layers[i].weights = np.array(parameters["layer_" + str(i) + "_weights"])
        layers[i].bias = np.array(parameters["layer_" + str(i) + "_bias"])
    
    network = neuronalNetwork(df_training, df_val, layers, lossFunctionName)
    return network

# def load_model(X, Y, NumberDataPoints):
#     model = np.load("./model/parameters.npz", allow_pickle=True)
#     architecture = model["architecture"].item()
#     lossFunctionName = architecture["lossFunction"]
#     numberLayers = len(architecture) -1
#     layers = []
#     numberInputs = X.shape[1]
#     for i in range(numberLayers):
#         layers.append(Layer(architecture["layer" + str(i)]["activationFunction"], numberInputs, architecture["layer" + str(i)]["numberNeurons"]))
#         numberInputs = architecture["layer" + str(i)]["numberNeurons"]
#         layers[i].weights = model["layer_" + str(i) + "_weights"]
#         layers[i].bias = model["layer_" + str(i) + "_bias"]
    
#     network = neuronalNetwork(X, Y, NumberDataPoints, layers, lossFunctionName)
#     return network

def predictValue(network: neuronalNetwork):
    predictionTrain, predictionVal = network.forwardPass()
    correct = 0
    incorrect = 0
    for prediction, realValue in zip(predictionTrain, network.Y):
        if abs(prediction - realValue) < 0.5:
            correct += 1
        else:
            incorrect += 1
    
    correct_val = 0
    incorrect_val = 0
    for prediction, realValue in zip(predictionVal, network.Y_val):
        if abs(prediction - realValue) < 0.5:
            correct_val += 1
        else:
            incorrect_val += 1
    print(f"training data - correct: {correct}, incorrect: {incorrect}, accuracy: {((correct * 100)/(correct + incorrect))}%")
    print(f"validation data - correct: {correct_val}, incorrect: {incorrect_val}, accuracy: {((correct_val * 100)/(correct_val + incorrect_val))}%")
