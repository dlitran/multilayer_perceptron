from basics import *

def saveModelHumanReadable(network: neuronalNetwork):
    architecture_dict = {}
    for i, layer in enumerate(network.layers):
        architecture_dict["layer" + str(i)] = {"activationFunction" : layer.activationFunctionName, "numberNeurons" : layer.numberOutputs}
    architecture_dict["lossFunction"] = network.lossFunctionName
    architecture_dict["numberLayers"] = len(network.layers)
    architecture_dict["numberInputs"] = network.numberInputs
    with open("./model/architecture.json", "w") as file:
        json.dump(architecture_dict, file, indent=4)

    parameters_dict = {}
    for i, layer in enumerate(network.layers):
        parameters_dict["layer_" + str(i) +"_weights"] = layer.weights.tolist()
        parameters_dict["layer_" + str(i) +"_bias"] = layer.bias.tolist()
    with open("./model/parameters.json", "w") as file:
        json.dump(parameters_dict, file, indent=4)
    print("Model saved in human readable format")

# def saveModel(network: neuronalNetwork):
#     architecture_dict = {}
#     architecture_dict["lossFunction"] = network.lossFunctionName
#     for i, layer in enumerate(network.layers):
#         architecture_dict["layer" + str(i)] = {"activationFunction" : layer.activationFunctionName, "numberNeurons" : layer.numberOutputs}

#     model_dict = {}
#     model_dict["architecture"] = np.array(architecture_dict, dtype=object)
#     for i, layer in enumerate(network.layers):
#         model_dict["layer_" + str(i) +"_weights"] = layer.weights
#         model_dict["layer_" + str(i) +"_bias"] = layer.bias
#     np.savez("./model/parameters.npz", **model_dict)
#     print("Model saved!")

def load_architecture(numberInputs):
    with open("./model/architecture.json", "r") as file:
        architecture = json.load(file)
    numberLayers = architecture["numberLayers"]
    lossFunctionName = architecture["lossFunction"]
    layers = []
    for i in range(numberLayers):
        layers.append(Layer(architecture["layer" + str(i)]["activationFunction"], numberInputs, architecture["layer" + str(i)]["numberNeurons"]))
        numberInputs = architecture["layer" + str(i)]["numberNeurons"]
    return lossFunctionName, layers

def accuracy_plot(accuracyArray, accuracyArrayVal):
    fig, ax =  plt.subplots()
    ax.plot(range(len(accuracyArray)), accuracyArray, label="training")
    ax.plot(range(len(accuracyArrayVal)), accuracyArrayVal, label="validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (Binary Cross Entropy)")
    ax.legend()
    plt.show()

def training(df_training, df_val):
    learningRate = 0.001
    # layers = [Layer("ReLU", 10, 16), Layer("ReLU", 16, 16), Layer("ReLU", 16, 16), Layer("Logistic", 16, 1)]
    # lossFunctionName = "binaryCrossEntropy"
    X, Y, NumberDataPoints = cleanData(df_training)
    #X_val, Y_val, NumberDataPointsVal = cleanData(df_val)
    numberInputs = X.shape[1] #TODO store numberInputs in the architecture.json file
    lossFunctionName, layers = load_architecture(numberInputs)

    network = neuronalNetwork(df_training, df_val, layers, lossFunctionName, learningRate) #add the epochs
    accuracyArray = []
    accuracyArrayVal = []
    for i in range(80):
        prediction_train, prediction_val = network.forwardPass()
        if i % 50 == 0:
            print(f"{i}-Error:", ((network.Y - prediction_train)**2).mean())
        if i % 10 == 0:
            Layer.learningRate = Layer.learningRate * 0.9995
        
        prediction_train = np.clip(prediction_train, epsilon, 1.0 - epsilon)
        prediction_train = network.lossFunction(network.Y, prediction_train).mean()
        accuracyArray.append(prediction_train.mean())

        prediction_val = np.clip(prediction_val, epsilon, 1.0 - epsilon)
        prediction_val = network.lossFunction(network.Y_val, prediction_val).mean()
        accuracyArrayVal.append(prediction_val.mean())

        network.backpropagation()
    accuracy_plot(accuracyArray, accuracyArrayVal)
    return network