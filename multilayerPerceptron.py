"""Class for neuronal network and layer"""

# class neuronalNetwork:


class layer:
    """docstring"""
    def __init__(self, f, inputSize, outputSize):
        self.activationFunction = f
        #maybe this in a different function -> self.input = input
        #same self.output = output (What's now the X_vector)
        self.inputSize = inputSize
        self.outputSize = outputSize
        # self.weights = #the correspondent initialize function depending on activation function used.
        self.bias = 0

    def __multiply__(self, layer2, input):
        self.input = input
        multiplied = multiplication(input, self.weights)
        added = multiplied + bias
        signal = vectorized_ReLU(added)
        self.output = signal
        X_vector.append(signal)
        input = signal #Something like this. #TODO multiply function