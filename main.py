"""Main program for multilayer perceptron"""
from basics import *
from training import *
from prediction import *

def load_data(ratioTrainingValidation) -> pd.DataFrame:
    """docstring for function: loads data"""
    columnNames = ["id", "diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compacteness", 
    "concavity", "concave points", "symmetry", "fractal dimension", "radius SE", "texture SE", "perimeter SE", "area SE", "smoothness SE", "compacteness SE", 
    "concavity SE", "concave points SE", "symmetry SE", "fractal dimension SE", "radius WORST", "texture WORST", "perimeter WORST", "area WORST", "smoothness WORST", "compacteness WORST", 
    "concavity WORST", "concave points WORST", "symmetry WORST", "fractal dimension WORST"]
    df = pd.read_csv("./data.csv", header=None, names=columnNames)
    # df.set_index("id")
    df = df[["diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compacteness", 
    "concavity", "concave points", "symmetry", "fractal dimension"]]

    df_training = df.sample(frac=ratioTrainingValidation, random_state=0)
    df_val = df.drop(df_training.index)
    print(df_training.shape, df_val.shape)
    return df_training, df_val

def validateInput():
    flag = "--validationRatio"
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if (index + 1) >= len(sys.argv):
            raise ValueError(f"A value after the {flag} flag must be specified.")
        if float(sys.argv[index + 1]) <= 0 or float(sys.argv[index + 1]) >= 1:
            raise ValueError("ratio training-validation must be between 1 and 0")
        ratioTrainingValidation = float(sys.argv[index + 1])
    else:
        ratioTrainingValidation = 0.7
    flag = "--epochs"
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if (index + 1) >= len(sys.argv):
            raise ValueError(f"A value after the {flag} flag must be specified.")
        if float(sys.argv[index + 1]) < 0:
            raise ValueError("Epoch value must be non-negative")
        epochs = int(sys.argv[index + 1])
    else:
        epochs = 80
    flag = "--learningRate"
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if (index + 1) >= len(sys.argv):
            raise ValueError(f"A value after the {flag} flag must be specified.")
        if float(sys.argv[index + 1]) < 0:
            raise ValueError("Learning rate value must be non-negative")
        learningRate = float(sys.argv[index + 1])
    else:
        learningRate = 0.5
    flag = "--batchSize"
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if (index + 1) >= len(sys.argv):
            raise ValueError(f"A value after the {flag} flag must be specified.")
        if int(sys.argv[index + 1]) <= 0:
            raise ValueError("batch size must equal or greater than 1")
        batchSize = int(sys.argv[index + 1])
    else:
        batchSize = None
    return ratioTrainingValidation, epochs, learningRate, batchSize

def main():
    np.random.seed(0)
    if len(sys.argv) < 2:
        print("A mode: --training or --prediction must be specified")
        return
    # derivative_Softmax(np.array([1, 2, 3]))
    # return
    ratioTrainingValidation, epochs, learningRate, batchSize = validateInput()
    df_training, df_val = load_data(ratioTrainingValidation)
    if batchSize == None:
        batchSize = df_training.shape[0]
    elif batchSize > df_training.shape[0]:
        raise ValueError(f"batch size must equal or smaller than the number of training datapoints ({df_training.shape[0]})")
    #TODO balancing the learning rate based on the B/M proportion of the training dataset
    #X, Y, NumberDataPoints = cleanData(df)
    print(batchSize)
    if sys.argv[1] == "--prediction":
        network = load_model_human_readable(df_training, df_val)
        predictValue(network)
    elif sys.argv[1] == "--training":
        network = training(df_training, df_val, learningRate, epochs, batchSize)
        return
        #TODO mirar por qué falla.
        predictValue(network)
        saveModelHumanReadable(network)
    else:
        print("specify a valid mode: --training/--prediction ")
    return
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    #TODO hacer varias versiones del forward pass. Cuand hay batches, hay que aplicar inmediatamente el backpropagation. Quizás cambiar también las clases.