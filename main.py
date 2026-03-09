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

def main():
    np.random.seed(0)
    if len(sys.argv) < 2:
        print("A mode: --training or --prediction must be specified")
        return
    ratioTrainingValidation = 0.7
    df_training, df_val = load_data(ratioTrainingValidation)
    #TODO balancing the learning rate based on the B/M proportion of the training dataset
    #X, Y, NumberDataPoints = cleanData(df)
    if sys.argv[1] == "--prediction":
        network = load_model_human_readable(df_training, df_val)
        predictValue(network)
    elif sys.argv[1] == "--training":
        network = training(df_training, df_val)
        predictValue(network)
        saveModelHumanReadable(network)
    else:
        print("specify a valid mode: --training/--prediction ")
    return
    
if __name__ == "__main__":
    main()