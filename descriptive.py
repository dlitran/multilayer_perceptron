"""This thing was used to describe the module"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data() -> pd.DataFrame:
    columnNames = ["id", "diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compacteness", 
    "concavity", "concave points", "symmetry", "fractal dimension", "radius SE", "texture SE", "perimeter SE", "area SE", "smoothness SE", "compacteness SE", 
    "concavity SE", "concave points SE", "symmetry SE", "fractal dimension SE", "radius WORST", "texture WORST", "perimeter WORST", "area WORST", "smoothness WORST", "compacteness WORST", 
    "concavity WORST", "concave points WORST", "symmetry WORST", "fractal dimension WORST"]
    df = pd.read_csv("./data.csv", header=None, names=columnNames)
    df.set_index("id")
    return df

def general_description(df):
    print(df.info())
    print(df.describe())

def create_bar_plot(df):
    counts = df["diagnosis"].value_counts()

    plt.bar(x=counts.index, height=counts.values, color=['skyblue', 'salmon'])
    plt.title('Count of Benign vs Malignant Cells')
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.show()

def create_scatter_plot(df):
    dfMean = df[["diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compacteness", 
    "concavity", "concave points", "symmetry", "fractal dimension"]]

    fig, axes = plt.subplots(3, 4, squeeze=True, figsize=(16, 12)) #4 inches height and 4 inches width per plot!
    flat_axes = axes.flatten()
    for i in range(len(dfMean.loc[:, dfMean.columns != "diagnosis"].columns)):
        # print(dfMean.columns[i + 1])
        mean = dfMean.groupby("diagnosis")[dfMean.columns[i + 1]].mean()
        std = dfMean.groupby("diagnosis")[dfMean.columns[i + 1]].std()
        flat_axes[i].set_xlabel("x")
        flat_axes[i].set_ylabel("y")


        bars = flat_axes[i].bar(mean.index, mean.values, yerr=std.values, capsize=5, tick_label=mean.index, color=["skyblue", "salmon"])
        flat_axes[i].set_title(f"mean of {dfMean.columns[i + 1]}")

    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].axis('off')

    fig.suptitle("Comparison bening vs maling cells for 10 real value parameters")
    

    labels = ['Benign', 'Malignant']
    fig.legend(bars, labels, loc='lower right', ncol=2, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=5.0) #rect[left, bottom, right, top] is the % where the plot begins and ends in the fig box. 0.95 means it just fills the 95% of the box. It allows us to leave space for the title, legend...
    plt.show()
    

def main():
    df = load_data()
    # general_description(df)

    
    # create_bar_plot(df)
    #Scatter plot
    create_scatter_plot(df)
    #pair plot


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e.__class__.__name__, ":", e)