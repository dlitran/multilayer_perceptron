# Multilayer Perceptron 🧠

This project features the from-scratch implementation of a multilayer perceptron to classify breast cancer cells as malignant or benign. It is part of the programming curriculum at 42 Barcelona.

**Status:** 🚧 Work in Progress (Core training, prediction, and batch processing functionalities are operational).

## Features

* **From-Scratch Implementation:** No machine learning libraries were used for the underlying algorithms; you must code everything from scratch. The gradient descent and backpropagation are completely custom-built.
* **Customizable Architecture:** Neural network topology (layers, neurons, activation functions, and loss functions) is completely modular and driven by an `architecture.json` file.
* **Mini-Batch Gradient Descent:** Fully supports dynamic batch sizing for forward and backward passes, allowing for optimized and iterative weight updates during training.
* **Visual Evaluation:** Automatically generates a Loss vs. Epoch plot at the end of the training phase to evaluate model performance and robustly determine accuracy on unknown examples.
* **Data Splitting:** Dynamically splits the dataset into training and validation sets based on user-defined ratios.

## The Dataset

The model trains on a breast cancer dataset, predicting a diagnosis of either `M` (Malignant) or `B` (Benign). The features of the dataset describe various characteristics of cell nuclei. Variables include radius, texture, perimeter, area, smoothness, and more.

## Requirements

The project relies strictly on foundational Python libraries for math, data manipulation, and visualization:
* `numpy` (for linear algebra operations)
* `pandas` (for CSV data loading and manipulation)
* `matplotlib` (for displaying the learning curves)

## Project Structure

* `main.py`: The entry point for the program, handling arguments and routing to training or prediction modes.
* `./model/architecture.json`: Configuration file specifying the network topology (number of layers, nodes per layer, activation functions, etc.).
* `./model/parameters.json`: Storage file for the learned weights and biases after a successful training run.

## Usage

The program is executed via `main.py` and requires specifying the operational mode (`--training` or `--prediction`).

### Command Line Arguments

* `--training` / `--prediction`: **(Mandatory)** Defines the execution mode. 
* `--epochs <int>`: Number of training iterations. Default is `80`.
* `--learningRate <float>`: The step size for gradient descent. Default is `0.01`.
* `--validationRatio <float>`: The proportion of data used for training (between 0 and 1). Default is `0.7`.
* `--batchSize <int>`: The number of datapoints used in one forward/backward pass. Defaults to the full size of the training dataset.

### Examples

**Training the model:**
```bash
python3 main.py --training --epochs 70 --learningRate 0.5 --validationRatio 0.8

**Running predictions:**
(Note: Requires a previously trained model saved in the ./model/ directory)


```Bash
python3 main.py --prediction