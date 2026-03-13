# Multilayer Perceptron 🧠

[cite_start]This repository contains a custom-built Artificial Neural Network (Multilayer Perceptron) designed to classify breast cancer tumors as malignant or benign[cite: 13]. [cite_start]Developed as part of the 42 Barcelona core curriculum, this project focuses on understanding the underlying math and logic of machine learning by building the core algorithms completely from scratch[cite: 91, 98, 99].

## Features

* [cite_start]**Built from Scratch:** No machine learning frameworks (like TensorFlow or PyTorch) were used[cite: 98]. [cite_start]The forward pass, backpropagation, and gradient descent algorithms were implemented entirely by hand using fundamental math[cite: 98, 99, 193].
* **Customizable Architecture:** The network's topology—including the number of layers, neurons per layer, activation functions, and loss functions—is completely modular and defined via a local `./model/architecture.json` file.
* **Mini-Batch Gradient Descent:** Supports dynamic batch sizing for optimized, iterative weight updates during training.
* [cite_start]**Data Splitting:** Automatically splits the dataset into training and validation sets based on user-defined ratios to test the model's accuracy on unknown examples[cite: 125, 147].
* [cite_start]**Visual Evaluation:** Generates a Loss vs. Epoch plot using Matplotlib at the end of the training phase to easily evaluate the model's learning curve[cite: 148].

## The Dataset

[cite_start]The model trains on the Wisconsin Breast Cancer dataset[cite: 13]. [cite_start]It analyzes features describing the characteristics of cell nuclei (such as radius, texture, perimeter, area, and smoothness) to predict a diagnosis label of either `M` (Malignant) or `B` (Benign)[cite: 119, 120, 121].

## Tech Stack

The project relies strictly on foundational Python libraries:
* **NumPy:** For matrix operations and linear algebra.
* **Pandas:** For loading and manipulating the CSV dataset.
* [cite_start]**Matplotlib:** For rendering the learning curves[cite: 99].

## Usage

The main interface is run through `main.py` and requires specifying the operational mode (`--training` or `--prediction`).

### Command Line Arguments

* `--training` / `--prediction`: **(Mandatory)** Defines the execution mode. 
* `--epochs <int>`: Number of training iterations over the dataset (Default: `80`).
* `--learningRate <float>`: The step size for the gradient descent (Default: `0.01`).
* `--validationRatio <float>`: The proportion of data used for training vs. validation, between 0 and 1 (Default: `0.7`).
* `--batchSize <int>`: The number of datapoints processed in one forward/backward pass (Default: Full dataset).

### Examples

**Training the model:**
```bash
python3 main.py --training --epochs 70 --learningRate 0.5 --validationRatio 0.8
```

**Running predictions:**
(Note: This mode loads the learned weights and biases from the ./model/parameters.json file generated during training.)

```bash
python3 main.py --prediction
```