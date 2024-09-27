# Surge
A repo for demonstrating the efficacy of NN surrogate models for XAS simulation

# Neural Network for Predicting EXAFS from RDF Data Using PyTorch

This repository contains code for training a neural network model to predict Extended X-ray Absorption Fine Structure (EXAFS) data from Radial Distribution Function (RDF) data using PyTorch. The model is designed to learn the complex mapping between RDF inputs and EXAFS outputs, which has applications in material science and computational chemistry.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Evaluation and Results](#evaluation-and-results)
  - [Euclidean Distance Plot](#euclidean-distance-plot)
  - [Histogram of Euclidean Distances](#histogram-of-euclidean-distances)
  - [Best and Worst Predictions](#best-and-worst-predictions)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

This project aims to build a neural network that can predict EXAFS spectra from RDF data. The EXAFS technique is widely used to determine the local structural information of materials, and being able to predict EXAFS from RDF can significantly accelerate material analysis.

The model utilizes convolutional neural networks (CNNs) to capture spatial hierarchies in the RDF data and fully connected layers to map the extracted features to the EXAFS outputs.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Dataset Preparation

The dataset consists of RDF and EXAFS examples stored in NumPy arrays:

```python
rdf_examples = rdf_array
exafs_examples = chi_array
```
We split the data into training and testing sets:
```python
from sklearn.model_selection import train_test_split
```
# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    rdf_examples, exafs_examples, test_size=0.20
)

## Model Architecture
The neural network model is defined using PyTorch's nn.Module. It consists of two 1D convolutional layers followed by three fully connected layers. The activation function used is hyperbolic tangent (Tanh).

```python
import torch
import torch.nn as nn

# Define the model
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=32, kernel_size=10, stride=2, padding=4
        )
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=10, stride=1, padding=4
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 39, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 201)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 240)    # Ensure input is of shape (batch_size, 240)
        x = x.view(-1, 3, 80)  # Reshape to (batch_size, 3, 80)
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x

# Instantiate the model
model_nano_Au = Model2().to(device)
print(model_nano_Au)  # Print the model summary
```