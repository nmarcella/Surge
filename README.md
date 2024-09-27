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
