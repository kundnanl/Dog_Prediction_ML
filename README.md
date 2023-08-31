# Dog Breed Classification Model README

This repository contains code for a dog breed classification model using TensorFlow and scikit-learn. The model takes features like weight, height, ear shape, and fur color as inputs and predicts the breed of the dog.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Prediction](#prediction)
- [License](#license)

## Overview

The project consists of the following components:

1. `model.py`: This script contains the code to train a dog breed classification model using TensorFlow. It uses a deep neural network architecture to learn and predict the dog breed from the provided features.

2. `preprocessing.py`: This script handles the preprocessing of the data. It converts categorical features like ear shape and fur color into numerical inputs using label encoding and scales the features using scikit-learn's StandardScaler.

3. `predict.py`: This script demonstrates how to use the trained model to make predictions on new data. It takes a sample prediction input, preprocesses it, scales it, and then uses the trained model to predict the dog breed.

## Installation

To run the scripts in this project, you need to have the following dependencies installed:

- Python (>=3.6)
- TensorFlow
- scikit-learn
- pandas

You can install the required packages using the following command:

```bash
pip install tensorflow scikit-learn pandas

Usage:
Clone this repository to your local machine:
git clone https://github.com/kundnanl/dog-breed-classification.git
cd dog-breed-classification

Run the training script to train the dog breed classification model:
python model.py

Once the model is trained, you can use the following steps to make predictions:
Load the pretrained model and scaler in the predict.py script.
Run the prediction script:
python predict.py

Preprocessing
The preprocessing of the data is done using the preprocessing.py script. It converts categorical features into numerical inputs using label encoding and scales the features using scikit-learn's StandardScaler. The encoded features are then used for training and prediction.

Prediction
The predict.py script demonstrates how to use the trained model to make predictions on new data. It shows how to preprocess the prediction input, scale it using the saved scaler, and then use the model to predict the dog breed.

