#-------------------------------------------------------------------------------------------------------------------------------

# AUDIENCE SEGMENTATION

# FILE NAME: visualize.py

# DEVELOPED BY: Vigneshwar Ravichandar, Moulishankar M R

# TOPICS: Multiclass Classification, Machine Learning, TensorFlow

#-------------------------------------------------------------------------------------------------------------------------------

# IMPORTING REQUIRED MODULES
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = 'data/custData.csv'
MODEL_PATH = './model/custDataModel'

# DATA SEGMENTATION
data = pd.read_csv(DATASET_PATH)
print("Dataset Description:\n",data.describe())
print("Dataset Head\n",data.head())

x = data.iloc[:,[4,5,6,7,8,9,10]].values
y = data.iloc[:,11].values

# SPLITTING THE MAIN DATA INTO TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# OPENING THE TRAINED MODEL
model = tf.keras.models.load_model(MODEL_PATH)

# CALCULATING THE ACCURACY
score = model.evaluate(x_val, y_val)
print(f"Model Accuracy: {round(score[1]*100,4)}")