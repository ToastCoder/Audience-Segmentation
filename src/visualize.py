# AUDIENCE SEGMENTATION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar, Moulishankar M R

# TOPICS: Multiclass Classification, Machine Learning, TensorFlow
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

os.system('cd ..')

DATASET_PATH = 'data/custData.csv'
MODEL_PATH = './model/custDataModel'