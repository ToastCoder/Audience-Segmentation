# CUSTOMER SEGMENTATION USING LOGISTIC REGRESSION

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# DATA PREPROCESSING
data = pd.read_csv("data/CustData.csv")
x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
