# AUDIENCE SEGMENTATION USING LOGISTIC REGRESSION
# FILE NAME: visualize.py

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR
# MOULISHANKAR M R 

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# DATA PREPROCESSING
data = pd.read_csv("data/custData.csv")
x = data.iloc[:,[3,4,5,6,7,8,9]].values
y = data.iloc[:,-1].values

# SPLITTING THE DATA INTO TEST DATA AND TRAIN DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# FEATURE SCALING
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# CREATING A MODEL AND FITTING DATA TO MODEL
log = LogisticRegression()
log.fit(x_train,y_train)

# PREDICTING THE RESULTS FOR TEST DATA
y_pred = log.predict(x_test)

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print(cm)

