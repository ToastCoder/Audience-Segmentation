#  CUSTOMER SEGMENTATION USING LOGISTIC REGRESSION

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR
# MOULISHANKAR M R 

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

# DATA PREPROCESSING
data = pd.read_csv("data/custData.csv")
x = data.iloc[:,[3,4,5,6,7,8,9]].values
y = data.iloc[:,-1].values

# INITIALIZING AND TRAINING THE MODEL WITH THE SEGMENTED DATA
log = LogisticRegression(random_state = 0)
log.fit(x,y)

# SAVING THE MODEL TO A FILE
pickle.dump(log,open("model/model.pkl","wb"))