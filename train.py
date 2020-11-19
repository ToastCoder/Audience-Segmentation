#  AUDIENCE SEGMENTATION
# FILE NAME: train.py

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR
# MOULISHANKAR M R 

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# DATA PREPROCESSING
data = pd.read_csv("data/custData2.csv")
x = data.iloc[:,[4,5,6,7,8,9,10]].values
y = data.iloc[:,-1].values

# INITIALIZING AND TRAINING THE MODEL WITH THE SEGMENTED DATA
log = LogisticRegression(random_state = 0)
log.fit(x,y)

# SAVING THE MODEL TO A FILE
pickle.dump(log,open("model/model.pkl","wb"))
