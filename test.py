#  AUDIENCE SEGMENTATION USING LOGISTIC REGRESSION

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

# OPENING AND DEFINING THE TRAINED MODEL
log = pickle.load(open("model/model.pkl", 'rb'))

# GETTING AND PROCESSING DATA FROM USER
interests = ["Technology","Politics","Food","Education","Media","Travel","Medicine"]
user_interests = []
for i in range(len(interests)):
    ch = input("Is the person interested in viewing content related to "+interests[i]+"? (Y/N): ").upper()
    if ch == "Y" or "YES":
        user_interests.append(1)
    elif ch == "N" or "NO":
        user_interests.append(0)

categories = ["Young Adult","Engineering Student","Medical Student","Teacher","Mature (40+)","Travelophilic","Media Addicted"]
res = log.predict([user_interests])
print("The person may be a "+categories[int(res)]+" type of audience.")

