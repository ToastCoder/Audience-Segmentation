#  AUDIENCE SEGMENTATION
# FILE NAME: test.py 

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR
# MOULISHANKAR M R 

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# DATA PREPROCESSING
data = pd.read_csv("data/custData.csv")
x = data.iloc[:,[4,5,6,7,8,9,10]].values
y = data.iloc[:,11].values

# OPENING THE TRAINED MODEL
model = load_model('./model/custDataModel')

# GETTING AND PROCESSING DATA FROM USER
interests = ["Technology","Politics","Food","Education","Media","Travel","Medicine"]
user_interests = []
for i in range(len(interests)):
    val = int(input("Amount of videos person watched which is related to "+interests[i]+": "))
    user_interests.append(val)

categories = ["Young Adult","Engineering Student","Medical Student","Teacher","Mature (40+)","Travelophilic","Media Addicted"]
res = model.predict([user_interests])
res = np.argmax(res, axis=None, out=None)
print("The person may be a "+categories[int(res)]+" type of audience.")

