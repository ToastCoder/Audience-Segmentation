# AUDIENCE SEGMENTATION

# FILE NAME: test.py

# DEVELOPED BY: Vigneshwar Ravichandar, Moulishankar M R

# TOPICS: Multiclass Classification, Machine Learning, TensorFlow

# IMPORTING REQUIRED MODULES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

os.system('cd ..')

DATASET_PATH = 'data/custData.csv'
MODEL_PATH = './model/custDataModel'

# DATA PREPROCESSING
data = pd.read_csv(DATASET_PATH)
print(data.describe())

x = data.iloc[:,[4,5,6,7,8,9,10]].values
y = data.iloc[:,11].values

# OPENING THE TRAINED MODEL
model = tf.keras.models.load_model(MODEL_PATH)

# GETTING AND PROCESSING DATA FROM USER
interests = ["Technology","Politics","Food","Education","Media","Travel","Medicine"]
user_interests = []
for i in range(len(interests)):
    val = int(input("Amount of videos person watched which is related to "+interests[i]+": "))
    user_interests.append(val)

categories = ["Young Adult","Engineering Student","Medical Student","Teacher","Mature (40+)","Travelophilic","Media Addicted"]
res = model.predict_classes([user_interests])
print("The person may be a "+categories[int(res)]+" type of audience.")

