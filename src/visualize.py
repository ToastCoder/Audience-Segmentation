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
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, accuracy_score, confusion_matrix
import seaborn as sn

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

# OPENING THE TRAINED MODEL
model = tf.keras.models.load_model(MODEL_PATH)

y_pred = model.predict(x_test)
y_pred = np.array(y_pred).argmax(axis=1)
y_pred = y_pred.flatten()

# FINDING AND PLOTTING CONFUSION MATRIX
cm = confusion_matrix(y_test,y_pred)
sn.heatmap(cm, annot=True)

# FINDING SPECIFICITY AND SENSITIVITY
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sp = tn/(tn+fp)
sn = tp/(tp+fn)

# PRINTING THE METRICS
print('f1 score =  %.2f'%f1_score(y_test, y_pred))
print('Precision =  %.2f'%precision_score(y_test, y_pred))
print('Test accuracy =  %.2f'%accuracy_score(y_test, y_pred))
print('Specificity =  %.2f'%sp)
print('Sensitivity =  %.2f'%sn)
