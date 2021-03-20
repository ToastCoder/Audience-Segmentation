#-------------------------------------------------------------------------------------------------------------------------------

# AUDIENCE SEGMENTATION

# FILE NAME: train.py

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
import argparse

# FUNCTION FOR PARSING ARGUMENTS
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs',
                        type = int, 
                        default = 50, 
                        required = False)

    parser.add_argument('-bs', '--batch_size',
                        type = int, 
                        default = 5, 
                        required = False)

    parser.add_argument('-l','--loss',
                        type = str, 
                        default = 'sparse_categorical_crossentropy', 
                        required = False)

    parser.add_argument('-op','--optimizer', 
                        type = str, 
                        default = 'adam', 
                        required = False)

    args = parser.parse_args()
    return args

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = 'data/custData.csv'
MODEL_PATH = './model/custDataModel'
args = parse()

# DATA SEGMENTATION
data = pd.read_csv(DATASET_PATH)
print("Dataset Description:\n",data.describe())
print("Dataset Head\n",data.head())

x = data.iloc[:,[4,5,6,7,8,9,10]].values
y = data.iloc[:,11].values

# SPLITTING THE MAIN DATA INTO TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# DEFINING THE NEURAL NETWORK FUNCTION
def custDataModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(14, input_dim = 7 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(7, activation = 'softmax'))
    return model

# INITITIALIZING THE CALLBACK
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy', mode = 'max')

model = custDataModel()

# TRAINING THE MODEL
model.compile(loss = args.loss , optimizer = args.optimizer , metrics = ['accuracy'] )
history = model.fit(x_train, y_train, epochs = args.epochs, batch_size = args.batch_size, callbacks = early_stopping, validation_data = (x_val, y_val))

# PLOTTING THE GRAPH FOR TRAIN-LOSS AND VALIDATION-LOSS
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
plt.show()
plt.savefig('graphs/loss_graph.png')

# PLOTTING THE GRAPH FOR TRAIN-ACCURACY AND VALIDATION-ACCURACY
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
plt.show()
plt.savefig('graphs/acc_graph.png')

# CALCULATING THE ACCURACY
score = model.evaluate(x_val, y_val)
print(f"Model Accuracy: {round(score[1]*100,4)}")

# SAVING THE MODEL
tf.keras.models.save_model(model,MODEL_PATH)