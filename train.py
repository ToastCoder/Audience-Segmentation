# AUDIENCE SEGMENTATION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Multiclass Classification, Machine Learning, TensorFlow

# DISABLE TENSORFLOW DEBUG INFORMATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("TensorFlow Debugging Information is hidden.")

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = 'data/custData.csv'
MODEL_PATH = './model/custDataModel'
ACC_THRESHOLD = 0.99

# DATA SEGMENTATION
data = pd.read_csv(DATASET_PATH)
print(data.describe())

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

class Callback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > ACC_THRESHOLD):   
            print("Reached Threshold Accuracy, Stopping Training.")   
            self.model.stop_training = True

model = custDataModel()
callback = Callback()

# TRAINING THE MODEL
model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
history = model.fit(x_train, y_train, epochs = 20, batch_size = 2,callbacks = [callback])

# PLOTTING THE GRAPH FOR TRAIN-LOSS AND VALIDATION-LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# CALCULATING THE ESTIMATED ACCURACY
score = model.evaluate(x_val, y_val)
print(f"Model Accuracy: {round(score[1]*100,4)}")

# SAVING THE MODEL
tf.keras.models.save_model(model,'./model/custDataModel')