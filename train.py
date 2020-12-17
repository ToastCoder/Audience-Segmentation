#  AUDIENCE SEGMENTATION
# FILE NAME: train.py

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR
# MOULISHANKAR M R 

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model

# DATA SEGMENTATION
data = pd.read_csv("data/custData.csv")
x = data.iloc[:,[4,5,6,7,8,9,10]].values
y = data.iloc[:,11].values

# ENCODING LABELS
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = to_categorical(y)

# SPLITTING THE MAIN DATA INTO TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# DEFINING THE NEURAL NETWORK LAYERS AND THEIR NODES
model = Sequential()
model.add(Dense(14, input_dim = 7 , activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

# TRAINING THE MODEL
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
model.fit(x_train, y_train, epochs = 20, batch_size = 2)

# CALCULATING THE ESTIMATED ACCURACY
score = model.evaluate(x_val, y_val)
print(f"The estimated accuracy of the model is: {round(score[1]*100,4)}")

# SAVING THE MODEL
save_model(model,'./model/custDataModel')