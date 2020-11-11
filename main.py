# 0 - Young adult
# 1 - Engineering Student
# 2 - Medical Student
# 3 - Teachers
# 4 - Adults
# 5 - Travelling kinda person
# 6 - Media Freak

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# DATA PREPROCESSING
data = pd.read_csv("data/custdata2.csv")
x = data.iloc[:,3:9].values
y = data.iloc[:,-1].values

sc = StandardScaler()
x = sc.fit_transform(x)

classifier = LogisticRegression(random_state = 0)
classifier.fit(x,y)
s = np.array([1,0,1,1,0,1,0])
s = s.reshape(len(s),1)

res = classifier.predict(sc.fit_transform(s))
print(res)