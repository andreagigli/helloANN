# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # converts categorical in binary
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(drop = "first") # avoids the "dummy variable trap" (multicollinearity)
X = np.hstack((X[:,0].reshape([-1,1]), onehotencoder.fit_transform(X[:,1].reshape([-1,1])).toarray(), X[:,2::]))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


### Create the ANN ###
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Rule of thumb: the dim of the central layer should be around the mean of input and outplut layer dimension
classifier.add(Dense(units = 6, kernel_initializer ="uniform", activation = "relu", input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer ="uniform", activation = "relu"))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer= "uniform", activation = "sigmoid"))

# Compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test) > 0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Exercise: predict if X is leaving the bank
xnew = np.array([600,0,0,1,40,3,60000,2,1,1,50000]).reshape([1,-1])
xnew = sc.transform(xnew)
yprednew = classifier.predict(xnew) > 0.5
