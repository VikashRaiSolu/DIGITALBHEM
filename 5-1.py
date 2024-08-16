import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing the dataset
dataset = pd.read_csv("G:\\codes\\6.csv")

# split the data into inputs and outputs
X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

# training and testing data
from sklearn.model_selection import train_test_split

# assign test data size 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# importing standard scaler
from sklearn.preprocessing import StandardScaler

# scaling the input data
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # Use transform instead of fit_transform on the test data

# importing classifier
from sklearn.naive_bayes import BernoulliNB

# import Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

# create a Gaussian Classifier

classifier1 = GaussianNB()

# training the model
classifier1.fit(X_train, y_train)

# testing the model
y_pred1 = classifier1.predict(X_test)

# importing accuracy score
from sklearn.metrics import accuracy_score

# printing the accuracy of the model
print("Accuracy of the model:", accuracy_score(y_test, y_pred1))
