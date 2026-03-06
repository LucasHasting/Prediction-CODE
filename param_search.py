#Name:          Lucas Hasting, Joseph Pope
#Description:   Use machine learning to predict pathogenicty
#               -> Parameter Search for models.py
#               https://scikit-learn.org/stable/api/index.html
#               https://mlbenchmarks.org/04-holdout-method.html

#include data wrangling library
import pandas as pd

#include model libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#include test and accuracy libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

#threshold for generalization gap
GG_THRESHOLD = 0.015

#get data from csv
file = open('DATA_CLEANED.pkl', 'rb')
df = pickle.load(file)
file.close()

#split data into dependent/independent variables
df = df[df["Clin. Sig."] != "VUS"]
df["Clin. Sig."] = df["Clin. Sig."].map({"pathogenic": 1, "benign": 0})
y = df["Clin. Sig."]
X = df[["SIFT", "PolyPhen", "REVEL", "MetaLR", "Mutation Assessor"]]

#split data into test/training (1/3 - test, 2/3 - training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Find best parameter: max_depth
train_accuracy = 1
test_accuracy = 0
param = 100 #good starting point

while(abs(train_accuracy - test_accuracy) > GG_THRESHOLD or param == 0):
    clf = DecisionTreeClassifier(max_depth=param) # Initialize the classifier
    clf.fit(X_train, y_train) # Train the classifier
    y_pred = clf.predict(X_test) # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

    #check for overfitting using the generalization gap
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"param: {param}, acc: {abs(train_accuracy - test_accuracy)} = {train_accuracy} - {test_accuracy}")
    param -= 1 #adjust parameter

print(f"Best max_depth: {param+1}\n")

#find best parameter: n_estimators
train_accuracy = 1
test_accuracy = 0
old_param = param
param = 1 #good starting point

while(abs(train_accuracy - test_accuracy) > GG_THRESHOLD or param == 0):
    rf = RandomForestClassifier(n_estimators=param,max_depth=old_param) # Initialize the classifier
    rf.fit(X_train, y_train) # Train the classifier
    y_pred = rf.predict(X_test) # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

    #check for overfitting using the generalization gap
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"param: {param}, acc: {abs(train_accuracy - test_accuracy)} = {train_accuracy} - {test_accuracy}")
    param += 1 #adjust parameter

print(f"Best n_estimators: {param-1}\n")

#Find best parameter: k
train_accuracy = 1
test_accuracy = 0
param = 100 #good starting point

while(abs(train_accuracy - test_accuracy) > GG_THRESHOLD or param == 100):
    knn = KNeighborsClassifier(n_neighbors=param,metric='euclidean') # Initialize the classifier
    knn.fit(X_train, y_train) # Train the classifier
    y_pred = knn.predict(X_test) # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

    #check for overfitting using the generalization gap
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"param: {param}, acc: {abs(train_accuracy - test_accuracy)} = {train_accuracy} - {test_accuracy}")
    param -= 2 #adjust parameter

print(f"Best k: {param + 2}")
