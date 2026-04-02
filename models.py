'''
Name: Lucas Hasting
Description: Construct models to predict pathogenicity using
             a Decision Tree, Random Forest, KNN
Course: MA 395 - Deep Learning Independent Study/MA 360H - CODE
Instructors: Mark Terwilliger, Cynthia Stenger
Github: https://github.com/LucasHasting/Prediction-CODE
* Sources are included in the GitHub ReadME
'''

#include data wrangling library
import pandas as pd

#include model libraries
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#include pickle for persistent objects
import pickle

#get data from file
file = open('data/DATA_CLEANED.pkl', 'rb')
df = pickle.load(file)
file.close()

#split data into dependent/independent variables, not VUS, map to numerical values
df = df[df["Clin. Sig."] != "VUS"]
df["Clin. Sig."] = df["Clin. Sig."].map({"pathogenic": 1, "benign": 0})
y = df["Clin. Sig."]
X = df[["SIFT", "PolyPhen", "REVEL", "MetaLR", "Mutation Assessor"]]

#split data into test/training (1/3 - test, 2/3 - training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#get count of train/test:
print(f"size of train: {len(X_train)}, size of test: {len(X_test)}")

#Found max_depth in separate program
clf = DecisionTreeClassifier(max_depth=4) # Initialize the classifier
clf.fit(X_train, y_train) # Train the classifier

#Found max_depth, n_estimators in separate program
rf = RandomForestClassifier(n_estimators=4, max_depth=4) # Initialize the classifier
rf.fit(X_train, y_train) # Train the classifier

#found K in separate program
knn = KNeighborsClassifier(n_neighbors=10,metric='euclidean') # Initialize the classifier
knn.fit(X_train, y_train) # Train the classifier

#save models
file = open('models/DT.pkl', 'wb')
pickle.dump(clf, file)
file.close()

file = open('models/RM.pkl', 'wb')
pickle.dump(rf, file)
file.close()

file = open('models/KNN.pkl', 'wb')
pickle.dump(knn, file)
file.close()

print("Done")
