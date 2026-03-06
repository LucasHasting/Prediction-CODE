#used to test models and display their confusion matrices
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#include data wrangling library
import pandas as pd

#include plotting library
import matplotlib.pyplot as plt

#include model libraries
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#include test and accuracy libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

import pickle

class PathogenicityNN(nn.Module):
    def __init__(self):
        super(PathogenicityNN, self).__init__()
        #define the layers
        self.fc1 = nn.Linear(5, 3) #input Layer  -> hidden Layer
        self.fc2 = nn.Linear(3, 1) #hidden Layer -> output layer

    def forward(self, x):
        #define the forward pass activation function -> ReLU 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #apply the output layer activation function -> sigmoid
        x = F.sigmoid(x)
        return x

#load data frame using pickle, and convert to tensor
file = open('DATA_CLEANED_FULL.pkl', 'rb')
df = pickle.load(file)
file.close()
df = df[df["Clin. Sig."] != "VUS"]

X = torch.tensor(df[["SIFT", "PolyPhen", "REVEL", "MetaLR", "Mutation Assessor"]].values, dtype=torch.float32)

#load the models
file = open('NN.pkl', 'rb')
nn_model = pickle.load(file)
file.close()

file = open('DT.pkl', 'rb')
dt = pickle.load(file)
file.close()

file = open('RF.pkl', 'rb')
rf = pickle.load(file)
file.close()

file = open('KNN.pkl', 'rb')
knn = pickle.load(file)
file.close()

#test NN
X_test = df[["SIFT", "PolyPhen", "REVEL", "MetaLR", "Mutation Assessor"]]
y_test = df["Clin. Sig."].map({"pathogenic": 1, "benign": 0})

X = torch.tensor(X_test.values, dtype=torch.float32)
y = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)
data = DataLoader(dataset, batch_size=32, shuffle=True)

#test loop for NN
output_predicted = torch.empty((0, 1))
output_true = torch.empty((0, 1))
difference = torch.empty((0, 1))
nn_model.eval()
with torch.no_grad():
    for inputs, labels in data:
        output_true = torch.cat((output_true, labels), dim=0)
        output_predicted = torch.cat((output_predicted, torch.round(nn_model(inputs))), dim=0)

difference = torch.abs(output_true - output_predicted)
print("NN Accuracy", 1 - torch.mean(difference))

y_pred = output_predicted.detach().numpy()

#display confusion matrix
fig, ax = plt.subplots(figsize=(200, 200))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=sorted(y.unique()))
disp.plot(cmap=plt.cm.Reds,ax=ax)
disp.ax_.set_xticks([])
plt.title('Figure 4: NN - Confusion Matrix')
plt.show()

#Test DT
y_pred = dt.predict(X_test) # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

#display confusion matrix
fig, ax = plt.subplots(figsize=(200, 200))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=sorted(y.unique()))
disp.plot(cmap=plt.cm.Blues,ax=ax)
disp.ax_.set_xticks([])
plt.title('Figure 3: Decision Tree - Confusion Matrix')
plt.show()

#display model accuracy
print(f"DT Accuracy: {accuracy:.2f}")
print()

#Test RM
y_pred = rf.predict(X_test) # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

#display confusion matrix
fig, ax = plt.subplots(figsize=(200, 200))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=sorted(y.unique()))
disp.plot(cmap=plt.cm.Blues,ax=ax)
disp.ax_.set_xticks([])
plt.title('Figure 3: Random Forest - Confusion Matrix')
plt.show()

#display model accuracy
print(f"RF Accuracy: {accuracy:.2f}")
print()

#Test KNN
y_pred = knn.predict(X_test) # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

#display confusion matrix
fig, ax = plt.subplots(figsize=(200, 200))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=sorted(y.unique()))
disp.plot(cmap=plt.cm.Reds,ax=ax)
disp.ax_.set_xticks([])
plt.title('Figure 4: k-NN - Confusion Matrix')
plt.show()

#display model accuracy
print(f"KNN Accuracy: {accuracy:.2f}")
print()

