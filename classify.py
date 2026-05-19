'''
Name: Lucas Hasting
Description: Predict pathogenicity in VUS
Course: MA 395 - Deep Learning Independent Study/MA 360H - CODE
Instructors: Mark Terwilliger, Cynthia Stenger
Github: https://github.com/LucasHasting/Prediction-CODE
* Sources are included in the GitHub ReadME
'''

#import libraries used
import torch
from torch import nn
import torch.nn.functional as F
import pickle

#definition of the MLP
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

#load data frame using pickle
file = open('data/DATA_CLEANED_FULL.pkl', 'rb')
df = pickle.load(file)
file.close()

#get variables used for prediction
df = df[df["Clin. Sig."] == "VUS"]
X = df[["SIFT", "PolyPhen", "REVEL", "MetaLR", "Mutation Assessor"]]

#load the models
file = open('models/NN.pkl', 'rb')
nn_model = pickle.load(file)
file.close()

file = open('models/DT.pkl', 'rb')
dt = pickle.load(file)
file.close()

file = open('models/RF.pkl', 'rb')
rf = pickle.load(file)
file.close()

file = open('models/KNN.pkl', 'rb')
knn = pickle.load(file)
file.close()

#make predictions
df["NN"] = torch.round(nn_model(torch.tensor(X.values, dtype=torch.float32))).detach().numpy()
df["DT"] = dt.predict(X)
df["RF"] = rf.predict(X)
df["KNN"] = knn.predict(X)

#store updated VUS file with predictions
df.to_csv('data/predictions_VUS.csv', index=False)
print("Done")
