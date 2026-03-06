import torch
from torch import nn
import torch.nn.functional as F
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

#load the model
file = open('NN.pkl', 'rb')
model = pickle.load(file)
file.close()

for name, param in model.named_parameters():
    print(name, param.shape)

print(model.fc1.weight)
print(model.fc1.bias)

print(model.fc2.weight)
print(model.fc2.bias)
