import torch
from torch import nn
import torch.nn.functional as F
import pickle

class PathogenicityNN(nn.Module):
    def __init__(self):
        super(PathogenicityNN, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(5, 3)  # Input Layer -> Hidden Layer
        self.fc2 = nn.Linear(3, 1) # Hidden Layer -> Output layer

    def forward(self, x):
        # Define the forward pass with a ReLU activation function
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #apply the activation function
        x = F.sigmoid(x)
        return x

# Load the model
file = open('NN.pkl', 'rb')
model = pickle.load(file)
file.close()

for name, param in model.named_parameters():
    print(name, param.shape)

print(model.fc1.weight)
print(model.fc1.bias)

print(model.fc2.weight)
print(model.fc2.bias)
