'''
Name: Lucas Hasting
Description: Construct a MLP to predict pathogenicity
Course: MA 395 - Deep Learning Independent Study/MA 360H - CODE
Instructors: Mark Terwilliger, Cynthia Stenger
Github: https://github.com/LucasHasting/Prediction-CODE
* Sources are included in the GitHub ReadME
'''

#import libraries used
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import warnings

#ignore warnings
warnings.filterwarnings("ignore")

#pytorch MLP
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
file = open('data/DATA_CLEANED.pkl', 'rb')
df = pickle.load(file)
file.close()

#get non-VUS data, map to numerical values
df_nn = df[df["Clin. Sig."] != "VUS"]
df_nn["Clin. Sig."] = df["Clin. Sig."].map({"pathogenic": 1, "benign": 0})

#split data into test/training (1/3 - test, 2/3 - training)
X_train, X_test, y_train, y_test = train_test_split(df_nn[["SIFT", "PolyPhen", "REVEL", "MetaLR", "Mutation Assessor"]],
                                                    df_nn["Clin. Sig."], test_size=0.3, random_state=42)

#convert needed variables to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

#create data set object
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

#create loaders for object
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#instantiate the model
model = PathogenicityNN()
model_inv = PathogenicityNN()

#specify loss and optimizer function/algorithm
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#training Loop
num_epochs = 10000 #experimented with this value
for epoch in range(num_epochs):
    for inputs, targets in train_loader:        
        #xero the parameter gradients
        optimizer.zero_grad()

        #forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        #backward pass - compute gradients
        loss.backward()

        #optimizer step - update parameters
        optimizer.step()
    
    #display current epoch info 
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

'''
Use this to check an already existing NN Generalization Gap:
file = open('models/NN.pkl', 'rb')
model = pickle.load(file)
file.close()
'''

#get accuracy by testing on training, used to ensure not overfitted
output_predicted = torch.empty((0, 1))
output_true = torch.empty((0, 1))
difference = torch.empty((0, 1))
model.eval()
with torch.no_grad():
    for inputs, labels in train_loader:
        output_true = torch.cat((output_true, labels), dim=0)
        output_predicted = torch.cat((output_predicted, torch.round(model(inputs))), dim=0)

difference = torch.abs(output_true - output_predicted)
accuracy1 = 1 - torch.mean(difference)
print("NN Train Accuracy", accuracy1)

#get accuracy by testing on test, used to ensure not overfitted
output_predicted = torch.empty((0, 1))
output_true = torch.empty((0, 1))
difference = torch.empty((0, 1))
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        output_true = torch.cat((output_true, labels), dim=0)
        output_predicted = torch.cat((output_predicted, torch.round(model(inputs))), dim=0)

difference = torch.abs(output_true - output_predicted)
accuracy2 = 1 - torch.mean(difference)
print("NN Test Accuracy", accuracy2)

print("Generalization GAP = ", abs(accuracy1 - accuracy2))

#save model using pickle
file = open('models/NN.pkl', 'wb')
pickle.dump(model, file)
file.close()
