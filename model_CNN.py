#Imports
import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
#Create fully connected network
class NN(nn.Module):
    def __init__(self, in_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#TODO Create a simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
in_channels = 1
num_classes = 10
epochs = 50
batch_size = 12
learning_rate = 1e-4
#Load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=T.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=T.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
#Initialize network
model = CNN().to(device)
#Loss and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = op.Adam(model.parameters(), lr=learning_rate)
#Train network
for epoch in range(epochs):
    for ite, (data, targets) in enumerate(train_loader):
        #Get data if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        #Get to correct shape
        #data = data.reshape(data.shape[0], -1)

        #forward
        score = model(data)
        loss = criteria(score, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or Adam step
        optimizer.step()

#Check accuracy on training & test see how good your model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, prediction = score.max(1)
            #y_max = torch.argmax(y, dim=0)
            #prediction_max = torch.argmax(prediction, dim=0)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {(float(num_correct)/float(num_samples))*100: .2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)