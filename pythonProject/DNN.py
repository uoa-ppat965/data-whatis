# Import Files
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.utils import shuffle
from torch.autograd import Variable

# Download training data
iris = load_iris()
X, y = iris.data, iris.target

# Split the Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

# Look at data
print('All:', np.bincount(y) / float(len(y)) * 100.0)
print('Training:', np.bincount(Y_train) / float(len(Y_train)) * 100.0)
print('Testing:', np.bincount(Y_test) / float(len(Y_test)) * 100.0)
batch_size = 64

# This gives the number of columns , used below.
print(X_train.shape[1])

# This gives the number of classes. used below
print(len(np.unique(Y_train)))

# Define training hyperparameters
batch_size = 60
num_epochs = 500
learning_rate = 0.01
size_hidden = 100

# Calculate some other hyperparameters based on data
batch_no = len(X_train) // batch_size  # batches
cols = X_train.shape[1]  # Number of columns in input matrix
classes = len(np.unique(Y_train))

# Get cpu, gpu or mps device for training.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print("Executing the model on :", device)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, cols, size_hidden, classes):
        super(NeuralNetwork, self).__init__()
        # Note that 17 is the number of columns in the input matrix.
        self.fc1 = nn.Linear(cols, size_hidden)
        # variety of # possible for hidden layer size is arbitrary
        self.fc2 = nn.Linear(size_hidden, classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


net = NeuralNetwork(cols, size_hidden, classes)

# Optimising the Model Parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Looking at the loss
running_loss = 0

# Training
for epoch in range(num_epochs):
    # Shuffle just mixes up the dataset
    X_train, Y_train = shuffle(X_train, Y_train)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        inputs = Variable(torch.FloatTensor(X_train[start:end]))
        labels = Variable(torch.LongTensor(Y_train[start:end]))
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print Statistics
        running_loss += loss.item()

    print('Epoch {}'.format(epoch + 1), "loss: ", running_loss)
    running_loss = 0.0


# This is a little bit tricky to get the resulting prediction.
def calculate_accuracy(x, y=[]):
    """
    This function will return the accuracy if passed x and y or return predictions if just passed x.
    """
    # Evaluate the model with the test set.
    X = Variable(torch.FloatTensor(x))
    result = net(X)  # This outputs the probability for each class.
    _, labels = torch.max(result.data, 1)
    if len(y) != 0:
        num_right = np.sum(labels.data.numpy() == y)
        print('Accuracy {:.4f}'.format(num_right / len(y)), "for a total of ", len(y), "records")
        return pd.DataFrame(data={'actual': y, 'predicted': labels.data.numpy()})
    else:
        print("returning predictions")
        return labels.data.numpy()


result1 = calculate_accuracy(X_train, Y_train)
result2 = calculate_accuracy(X_test, Y_test)

