
# coding: utf-8

# Import all the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# Loading the MNIST dataset in PyTorch
train_dataset = datasets.MNIST(root='./', train = True, 
                               transform=transforms.ToTensor(),
                               download = True)
test_dataset = datasets.MNIST(root='./', train = False, 
                               transform=transforms.ToTensor(),
                               download = True)

# Print the sizes of set
print('Train dataset size: {}\n'.format(len(train_dataset)))
print('Test dataset size: {}\n'.format(len(test_dataset)))



#Setting the batch Size
batch_size = 128

# Initializing the data loader

train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


# Defining the structure of neetwork
model = nn.Sequential(
          nn.Linear(28*28, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 512),
          nn.ReLU(),
          nn.Linear(512, 10),
        )


# Defining the loss function, it also adds the softmax at the output layer - https://discuss.pytorch.org/t/why-does-crossentropyloss-include-the-softmax-function/4420
loss_fn = nn.CrossEntropyLoss()


# Defining the optimization algorithm

optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
# variable to track accuracy
correct = 0


# Start training the network now
model.train()

for epoch in range(0, 10):
    correct = 0
    for batch_idx, (data, target) in enumerate(train_data_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = loss_fn(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
                epoch, loss.item(), correct, len(train_data_loader.dataset),
                100. * float(correct) / len(train_data_loader.dataset)))
           



# Time to test the performance on test set using the above model
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_data_loader:
        data, target = Variable(data), Variable(target)
        output = model(data.view(-1, 784))
        # sum up batch loss
        test_loss += loss_fn(output, target).item()
        # get the index of the max - data loader does the one hot encoding
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()    



test_loss /= len(test_data_loader.dataset)
print('\nTest set Results: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_data_loader.dataset),
    100. * float(correct) / len(test_data_loader.dataset)))


