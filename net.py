import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import glob
import numpy as np
import math
from dataset_det import Balls_CF_Detection, COLORS

batch_size = 100
test_batch_size = 100
epochs = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 2560, 3, 1)
        self.conv2 = nn.Conv2d(2560, 1280, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128000, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = F.relu(x)
        #print(x.size())
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        print(x.size())
        x = self.fc1(x)
        # print(x.size())
        x = F.relu(x)
        x = self.dropout2(x)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x
#weight=torch.tensor([1,2])

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, bounding_box) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        crit = torch.nn.BCEWithLogitsLoss()
        loss = crit(torch.flatten(output), torch.flatten(target).float())
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    torch.manual_seed(0)

    dataset_train = Balls_CF_Detection("data\\train\\train\\")
    dataset_test = Balls_CF_Detection("data\\train\\val\\")

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size)

    model = Net()
    optimizer = optim.Adadelta(model.parameters())

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
    for epoch in range(1, epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

'''   if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
'''

if __name__ == '__main__':
    main()

