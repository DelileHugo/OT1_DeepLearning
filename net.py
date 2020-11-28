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
epochs = 300
end_fc1 = 64
end_fc2 = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(294912, end_fc1)
        self.fc2 = nn.Linear(end_fc1, end_fc2)
        self.fc3 = nn.Linear(end_fc2, 9)
        self.fcbb1 = nn.Linear(end_fc1, end_fc2)
        self.fcbb2 = nn.Linear(end_fc2, 4*9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        y = self.fcbb1(x)
        x = self.fc2(x)
        x = F.relu(x)
        y = F.relu(y)
        y = self.fcbb2(y)
        x = self.fc3(x)
        return x, y
#weight=torch.tensor([1,2])

def train(model, train_loader, optimizer, epoch):
    model.share_memory()
    model.train()
    correct = 0
    for batch_idx, (data, target, bounding_box) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to("cuda")
        target = target.to("cuda")
        bounding_box = bounding_box.to("cuda")
        outcol, outcoor = model(data)
        
        bce = torch.nn.BCEWithLogitsLoss()
        #bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.add(torch.flatten(target), 1)) # try without weight
        losscol = bce(torch.flatten(outcol), torch.flatten(target).float())
        mse = torch.nn.MSELoss()
        losscoor = mse(torch.flatten(outcoor), torch.flatten(bounding_box).float())
        
        sumloss = losscol + losscoor
        sumloss.backward()
        optimizer.step()
        pred = torch.sigmoid(outcol) > 0.5  # get the index of the max log-probability
        correct += pred.eq(target).sum().item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tColor loss: {:.6f}\tCoor. loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), losscol.item(), losscoor.item()))

    correct /= 9
    print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))


def test(model, test_loader):
    model.eval()
    test_loss_col = 0
    test_loss_coor = 0
    correct_col = 0
    correct_coor = 0
    with torch.no_grad():
        for data, target, bounding_box in test_loader:
            data = data.to("cuda")
            outcol, outcoor = model(data)
            target = target.to("cuda")
            bounding_box = bounding_box.to("cuda")
            #critcol = torch.nn.BCEWithLogitsLoss(pos_weight=torch.add(torch.flatten(target), 1))
            critcol = torch.nn.BCEWithLogitsLoss()
            test_loss_col += critcol(torch.flatten(outcol), torch.flatten(target).float()).item()

            crit_coor = torch.nn.MSELoss()
            test_loss_coor += crit_coor(torch.flatten(outcoor), torch.flatten(bounding_box).float()).item()
            
            outcoor = torch.reshape(outcoor, (100, 9, 4))

            pred_coor = torch.abs(bounding_box - outcoor)
            
            vector_to_check = pred_coor[target == 1]
            print(vector_to_check[0])
            print((vector_to_check.max()).max())
            correct_coor += (vector_to_check.sum(dim=1)).sum().item()
            
            pred_col = torch.sigmoid(outcol) > 0.5  # get the index of the max log-probability
            correct_col += pred_col.eq(target).sum().item()
    
    correct_col /= 9.0
    correct_coor /= (12*4200)
    test_loss_col /= len(test_loader.dataset)

    print('\nTest set: Average color loss: {:.4f}, Average coor. loss: {:.4f}'.format(test_loss_col, correct_col))
    print('Accuracy col: {}/{} ({:.0f}%)'.format(correct_col, len(test_loader.dataset), 100.0 * correct_col / len(test_loader.dataset)))
    print('Accuracy coor: {}/{} ({:.0f}%)\n'.format(correct_coor, len(test_loader.dataset), 100.0 * correct_coor / len(test_loader.dataset)))


def main():
    torch.seed()

    dataset_train = Balls_CF_Detection("data\\train\\train\\")
    dataset_test = Balls_CF_Detection("data\\train\\val\\")

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size)

    model = Net().to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.0025)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
    for epoch in range(1, epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

if __name__ == '__main__':
    main()

