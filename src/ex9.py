import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def Loss(self, Scores, target):
        y = F.softmax(Scores, dim=1)
        err = self.criterion(y, target)
        return err

    def TestOK(self, Scores, target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        pred = pred.reshape(target.shape)
        eq = pred == target  # True when correct prediction
        nbOK = eq.sum().item()  # count
        return nbOK


class Net1(Net):
    def __init__(self):
        super(Net1, self).__init__()
        self.Conv = nn.Conv2d(3, 32, (3, 3), stride=(1, 1), padding=1)
        self.FC = nn.Linear(32 * 32 * 32, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        n = x.shape[0]
        x = self.Conv(x)
        x = F.relu(x)
        x = x.reshape((n, 32 * 32 * 32))
        x = self.FC(x)
        return x


class Net2(Net):
    def __init__(self):
        super(Net2, self).__init__()
        self.Conv1 = nn.Conv2d(3, 32, (3, 3), stride=(1, 1), padding=1)
        self.Conv2 = nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1)
        self.FC = nn.Linear(64 * 32 * 32, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        n = x.shape[0]
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = x.reshape((n, 64 * 32 * 32))
        x = self.FC(x)
        return x


##############################################################################

def TRAIN(args, model, train_loader, optimizer, epoch):
    for batch_it, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        Scores = model.forward(data)
        loss = model.Loss(Scores, target)
        loss.backward()
        optimizer.step()

        if batch_it % 50 == 0:
            print(f'   It: {batch_it:3}/{len(train_loader):3} --- Loss: {loss.item():.6f}')


def TEST(model, test_loader):
    ErrTot = 0
    nbOK = 0
    nbImages = 0

    with torch.no_grad():
        for data, target in test_loader:
            Scores = model.forward(data)
            nbOK += model.TestOK(Scores, target)
            ErrTot += model.Loss(Scores, target)
            nbImages += data.shape[0]

    pc_success = 100. * nbOK / nbImages
    print(f'\nTest set:   Accuracy: {nbOK}/{nbImages} ({pc_success:.2f}%)\n')
    return pc_success


##############################################################################

def main(batch_size):
    TRS = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    TrainSet = datasets.CIFAR10('./data', train=True, download=True, transform=TRS)
    TestSet = datasets.CIFAR10('./data', train=False, download=True, transform=TRS)

    train_loader = torch.utils.data.DataLoader(TrainSet, batch_size)
    test_loader = torch.utils.data.DataLoader(TestSet, len(TestSet))

    model = Net2()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    results = [TEST(model, test_loader)]
    import json
    for epoch in range(120):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        TRAIN(batch_size, model, train_loader, optimizer, epoch)
        results.append(TEST(model, test_loader))
        with open('results.json', 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main(batch_size=64)
