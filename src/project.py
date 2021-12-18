import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def Loss(self, scores, target):
        y = F.softmax(scores, dim=1)
        err = self.criterion(y, target)
        return err

    def Test_OK(self, scores, target):
        predictions = scores.argmax(dim=1, keepdim=True)  # get the index of the max
        predictions = predictions.reshape(target.shape)
        eq = predictions == target  # True when correct prediction
        nb_ok = eq.sum().item()  # count
        return nb_ok


class Version1(Net):
    def __init__(self):
        super(Version1, self).__init__()
        self.Conv1 = nn.Conv2d(1, 4, (5, 5), stride=(1, 1))
        self.Pool1 = nn.MaxPool2d((2, 2))
        self.Conv2 = nn.Conv2d(4, 16, (5, 5), stride=(1, 1))
        self.Pool2 = nn.MaxPool2d((2, 2))
        self.FC1 = nn.Linear(256, 128)
        self.FC2 = nn.Linear(128, 26)

    def forward(self, x):
        n = x.shape[0]
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Pool1(x)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Pool2(x)
        x = F.relu(x)
        x = x.reshape((n, 256))
        x = self.FC1(x)
        x = F.relu(x)
        x = self.FC2(x)
        return x


class Version2(Net):
    def __init__(self):
        super(Version2, self).__init__()
        self.Conv1 = nn.Conv2d(1, 4, (5, 5), stride=(1, 1))
        self.Pool1 = nn.MaxPool2d((2, 2))
        self.Conv2 = nn.Conv2d(4, 16, (3, 3), stride=(1, 1))
        self.Pool2 = nn.MaxPool2d((2, 2))
        self.FC1 = nn.Linear(400, 200)
        self.FC2 = nn.Linear(200, 26)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        n = x.shape[0]
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Pool1(x)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Pool2(x)
        x = F.relu(x)
        x = x.reshape((n, 400))
        x = self.FC1(x)
        x = F.relu(x)
        x = self.FC2(x)
        return x


class Version3(Net):
    def __init__(self):
        super(Version3, self).__init__()
        self.Conv1 = nn.Conv2d(1, 4, (5, 5), stride=(1, 1))
        self.Pool1 = nn.MaxPool2d((2, 2))
        self.Conv2 = nn.Conv2d(4, 16, (3, 3), stride=(1, 1))
        self.Pool2 = nn.MaxPool2d((2, 2))
        self.FC1 = nn.Linear(400, 200)
        self.FC2 = nn.Linear(200, 26)
        self.criterion = nn.CrossEntropyLoss()
        self.activation = lambda x: F.leaky_relu(x, negative_slope=.03)

    def forward(self, x):
        n = x.shape[0]
        x = self.Conv1(x)
        x = self.activation(x)
        x = self.Pool1(x)
        x = self.activation(x)
        x = self.Conv2(x)
        x = self.activation(x)
        x = self.Pool2(x)
        x = self.activation(x)
        x = x.reshape((n, 400))
        x = self.FC1(x)
        x = self.activation(x)
        x = self.FC2(x)
        return x


##############################################################################

def TRAIN(model, train_loader, optimizer):
    for batch_it, (data, target) in enumerate(train_loader):
        target = target - 1
        optimizer.zero_grad()
        scores = model.forward(data)
        loss = model.Loss(scores, target)
        loss.backward()
        optimizer.step()

        if batch_it % 50 == 0:
            print(f'   It: {batch_it:3}/{len(train_loader):3} --- Loss: {loss.item():.6f}')


def TEST(model, test_loader):
    err_tot = 0
    nb_ok = 0
    nb_images = 0

    with torch.no_grad():
        for data, target in test_loader:
            target = target - 1
            scores = model.forward(data)
            nb_ok += model.Test_OK(scores, target)
            err_tot += model.Loss(scores, target)
            nb_images += data.shape[0]

    pc_success = 100. * nb_ok / nb_images
    print(f'\nTest set:   Accuracy: {nb_ok}/{nb_images} ({pc_success:.2f}%)\n')
    return pc_success


##############################################################################

def main(batch_size):
    trs = transforms.Compose([transforms.ToTensor(), transforms.Normalize(.5, .5)])
    train_set = datasets.EMNIST('./data', split='letters', train=True, download=True, transform=trs)
    test_set = datasets.EMNIST('./data', split='letters', train=False, download=True, transform=trs)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, len(test_set))

    model = Version3()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    results = [TEST(model, test_loader)]

    import json
    for epoch in range(120):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        TRAIN(model, train_loader, optimizer)
        results.append(TEST(model, test_loader))
        with open('results/ex10.json', 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main(batch_size=64)
