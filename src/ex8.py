import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import datasets, transforms

####  1 couche Linear
#	Qu1 : quel est le % de bonnes prédictions obtenu au lancement du programme , pourquoi ?
#       >> ~10% car les poids ont été initialisé au hasard. Et le résultat moyen de réponse au hasard avec une probabilité de 1/10 est 10% de succes.
#	Qu2 : quel est le % de bonnes prédictions obtenu avec 1 couche Linear ?
#       >> ~90%
#	Qu3 : pourquoi le test_loader n’est pas découpé en batch ?
#       >> Parce que le test_loader n'a pas besoin de calculer le grad
#   Qu4 : pourquoi la couche Linear comporte-t-elle 784 entrées ?
#       >> Images en niveau de gris de taille 28x28
#   Qu5 : pourquoi la couche Linear comporte-t-elle 10 sorties ?
#       >> Car il y a 10 réponses possibles (reconnaissance de nombres écrit à la main de 0 à 9)

####  2 couches Linear
#   Qu6 : quelles sont les tailles des deux couches Linear ?
#       >> première : (748, 128), deuxième : (128, 10)
# 	Qu7 : quel est l’ordre de grandeur du nombre de poids utilisés dans ce réseau ?
#       >> 100'000 (748 * 128)
#	Qu8 : quel est le % de bonnes prédictions obtenu avec 2 couches Linear ?
#       >> aux alentours de 97.5%

####  3 couches Linear
#   Qu9 : obtient-on un réel gain sur la qualité des prédictions ?
#       >> non nous avons toujours environ 97.5% de bonnes réponses.
#          Cependant, la convergence vers cette valeur semble plus rapide

####  Fonction Softmax
#   Qu10 : pourquoi est il inutile de changer le code de la fonction Test_OK ?
#       >> Le calcul pour savoir quelle est la catégorie prédite ne dépend pas 
#           de la valeur exacte du score mais du maximum comparé aux autres


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.FC1 = nn.Linear(784, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, 10)

    def forward(self, x):
        n = x.shape[0]
        x = x.reshape((n,784))
        output = self.FC1(x)
        output = F.relu(output)
        output = self.FC2(output)
        output = F.relu(output)
        output = self.FC3(output)
        return output


    def Loss(self,Scores,target):
        loss = F.log_softmax(Scores)
        return F.nll_loss(loss, target)

        # nb = Scores.shape[0]
        # TRange = torch.arange(0,nb,dtype=torch.int64)
        # scores_cat_ideale = Scores[TRange,target]
        # scores_cat_ideale = scores_cat_ideale.reshape(nb,1)
        # delta = 1
        # Scores = Scores + delta - scores_cat_ideale
        # x = F.relu(Scores)
        # err = torch.sum(x)
        # return err


    def TestOK(self,Scores,target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        pred = pred.reshape(target.shape)
        eq   = pred == target                      # True when correct prediction
        nbOK = eq.sum().item()                     # count
        return nbOK

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
    ErrTot   = 0
    nbOK     = 0
    nbImages = 0

    with torch.no_grad():
        for data, target in test_loader:
            Scores  = model.forward(data)
            nbOK   += model.Test_OK(Scores, target)
            ErrTot += model.Loss(Scores,target)
            nbImages += data.shape[0]

    pc_success = 100. * nbOK / nbImages
    print(f'\nTest set:   Accuracy: {nbOK}/{nbImages} ({pc_success:.2f}%)\n')

##############################################################################

def main(batch_size):

    moy, dev = 0.1307, 0.3081
    TRS = transforms.Compose([transforms.ToTensor(), transforms.Normalize(moy,dev)])
    TrainSet = datasets.MNIST('./data', train=True,  download=True, transform=TRS)
    TestSet  = datasets.MNIST('./data', train=False, download=True, transform=TRS)

    train_loader = torch.utils.data.DataLoader(TrainSet , batch_size)
    test_loader  = torch.utils.data.DataLoader(TestSet, len(TestSet))

    model = Net()
    optimizer = torch.optim.Adam(model.parameters())

    TEST(model,  test_loader)
    for epoch in range(40):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        TRAIN(batch_size, model,  train_loader, optimizer, epoch)
        TEST(model,  test_loader)


main(batch_size = 64)