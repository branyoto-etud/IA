import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

######################################################


import random
import math
# (x,y,category)
points= []
N = 30    # number of points per class
K = 3     # number of classes
for i in range(N):
   r = i / N
   for k in range(K):
      t = ( i * 4 / N) + (k * 4) + random.uniform(0,0.2)
      points.append( [ ( r*math.sin(t), r*math.cos(t) ) , k ] )


# On se propose de travailler avec 2 couches de neurones :
# Input => Linear => Relu => Linear => Scores

######################################################
#
#  outils d'affichage -  NE PAS TOUCHER

def DessineFond():
    iS = ComputeCatPerPixel()
    levels = [-1, 0, 1, 2]
    c1 = ('r', 'g', 'b')
    plt.contourf(XXXX, YYYY, iS, levels, colors = c1)

def DessinePoints():
    c2 = ('darkred','darkgreen','lightblue')
    for point in points:
        coord = point[0]
        cat   = point[1]
        plt.scatter(coord[0], coord[1] ,  s=50, c=c2[cat],  marker='o')

XXXX , YYYY = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))


##############################################################
#
#  PROJET

# Nous devons apprendre 3 catégories : 0 1 ou 2 suivant ce couple (x,y)

# Pour chaque échantillon, nous avons comme information [(x,y),cat]

# Construisez une couche Linear pour un échantillon prédit un score pour chaque catégorie

# Le plus fort score est associé à la catégorie retenue
# Pour calculer l'erreur, on connait la bonne catégorie k de l'échantillon de l'échantillon.
# On calcule Err = Sigma_(j=0 à nb_cat) max(0,Sj-Sk)  avec Sj score de la cat j

# Comment interpréter cette formule :
# La grandeur Sj-Sk nous donne l'écart entre le score de la bonne catégorie et le score de la cat j.
# Si j correspond à k, la contribution à l'erreur vaut 0, on ne tient pas compte de la valeur Sj=k dans l'erreur
# Sinon Si cet écart est positif, ce n'est pas bon signe, car cela sous entend que le plus grand
#          score ne correspond pas à la bonne catégorie et donc on obtient un malus.
#          Plus le mauvais score est grand? plus le malus est important.
#       Si cet écart est négatif, cela sous entend que le score de la bonne catégorie est supérieur
#          au score de la catégorie courante. Tout va bien. Mais il ne faut pas que cela influence
#          l'erreur car l'algorithme doit corriger les mauvaises prédictions. Pour cela, max(0,.)
#          permet de ne pas tenir compte de cet écart négatif dans l'erreur.

class Net(nn.Module):
    def __init__(self, neuron=500):
        super().__init__()
        self.couche1 = nn.Linear(2,neuron)
        self.couche2 = nn.Linear(neuron, 3)
    
    def forward(self, input):
        input = self.couche1(input)
        input = F.relu(input)
        input = self.couche2(input)
        return input

    def loss(self, scores, category):
        score_cat = scores[category]
        s = torch.sum(torch.fmax(torch.FloatTensor([0]), scores-score_cat+1))
        return s

model = Net(300)

def ComputeCatPerPixel():
    s = XXXX.shape
    CCCC = torch.ones((*s, 2))
    CCCC[:, :, 0] = torch.from_numpy(XXXX)
    CCCC[:, :, 1] = torch.from_numpy(YYYY)
    R = model(CCCC)
    P = torch.argmax(R, dim=2)
    return P.detach().numpy()

opti = optim.SGD(model.parameters(), lr=.01)

for it in range(50):
 
    for _ in range(100):
        # apprentissage
        opti.zero_grad()
        errtot = torch.FloatTensor([0])
        for (x,y), c in points:
            R = model(torch.FloatTensor([x, y]))
            errtot += model.loss(R, c)
        errtot.backward()
        opti.step()

    print(f'Iteration {it}, total error {errtot.item()}')
    DessineFond()
    DessinePoints()

    plt.title(str(it))
    plt.pause(0.1)
    plt.show(block=False)
    
    

