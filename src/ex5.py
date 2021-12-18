import torch, numpy, matplotlib.pyplot as plt

def step1():
  layer = torch.nn.Linear(1,1)
  activ = torch.nn.ReLU()
  Lx = numpy.linspace(-2,2,50)

  Ly = activ(
    layer(
      torch.FloatTensor(Lx)
           .reshape(50, 1)
    )
  ).detach().numpy()
  return Lx, Ly


def step2():
  l1 = torch.nn.Linear(1, 3)
  l2 = torch.nn.Linear(3, 1)
  activ = torch.nn.ReLU()
  Lx = numpy.linspace(-2, 2, 50)

  Ly = activ(
    l2(
      activ(
        l1(
          torch.FloatTensor(Lx)
            .reshape(50, 1)
        )
      )
    )
  ).detach().numpy()
  return Lx, Ly



Lx, Ly = step1()
Lx, Ly = step2()


plt.plot(Lx,Ly,'-')
plt.axis('equal')
plt.show()
