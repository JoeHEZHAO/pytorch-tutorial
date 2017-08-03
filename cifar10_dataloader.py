import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class CIFAR10(object):
  """docstring for CIFAR10"""
  def __init__(self):
    super(CIFAR10, self).__init__()
    self.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # self._trainset = torchvision.datasets.CIFAR10(root='/Users/zhaohe/workspace/pytorch_toying/data', train=True, download=True, transform=self.transform)
    self._trainset = torchvision.datasets.CIFAR10(root='/home/dyj/workspace/pytorch-exercise/data', train=True, download=True, transform=self.transform)
    self._trainloader = torch.utils.data.DataLoader(self._trainset, batch_size=4, shuffle=True, num_workers=2)
    # self._testset = torchvision.datasets.CIFAR10(root='/Users/zhaohe/workspace/pytorch_toying/data', train=False, download=True, transform=self.transform)
    self._testset = torchvision.datasets.CIFAR10(root='/home/dyj/workspace/pytorch-exercise/data', train=False, download=True, transform=self.transform)
    self._testloader = torch.utils.data.DataLoader(self._testset, batch_size=4, shuffle=False, num_workers=2)

  def trainSet(self):
    return self._trainset

  def trainLoader(self):
    return self._trainloader

  def testSet(self):
    return self._testset

  def testLoader(self):
    return self._testloader

  def imshow(self):
    dataiter = iter(self._trainloader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

if __name__ == '__main__':
    CIFAR10 = CIFAR10()
    CIFAR10.imshow()







