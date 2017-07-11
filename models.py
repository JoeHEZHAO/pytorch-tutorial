import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Cifar10_Net(nn.Module):
    def __init__(self):
        super(Cifar10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    pass

if __name__ == '__main__':
    main()

# net = Net()
# print(net)

# target = Variable(torch.arange(1,11))
# criterion = nn.MSELoss()

# optimizer = optim.SGD(net.parameters(), lr=0.01)

# for i in range(1,1000):
#     input = Variable(torch.rand(1,1,32,32))
#     optimizer.zero_grad()
#     output = net(input)
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()
#     print("step " + str(i) + " loss is : " + str(loss.data))

# print(net(input))
