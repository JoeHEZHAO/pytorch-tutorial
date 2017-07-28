import torch
import torch.nn as nn
from torch.autograd import Variable
from models import Cifar10_Net
from cifar10_dataloader import CIFAR10
import torch.optim as optim

cifar10_data = CIFAR10()
train_model = Cifar10_Net()
optimizer = optim.SGD(train_model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):

	running_loss = 0.0
	for i ,data in enumerate(cifar10_data.trainLoader()):
		inputs, labels = data
		inputs, labels = Variable(inputs), Variable(labels)

		optimizer.zero_grad()

		outputs = train_model(inputs)
		loss = criterion(outputs, labels)

		loss.backward()
		optimizer.step()
	    
	    # print statistics
		running_loss += loss.data[0]
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finish training')
