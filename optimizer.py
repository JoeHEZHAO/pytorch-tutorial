import torch.optim as optim
import torch.nn as nn

def optimizer_SGD(lr=0.001, momentum=0.9, net):
	return optimizer = optim.SGD(net.parameters(), lr, momentum)