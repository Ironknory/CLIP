import torch.nn as nn

class SlipModel(nn.Module):
	def __init__(self, sizes, activate_function='sigmoid', mean=0.0, std=1.0):
		super(SlipModel, self).__init__()

		if activate_function.lower() == 'sigmoid':
			self.activate_function = nn.Sigmoid
		elif activate_function.lower() == 'relu':
			self.activate_function = nn.ReLU
		else:
			raise ValueError("Unknown activate function specified.")

		self.mean, self.std = mean, std

		layers = [nn.Flatten()]
		for i in range(len(sizes) - 1):
			layers.append(nn.Linear(sizes[i], sizes[i + 1]))
			layers.append(self.activate_function())

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		x = (x - self.mean) / self.std
		return self.layers(x)

def getModel(data_set, activate_function):
	if data_set == 'Mnist':
		sizes = [784, 200, 80, 10]
		return SlipModel(sizes, activate_function)
	else:
		raise ValueError("Unknown dataset specified.")
