from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def getDataLoader(data_set, batch_size, num_workers):
	root = './datasets/'
	transform = transforms.Compose([transforms.ToTensor()])

	# listed datasets only
	if data_set == 'Mnist':
		train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
		test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)
	elif data_set == 'Fashion-Mnist':
		train_set = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
		test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
	else:
		raise ValueError("Unknown dataset specified.")

	# split train and valid from train set
	total_count = len(train_set)
	train_count = int(total_count * 0.9)
	valid_count = total_count - train_count
	train_set, valid_set = random_split(train_set, [train_count, valid_count])

	# get loader
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
	valid_loader = DataLoader(valid_set, batch_size=1000, shuffle=True, pin_memory=True, num_workers=num_workers)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

	return train_loader, valid_loader, test_loader
