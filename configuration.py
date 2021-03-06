from utils.datasets import getDataLoader
from utils.models import getModel
from itertools import cycle
import torch.nn.functional as F
import torch

class Conf:
	def __init__(self, **kwargs):
		# basic setting
		self.epochs = kwargs.get('epochs', 100)
		self.num_worker = kwargs.get('num_worker', 4)
		self.batch_size = kwargs.get('batch_size', 128)
		self.device = torch.device(kwargs.get('device', 'cpu'))
		self.goal_acc = kwargs.get('goal_acc', 0.95)

		# learning setting
		self.lr = kwargs.get('lr', 0.1)
		self.momentum = kwargs.get('momentum', 0.9)
		self.activate_function = kwargs.get('activate_function', 'sigmoid')

		# get dataset and loader
		self.data_set = kwargs.get('data_set', None)
		if self.data_set is None:
			raise ValueError("Dataset not specified.")
		else:
			self.train_loader, self.valid_loader, self.test_loader \
				= getDataLoader(self.data_set, self.batch_size, self.num_worker)
			self.lip_loader = cycle(self.valid_loader)

		# get model
		self.model = getModel(self.data_set, self.activate_function)

		self.loss_function = kwargs.get('loss_function', F.cross_entropy)

		# optional setting
		self.valid_on = kwargs.get('valid_on', False)

		# lipschitz setting
		self.lip_on = kwargs.get('lip_on', False)
		self.lip_iters = kwargs.get('lip_iters', 2000)
		self.lip_lr = kwargs.get('lip_lr', 1.0)
		self.lip_limit = kwargs.get('lip_limit', 5e3)

		self.lamda = kwargs.get('lamda', 0.0)
		self.warmup = kwargs.get('warmup', 5)
		self.cooldown = kwargs.get('cooldown', 2)
		self.lamda_increment = kwargs.get('lamda_increment', 0.1)


def plainExample(epochs=100, data_set=None, activate_function='sigmoid', device='cpu', valid_on=False, lip_on=False):
	conf_args = {'epochs': epochs, 'data_set': data_set, 'device': device, 'valid_on': valid_on, 'lip_on': lip_on,
				 'activate_function': activate_function}
	conf = Conf(**conf_args)
	return conf
