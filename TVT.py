from utils.advattack import PGD, fgsm
from utils.lipschitz import Lip, Lamda
import torch

def trainStep(conf, model, optimizer, lipschitz=None, lamda=None):
	# set train mode
	model.train()

	# init accuracy and loss
	train_acc = 0.0
	train_loss = 0.0
	total_count = 0
	lip_max = 0.0

	# get loader and loss
	train_loader = conf.train_loader
	loss_function = conf.loss_function

	for batch_idx, (x, y) in enumerate(train_loader):
		x, y = x.to(conf.device), y.to(conf.device)

		if conf.lip_on is True:
			# adversarial update
			if lipschitz.iters % conf.lip_iters == 0:
				# reset lipschitz
				lipschitz.reset(conf.lip_loader)
			else:
				lipschitz.iters += 1

			# construct the calc graph
			lip = lipschitz.advUpdate(conf)
			lip_idx = torch.argmax(lip)
			u, v = lipschitz.u[lip_idx:lip_idx + 1], lipschitz.v[lip_idx:lip_idx + 1]
			u, v = u.detach().to(conf.device), v.detach().to(conf.device)
			lip_loss = Lip(model, u, v).calcLip()
			lip_max = max(lip_max, lip_loss.item())

		# reset gradients
		optimizer.zero_grad()

		# calc loss and gradients
		output = model(x)
		loss = loss_function(output, y)
		if conf.lip_on is True:
			if lip_loss.item() <= conf.lip_limit:
				lamda_tmp = lamda.getLamda()
				loss = loss + lamda_tmp * lip_loss

		# backward and update
		loss.backward()
		optimizer.step()

		# update accuracy and loss
		output_label = output.max(1)[1]
		train_acc += (output_label == y).sum().item()
		train_loss += loss.item()
		total_count += y.shape[0]

	train_acc = train_acc / total_count
	if conf.lip_on is True:
		print("[Log] Lipschitz Lamda:", lamda_tmp)
		print("[Log] Lipschitz constant:", lip_max)
		lamda.updateLamda(train_acc, conf.goal_acc)
	print("[Log] Train Accuracy:", train_acc)
	print("[Log] Train Loss:", train_loss)

def validStep(conf, model):
	# set valid mode
	model.eval()

	# init accuracy and loss
	valid_acc = 0.0
	valid_loss = 0.0
	total_count = 0

	# get loader and loss
	valid_loader = conf.valid_loader
	loss_function = conf.loss_function

	for batch_idx, (x, y) in enumerate(valid_loader):
		x, y = x.to(conf.device), y.to(conf.device)

		# calc loss and gradients
		output = model(x)
		loss = loss_function(output, y)

		# update accuracy and loss
		output_label = output.max(1)[1]
		valid_acc += (output_label == y).sum().item()
		valid_loss += loss.item()
		total_count += y.shape[0]

	print("[Log] Valid Accuracy:", valid_acc / total_count)
	print("[Log] Valid Loss:", valid_loss)

def testStep(conf, model, attack="no attack"):
	# set test mode
	model.eval()

	# init accuracy and loss
	test_acc = 0.0
	test_loss = 0.0
	total_count = 0

	# get loader and loss
	test_loader = conf.test_loader
	loss_function = conf.loss_function

	for batch_idx, (x, y) in enumerate(test_loader):
		x, y = x.to(conf.device), y.to(conf.device)

		# attack
		if attack.lower() == "no attack":
			pass
		elif attack.lower() == "fgsm":
			x = fgsm(model, loss_function, x, y)
		elif attack.lower() == "pgd":
			x = PGD(model, loss_function, x, y)
		else:
			raise ValueError("Unknown attack specified.")

		# calc loss and gradients
		output = model(x)
		loss = loss_function(output, y)

		# update accuracy and loss
		output_label = output.max(1)[1]
		test_acc += (output_label == y).sum().item()
		test_loss += loss.item()
		total_count += y.shape[0]

	print("[Log] Test Accuracy under {}:".format(attack), test_acc / total_count)
	print("[Log] Test Loss under {}:".format(attack), test_loss)
