from utils.advattack import PGD

def train_step(conf, model, optimizer):
	# set train mode
	model.train()

	# init accuracy and loss
	train_acc = 0.0
	train_loss = 0.0
	total_count = 0

	# get loader and loss
	train_loader = conf.train_loader
	loss_function = conf.loss_function

	for batch_idx, (x, y) in enumerate(train_loader):
		x, y = x.to(conf.device), y.to(conf.device)

		# reset gradients
		optimizer.zero_grad()

		# calc loss and gradients
		output = model(x)
		loss = loss_function(output, y)

		# backward and update
		loss.backward()
		optimizer.step()

		# update accuracy and loss
		output_label = output.max(1)[1]
		train_acc += (output_label == y).sum().item()
		train_loss += loss.item()
		total_count += y.shape[0]

	print("[Log] Train Accuracy:", train_acc / total_count)
	print("[Log] Train Loss:", train_loss)

def valid_step(conf, model):
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

def test_step(conf, model):
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

		if conf.pgd_on is True:
			x = PGD(model, conf.loss_function, x, y)
		# calc loss and gradients
		output = model(x)
		loss = loss_function(output, y)

		# update accuracy and loss
		output_label = output.max(1)[1]
		test_acc += (output_label == y).sum().item()
		test_loss += loss.item()
		total_count += y.shape[0]

	print("[Log] Test Accuracy:", test_acc / total_count)
	print("[Log] Test Loss:", test_loss)
