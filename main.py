import torch
import time
import configuration as config

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


if __name__ == "__main__":
	# initialize
	# popular settings can be changed here, others will be set to default
	conf = config.plainExample(data_set='Mnist')

	# get model and optimizer(SGD with momentum)
	model = conf.model.to(conf.device)
	optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum)

	start_time = time.time()
	print("[Info] Training starts.")
	for i in range(conf.epochs):
		print(50 * '=')
		epoch_start_time = time.time()
		print("[Info] Epoch {epoch_id} starts.".format(epoch_id=i))

		# train step
		train_step(conf, model, optimizer)

		epoch_end_time = time.time()
		epoch_time = epoch_end_time - epoch_start_time
		print("[Info] Epoch {epoch_id} ends, {epoch_time:.2f} seconds costs.".format(epoch_id=i, epoch_time=epoch_time))

	end_time = time.time()
	print("[Info] Program ends, {:.2f} minutes costs.".format((end_time - start_time) / 60))