import time
import torch
import TVT
import configuration as config

if __name__ == "__main__":
	# initialize
	# popular settings can be changed here, others will be set to default
	conf = config.plainExample(epochs=10, data_set='Mnist', device='cuda', pgd_on=True)

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
		print(50 * '-')
		print("[Info] Training.")
		TVT.train_step(conf, model, optimizer)

		# valid step
		if conf.valid_on is True:
			print(50 * '-')
			print("[Info] Validating.")
			TVT.valid_step(conf, model)

		print(50 * '-')
		epoch_end_time = time.time()
		epoch_time = epoch_end_time - epoch_start_time
		print("[Info] Epoch {epoch_id} ends, {epoch_time:.2f} seconds costs.".format(epoch_id=i, epoch_time=epoch_time))

	print(50 * '-')
	print("[Info] Testing.")
	TVT.test_step(conf, model)

	end_time = time.time()
	print("[Info] Program ends, {:.2f} minutes costs.".format((end_time - start_time) / 60))
