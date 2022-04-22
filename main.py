import time
import torch
import TVT, utils
import configuration as config

if __name__ == "__main__":
	# initialize
	# popular settings can be changed here, others will be set to default
	conf = config.plainExample(data_set='Mnist', device='cuda', lip_on=True)

	# get model and optimizer(SGD with momentum)
	model = conf.model.to(conf.device)
	optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum)

	# init lipschitz class
	lipschitz = utils.lipschitz.Lip(model)
	lamda = utils.lipschitz.Lamda(conf.lamda, conf.warmup, conf.cooldown, conf.lamda_increment)

	start_time = time.time()
	print("[Info] Training starts.")
	for i in range(conf.epochs):
		print(50 * '=')
		epoch_start_time = time.time()
		print("[Info] Epoch {epoch_id} starts.".format(epoch_id=i))

		# train step
		print(50 * '-')
		print("[Info] Training.")
		TVT.trainStep(conf, model, optimizer, lipschitz, lamda)

		# valid step
		if conf.valid_on is True:
			print(50 * '-')
			print("[Info] Validating.")
			TVT.validStep(conf, model)

		print(50 * '-')
		epoch_end_time = time.time()
		epoch_time = epoch_end_time - epoch_start_time
		print("[Info] Epoch {epoch_id} ends, {epoch_time:.2f} seconds costs.".format(epoch_id=i, epoch_time=epoch_time))

	for attack in ["no attack", "fgsm", "PGD"]:
		print(50 * '-')
		print("[Info] Testing.")
		TVT.testStep(conf, model, attack=attack)

	end_time = time.time()
	print("[Info] Program ends, {:.2f} minutes costs.".format((end_time - start_time) / 60))
