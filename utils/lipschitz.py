import torch

class Lip:
	def __init__(self, model, u=None, v=None):
		self.flag = False
		self.iters = 0
		self.model = model
		self.u, self.v = u, v

	def reset(self, lip_loader):
		# get u loader and v loader
		x, _ = next(lip_loader)
		self.u, self.v = x, x + torch.rand_like(x) * 1e-1
		self.iters = 1

	def calcLip(self):
		output_u = self.model(self.u)
		output_v = self.model(self.v)

		def l2_norm(a):
			return torch.norm(a.view(a.shape[0], -1), p=2, dim=1)
		loss = l2_norm(output_u - output_v) / l2_norm(self.u - self.v)
		return loss

	def advUpdate(self, conf, decay=1.3):
		self.u = self.u.to(conf.device)
		self.v = self.v.to(conf.device)
		self.u.requires_grad = True
		self.v.requires_grad = True

		loss_max = self.calcLip()
		lip_lr = torch.tensor(conf.lip_lr).repeat(self.u.shape[0]).view(-1, 1, 1, 1).to(conf.device)

		for i in range(5):
			self.u.requires_grad = True
			self.v.requires_grad = True

			# get loss and gradient
			loss = self.calcLip()
			loss_sum = torch.sum(loss)
			loss_sum.backward()

			# update u, v
			u_new = self.u + lip_lr * self.u.grad
			v_new = self.v + lip_lr * self.v.grad
			loss_new = Lip(self.model, u_new, v_new).calcLip()
			mask = (loss_new > loss_max).type(torch.int).view(-1, 1, 1, 1)
			self.u = u_new * mask + self.u * (1 - mask)
			self.v = v_new * mask + self.v * (1 - mask)
			self.u = self.u.detach()
			self.v = self.v.detach()


			# update lipschitz learning rate
			lip_lr = lip_lr * decay * mask + lip_lr / decay * (1 - mask)
			loss_max = loss_new * mask.squeeze() + loss_max * (1 - mask).squeeze()

		return loss_max

class Lamda:
	def __init__(self, lamda=0.0, warmup=5, cooldown=2, lamda_increment=0.2):
		self.lamda = lamda
		self.warmup = warmup
		self.cooldown = cooldown
		self.cooling = 0
		self.increment = lamda_increment
		self.decay = 1.0

	def getLamda(self):
		return self.lamda

	def updateLamda(self, train_acc, goal_acc):
		if self.warmup > 0:
			self.warmup -= 1
		elif self.cooling > 0:
			self.cooling -= 1
		elif 0 <= train_acc - goal_acc < 1e-3:
			pass
		else:
			# update lamda
			if train_acc - goal_acc >= 2e-3:
				self.lamda += self.increment
				self.decay = 2.0
			else:
				self.lamda = max(0.0, self.lamda - self.increment)
				self.increment /= self.decay
				self.decay = 1.0
			self.cooling = self.cooldown
		return self.lamda
