import torch

def fgsm(model, loss_function, x, y, epsilon=0.1):
	# copy data and detach them
	x = x.clone().detach()
	y = y.clone().detach()
	x.requires_grad = True

	# calc the gradient of x
	outputs = model(x)
	loss = loss_function(outputs, y)
	loss.backward()

	# update x
	x = x + epsilon * x.grad.sign()
	x = torch.clamp(x, min=0, max=1).detach()

	return x

def PGD(model, loss_function, x, y, eps=0.1, alpha=0.01, iters=40):
	# copy data and detach them
	x = x.clone().detach()
	y = y.clone().detach()
	ori_x = x.clone().detach()

	for _ in range(iters):
		# the gradient of x is needed
		x.requires_grad = True

		# calc the gradient of x
		outputs = model(x)
		loss = loss_function(outputs, y)
		loss.backward()

		# update x while clamping it within eps
		x = x + alpha * x.grad.sign()
		eta = torch.clamp(x - ori_x, min=-eps, max=eps)
		x = torch.clamp(ori_x + eta, min=0, max=1).detach()

	return x
