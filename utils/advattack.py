import torch

def PGD(model, loss_function, x, y, eps=0.3, alpha=2/255, iters=10):
	x = x.clone().detach()
	y = y.clone().detach()
	ori_x = x.clone().detach()

	for _ in range(iters):
		x.requires_grad = True

		outputs = model(x)
		loss = loss_function(outputs, y)
		loss.backward()

		x = x + alpha * x.grad.sign()
		eta = torch.clamp(x - ori_x, min=-eps, max=eps)
		x = torch.clamp(ori_x + eta, min=0, max=1).detach()

	return x
