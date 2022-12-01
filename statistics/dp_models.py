import numpy as np
import torch

class DPLogisticRegression(torch.nn.Module):
	def __init__(
		self, 
		dim,
		regularization_lambda,
		alpha,
		epsilon,
		b_dist = 1
	):
		super(DPLogisticRegression, self).__init__()

		self.dim = dim

		self.w = torch.nn.Linear(self.dim, 1)

		self.s = self.dim+1
		self.epsilon = epsilon
		self.b_dist = b_dist
		self.set_b(self.b_dist) 
		# self.w = np.random.normal(size=self.s)
		self.regularization_lambda = regularization_lambda
		self.alpha = alpha
		self.psi = None

	def set_b(self, dist):
		if dist == 1:
			b = np.random.laplace(0, 2, self.s)
		if dist == 2:
			w = np.random.normal(0, 1, self.s)
			w /= np.linalg.norm(w)
			b = w*np.random.chisquare(2*self.s)

		self.b = torch.from_numpy(b.astype(np.float32))

	def set_psi(self, x):
		kappa = np.linalg.norm(x, self.b_dist, 1).max()
		self.psi = 2*kappa

	def forward(self, x):
		logits = self.w(x)
		return logits

	def pred(self, x):
		logits = self.w(x)
		y_pred = torch.sign(logits).reshape((-1))
		if (y_pred == 0).any().item():
			y_pred = y+pred + 1 - torch.abs(y_pred)
		return y_pred

	def logistic_loss(self, y, logits):
		l = torch.log(1 + torch.exp(-y*logits)).mean()
		return l

	def regularization_loss(self, num_samples):
		r = (1/2)*self.regularization_lambda*(1 - self.alpha)*(
			torch.pow(self.w.weight, 2).sum() + torch.pow(self.w.bias, 2).sum()
			)
		r += self.regularization_lambda*self.alpha*(
			torch.abs(self.w.weight).sum() + torch.abs(self.w.bias).sum()
			)
		n = self.psi/(self.epsilon*num_samples) * (
			torch.dot(self.b[:-1], self.w.weight[0]) + 
			torch.dot(self.b[-1:], self.w.bias)
			)
		reg_loss = r+n
		return reg_loss