import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
x = np.reshape(x, (100,1))
y = 0.2 + 1.3*x - 5.9*x**2 +6.25*x**3
# plt.plot(x, y)
# plt.show()

class byprod:
	def __init__(self, a, z, y):
		self.a = a
		self.z = z
		self.y = y


class WLayer:
	# def __init__(n_layer, node_per_layer=None):
	def __init__(n_layer,):
		self.n_layer = n_layer	

		# if node_per_layer:
		# 	if len(node_per_layer) != n_layer:
		# 		print 'Mismatched number of layers:'
		# 		print 'You specified: %d' % n_layer
		# 		print 'And: %s' % node_per_layer

		# 	self.W

		# else:
		self.W = [None for range(n_layer)]
		

	def set(W, layer):
		self.W[layer] = W
		return

	def get(layer):
		return self.W[layer]


class XSample:
	def __init_(n_layer):
		self.X = [None for range(n_layer)]
		self.n_layer = n_layer
		return

	def set(x, layer):
		self.X[layer] = x
		return 

	def get(layer):
		return self.X[layer]


def predict(X, W):
	N = x.shape[0]
	I = np.ones((N,1))

	z0 = np.concatenate((x, I), 1)

	a1 = np.dot(z0, Wji)
	z1 = 1.0 / (1.0 + np.exp(-a))
	z1 = np.concatenate((z, I), 1)

	y  = np.dot(z, Wkj)

	return byprod(a, z, y)

	# Append
	L = W.n_layer
	for l in range(L):



N = 100
M = 1
n_hiddens = 10
learning_rate = 0.01
epoch = 1000
Wji = np.random.normal(0, 0.1, (M+1, n_hiddens))
Wkj = np.random.normal(0, 0.1, (n_hiddens+1, 1))

I = np.ones((1,1))
for ep in range(epoch):
	print 'Epoch %5d' % ep
	for n in range(N):
		xn    = np.reshape(x[n,:], (1,1))
		
		# zi    = np.append(xn, I, axis=1)
		zi    = np.concatenate((xn, I), 1)
		zi    = np.transpose(zi)			# 2x1
		zi = np.array(zi)
		# print zi
		# print 'zi:'
		# print zi.shape

		B = predict(xn, Wkj, Wji)

		y_hat = B.y

		zj    = np.transpose(B.z)	# 4x1
		sk    = y_hat - y[n]		# 1x1
		VEkj  = sk * zj

		hh    = zj * (1. - zj)	# 4x1
		hh[-1]= 1
		prp   =(Wkj * sk)

		# print 'hh   prp'
		# print hh.shape
		# print prp.shape

		sj    = hh * prp
		VEji  = zi * np.transpose(sj[:-1])

		# print 'VEkj:%s, VEji:%s' % (VEkj.shape, VEji.shape)	
		VEkj = VEkj
		VEji = VEji

		# print 'Wji:%s, VEji%s' % (Wji.shape, VEji.shape)
		Wji = Wji - learning_rate * VEji
		Wkj = Wkj - learning_rate * VEkj


print Wkj
print Wji

xr = predict(x, Wkj, Wji)

plt.plot(x, y, 'r')
plt.hold(True)
plt.plot(x, xr.y)
plt.show()


# """
# Input and output must be both 1D
# """
class OneHiddenLayerNeuralRegressor:
	def __init__(self):
		self.M  = 3
		self.W1 = None
		self.W2 = None
		self.learning_rate = 0.005
		self.epoch = 1000



	def fit(self, x, y):
		for ep in range(self.epoch):
			for n in range(N):
				xn = np.reshape(x[n,:], (1,1))

				zi = np.concatenate((xn, I), 1)
				zi = np.transpose(zi)			# 2x1
				zi = np.array(zi)

				prediction = predict(xn, Wkj, Wji)

				y_hat = prediction.y

				zj   = np.transpose(prediction.z)	# 4x1
				sk   = y_hat - y[n]		# 1x1
				VEkj = sk * zj

				hh    = zj * (1. - zj)	# 4x1
				hh[-1]= 1
				prp   =(Wkj * sk)
				sj    = hh * prp
				VEji  = zi * np.transpose(sj[:-1])

				Wji = Wji - learning_rate * VEji
				Wkj = Wkj - learning_rate * VEkj

# 		N = x.shape[0]
# 		if len(x.shape) < 2:
# 			M = 1
# 		else:
# 			M = x.shape[1]

# 		self.Wji = np.zeros((M+1, self.M))
# 		self.Wkj = np.zeros((self.M+1, 1))

# 		Zi = np.concatenate((x, ones(N,1)), 1)

# 		# Update

# 		for epoch in range(self.epoch):
# 			for n in range(N):
# 				zi = Zi[n,:]

# 				y_hat = self.predict(zi)

# 				zj = self.z[n,:]

# 				sk = y_hat - y[n]
# 				sj = np.multiply((zj * (1. - zj)), (self.Wkj * sk))


# 				VEkj = sk * zj
# 				VEji = sj * zi

# 				# self.Wji = 


# 		return


# 	def predict(self, x):
# 		(N,M) = x.shape
# 		x = np.concatenate((x, ones(N,1)), 1)
# 		self.a = np.dot(x, self.Wji)
# 		z = 1.0 / (1.0 + np.exp(-a))
# 		self.z = np.concatenate((z, ones(N,1)), 1)
# 		y = np.dot(self.z, self.Wkj)
# 		return y


# 	def loss(self, y_hat, y):
# 		x = (y_hat - y) ** 2
# 		return 0.5 * x.sum()


