import numpy as np
from numpy import random


class Network:
    def __init__(self, structure):
        # Neural Network Architecture
        self.input_nodes = structure[0]
        self.output_nodes = structure[-1]
        self.network_structure = structure
        self.network_size = len(self.network_structure)

        # Neural Network Parameter Initialization
        self.theta = None
        self.lmd = None
        self.alpha = None

        # Data Sets
        self.x_set = None
        self.y_set = None
        self.sets = {}

        # Activation function
        self.activation = self.sigmoid

    def random_initialization(self, seed=None):
        np.random.seed(seed)
        theta = [np.random.rand(self.network_structure[i] + 1, self.network_structure[i + 1])
                 for i in range(self.network_size - 1)]
        return theta

    def forward_pass(self, theta, x):
        a = [np.copy(x)]  # Nodal activation initialized
        z = np.ones(x.shape)  # Nodal outputs initialized
        da = []  # Derivative of nodal activation initialized

        # Forward pass
        for i in theta:
            a[-1] = np.insert(a[-1], 0, 1, axis=1)
            da.append(self.activation(np.insert(z, 0, 1, axis=1), True))
            z = np.dot(a[-1], i)
            a.append(self.activation(z))
        return a, da  # Return nodal activation & derivatives at each layer

    def back_propagate(self, theta, data_set, lmd):
        x, y = data_set
        a, da = self.forward_pass(theta, x)  # Call nodal activation & derivatives at each layer
        theta_ = []  # Create empty list to exclude bias from regularization.
        m, _ = x.shape

        for i in theta:
            k = np.copy(i)
            k[0] = 0
            theta_.append(k)

        delta = [a[-1] - y]  # Calculate error in last layer
        for c, i in enumerate(reversed(theta)):  # propagate errors backwards
            delta.append((np.dot(delta[-1], i.T) * da[-c - 1])[:, 1:])

        # flip error array
        delta.reverse()

        # calculate gradient
        gradient = [np.dot(i.T, delta[n + 1]) / m + lmd * theta_[n] / m for n, i in enumerate(a[:-1])]
        return gradient

    def cost_function(self, theta, data_set, lmd=0):
        x, y = data_set
        a, _ = self.forward_pass(theta, x)
        m, _ = x.shape
        sumsqrt_theta = 0
        for i in theta:
            k = np.copy(i)
            k[0] = 0
            sumsqrt_theta += np.sum(k ** 2)
        # calculate cost
        cost = -np.sum(
            np.log(a[-1] + 1e-15) * y + (np.log(1 - a[-1] + 1e-15) * (1 - y))) / m + lmd * sumsqrt_theta / 2 / m
        return cost

    def sigmoid(self, x, grad=False):
        # Standard sigmoid function. Returns gradient @x if grad argument passed as True.
        if grad:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        else:
            return 1 / (1 + np.exp(-x))

    def batch_gradient_descent(self, theta_init, data_set, alpha, lmd, max_iter=2000):
        cnt = 0
        theta = theta_init
        cost_history = []
        while cnt < max_iter:
            cost = self.cost_function(theta, data_set, lmd)
            cost_history.append(cost)
            gradient = self.back_propagate(theta, data_set, lmd)
            theta_old = theta.copy()
            theta = [i - alpha * gradient[n] for n, i in enumerate(theta_old)]
            if cnt % 200 == 0:
                print("iteration : ", cnt, "cost = ", cost)
            cnt += 1
        print("Finished")
        return theta

    def stochastic_gradient_descent(self, theta_init, data_set, alpha, lmd, epoch=10):
        theta = theta_init
        x, y = data_set
        m, _ = x.shape

        for e in range(epoch):
            for k in range(m):
                instance_set = (x[k].reshape(1, len(x[k])), y[k])
                cost = self.cost_function(theta, instance_set, lmd)
                gradient = self.back_propagate(theta, instance_set, lmd)
                theta_old = theta.copy()
                theta = [i - alpha * gradient[n] for n, i in enumerate(theta_old)]

            print("Epoch: {}, Cost: {}".format(e+1, cost))
        return theta

    def train_model(self, hyper_parameters, set_type='train', optimization=None, max_iter=2000, save_state=False):
        if not optimization:
            optimization = self.batch_gradient_descent
        print("Training initialized")
        print("Optimization Algorithm:")
        theta_init = self.random_initialization(seed=1)
        alpha, lmd = hyper_parameters
        theta = optimization(theta_init, self.sets[set_type], alpha, lmd, max_iter)

        if save_state:
            self.theta = theta
            self.thetas[save_state] = theta

        return theta

    def predict(self, x, threshold=0.9, eval_type=None, theta=False):
        if not theta:
            theta = self.theta
        a, _ = self.forward_pass(theta, x)
        if not eval_type:
            return a[-1]
        elif eval_type == 'threshold':
            return a[-1] > threshold
        elif eval_type == 'max':
            return a[-1] == np.max(a[-1], axis=1)[:, None]
