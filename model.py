import numpy as np
from numpy import random
import time
import matplotlib.pyplot as plt
import pandas as pd


class NN:
    def __init__(self, input_nodes, output_nodes, hidden_layers, hidden_units):
        # Neural Network Structure Parameters
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units

        self.network_structure = [self.input_nodes] + hidden_units + [self.output_nodes]
        self.network_size = len(self.network_structure)

        # Neural Network Parameter Initialization
        self.THETAS = None
        self.theta = None
        self.lmd = None
        self.alpha = None

        # Data Sets
        self.x_train = None
        self.y_train = None
        self.x_cv = None
        self.y_cv = None
        self.x_test = None
        self.y_test = None

    def import_sets(self, x_train, y_train, x_cv, y_cv, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_cv = x_cv
        self.y_cv = y_cv
        self.x_test = x_test
        self.y_test = y_test

    def hyper_parameter_sets(self, alpha, lmd):
        self.lmd = lmd
        self.alpha = alpha

    def random_initialization(self, seed=None):
        np.random.seed(seed)
        self.theta = [np.random.rand(self.network_structure[i] + 1, self.network_structure[i + 1])
                      for i in range(self.network_size - 1)]

    def forward_pass(self, theta, x, activation=None):
        if not activation:
            activation = self.sigmoid
        a = [np.copy(x)]  # Nodal activation initialized
        z = np.ones(x.shape)  # Nodal outputs initialized
        da = []  # Derivative of nodal activation initialized

        # Forward pass

        for i in theta:
            a[-1] = np.insert(a[-1], 0, 1, axis=1)
            da.append(activation(np.insert(z, 0, 1, axis=1), True))
            z = np.dot(a[-1], i)
            a.append(activation(z))
        return a, da  # Return nodal activation & derivatives at each layer

    def back_propagate(self, x, y, lmd):
        a, da = self.forward_pass(self.theta, x)  # Call nodal activation & derivatives at each layer
        theta_ = []  # Create empty list to exclude bias from regularization.
        m, _ = x.shape

        for i in self.theta:
            k = np.copy(i)
            k[0] = 0
            theta_.append(k)

        delta = [a[-1] - y]  # Calculate error in last layer
        for c, i in enumerate(reversed(self.theta)):  # propagate errors backwards
            delta.append((np.dot(delta[-1], i.T) * da[-c - 1])[:, 1:])

        # flip error array
        delta.reverse()

        # calculate gradient
        gradient = [np.dot(i.T, delta[n + 1]) / m + lmd * theta_[n] / m for n, i in enumerate(a[:-1])]
        return gradient

    def cost_function(self, x, y, lmd):
        a, _ = self.forward_pass(self.theta, x, self.sigmoid)
        m, _ = x.shape
        sumsqrt_theta = 0
        for i in self.theta:
            k = np.copy(i)
            k[0] = 0
            sumsqrt_theta += np.sum(k ** 2)
        # calculate cost
        cost = -np.sum(np.log(a[-1]) * y + (np.log(1 - a[-1]) * (1 - y))) / m + lmd * sumsqrt_theta / 2 / m
        return cost

    def sigmoid(self, x, grad=False):
        # Standard sigmoid function. Returns gradient @x if grad argument passed as True.
        if grad:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        else:
            return 1 / (1 + np.exp(-x))

    def gradient_descent(self, theta_init, x, y, alpha, lmd, max_iter=2000):
        cnt = 0
        theta = theta_init
        while cnt < max_iter:
            cost = self.cost_function(theta, x, lmd)
            gradient = self.back_propagate(theta, x, lmd)
            theta_old = np.copy(theta)
            theta = [i - alpha * gradient[n] for n, i in enumerate(theta_old)]
            if cnt % 100 == 0:
                print("iteration : ", cnt, "cost = ", cost)
            cnt += 1
        print("Finished")
        return theta

