import numpy as np
from numpy import random


def normalize(sets, inverse=False):
    if not inverse:
        j = []
        for i in sets:
            min_val = np.amin(i, axis=0)
            max_val = np.amax(i, axis=0)
            diff = max_val - min_val
            j.append((i - min_val) / diff)
        return j


class NN:
    def __init__(self, structure):
        # Neural Network Architecture
        self.input_nodes = structure[0]
        self.output_nodes = structure[-1]
        self.network_structure = structure
        self.network_size = len(self.network_structure)

        # Neural Network Parameter Initialization
        self.THETAS = None
        self.theta = None
        self.lmd = None
        self.alpha = None

        # Data Sets
        self.x_set = None
        self.y_set = None
        self.sets = {}

        # Activation function
        self.activation = self.sigmoid

    def import_set(self, x_set_raw, y_set_raw, ratio=None, shuffle=True, seed=None, normalization=True):
        np.random.seed(seed)

        if not ratio:
            ratio = [2, 2, 1]

        m, _ = x_set_raw.shape

        if normalization:
            x_set, y_set = normalize((x_set_raw, y_set_raw))
        else:
            x_set = x_set_raw
            y_set = y_set_raw

        if shuffle:
            rnd = np.arange(m)
            np.random.shuffle(rnd)
            self.x_set = x_set[rnd]
            self.y_set = y_set[rnd]
        else:
            self.x_set = x_set
            self.y_set = y_set

        train_size = int(m * ratio[0] / sum(ratio))
        cv_size = int(m * ratio[1] / sum(ratio))

        self.sets['whole'] = (self.x_set, self.y_set)
        self.sets['train'] = (self.x_set[:train_size], self.y_set[:train_size])
        self.sets['cv'] = (self.x_set[train_size:train_size + cv_size],
                           self.y_set[train_size:train_size + cv_size])
        self.sets['test'] = (self.x_set[train_size + cv_size:],
                             self.y_set[train_size + cv_size:])

    def hyper_parameter_sets(self, alpha, lmd):
        self.lmd = lmd
        self.alpha = alpha

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
        cost = -np.sum(np.log(a[-1]) * y + (np.log(1 - a[-1]) * (1 - y))) / m + lmd * sumsqrt_theta / 2 / m
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
        return theta, cost_history

    def train_model(self, hyper_parameters, set_type='train', optimization=None, max_iter=2000):
        if not optimization:
            optimization = self.batch_gradient_descent

        theta_init = self.random_initialization(seed=1)
        alpha, lmd = hyper_parameters
        theta, cost_history = optimization(theta_init, self.sets[set_type], alpha, lmd, max_iter)
        return theta, cost_history

    def f1_score(self, theta, data_set, threshold=0.95):
        x, y = data_set
        a, _ = self.forward_pass(theta, x)  # Call nodal activation & derivatives at each layer
        prediction = a[-1] > threshold
        evaluation = prediction == y
        true_prediction = np.sum(evaluation)
        positive_prediction = np.sum(prediction)
        positive_data = np.sum(y)
        true_positive = np.sum(np.logical_and(y, prediction == y))
        true_negative = np.sum(np.logical_and(np.logical_not(y), prediction == y))
        precision = true_positive / positive_prediction
        recall = true_positive / positive_data
        return 2 * ((recall * precision) / (recall + precision))


