from network import NN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

X = np.genfromtxt('../data_sets/x.csv', delimiter=',')
Y_values = np.genfromtxt('../data_sets/y.csv', delimiter=',')
Y_values = np.where(Y_values == 10, 0, Y_values)
Y = np.array([[1 if j == i else 0 for j in range(10)] for i in Y_values])

nn_model = NN([400, 25, 25, 10])
nn_model.import_set(X, Y, normalization=False, ratio=[7, 2, 1], seed=1)

theta_init = nn_model.random_initialization()
theta_1 = nn_model.train_model((0.3, 0.001), nn_model.sets['train'], epoch=5)
file_name = '../theta_2.thetas'
out = open(file_name, 'wb')
pickle.dump(theta_1, out)

predict_1 = nn_model.predict(nn_model.sets['test'][0], theta=theta_1, eval_type='threshold')
print(predict_1 == nn_model.sets['test'][1])
