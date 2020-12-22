from network import NN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

np.set_printoptions(precision=3, suppress=True)

df = pd.read_csv("../data_sets/iris.data", header=None)
df.columns = [1, 2, 3, 4, "class"]

mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

df = df.replace({"class": mapping})

X = df[[1, 2, 3, 4]].to_numpy()
Y_values = df[["class"]].to_numpy()
Y = np.array([[1 if j == i else 0 for j in range(3)] for i in Y_values])

nn_model = NN([4, 4, 3])  # 10, 10, 10, 10, 1])
nn_model.import_set(X, Y, normalization=True, ratio=[6, 2, 2])
theta_init = nn_model.random_initialization()
theta_1 = nn_model.stochastic_gradient_descent(theta_init, nn_model.sets['train'], 0.3, 0.001, epoch=30)
theta_2 = nn_model.batch_gradient_descent(theta_init, nn_model.sets['train'], 1, 0.3, max_iter=3000)

predict_1 = nn_model.predict(nn_model.sets['test'][0], theta=theta_1, eval_type='max')
print(predict_1 == nn_model.sets['test'][1])

predict_2 = nn_model.predict(nn_model.sets['test'][0], theta=theta_2, eval_type='max')
print(predict_2 == nn_model.sets['test'][1])
