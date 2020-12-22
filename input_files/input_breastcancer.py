from network import NN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../data_sets/breast-cancer-wisconsin.data", index_col=0, na_values=['?'], header=None)
df.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, "Diagnosis"]
df.index.name = "ID"
df = df.fillna(df.mean())

X = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]].to_numpy()
Y = df[["Diagnosis"]].to_numpy()  # - 2) / 2

nn_model = NN([9, 9, 1])  # 10, 10, 10, 10, 1])
nn_model.import_set(X, Y, normalization=True, ratio=[7, 2, 1])
theta_init = nn_model.random_initialization()
theta_1 = nn_model.stochastic_gradient_descent(theta_init, nn_model.sets['train'], 0.3, 0.001, epoch=30)
theta_2 = nn_model.batch_gradient_descent(theta_init, nn_model.sets['train'], 1, 0.3, max_iter=3000)

predict_1 = nn_model.predict(nn_model.sets['test'][0], theta=theta_1, eval_type='threshold')
print(predict_1 == nn_model.sets['test'][1])

predict_2 = nn_model.predict(nn_model.sets['test'][0], theta=theta_2, eval_type='threshold')
print(predict_2 == nn_model.sets['test'][1])
