from model import NN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("breast-cancer-wisconsin.data", index_col=0, na_values=['?'], header=None)
df.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, "Diagnosis"]
df.index.name = "ID"
df = df.fillna(df.mean())

X = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]].to_numpy()
Y = df[["Diagnosis"]].to_numpy()  # - 2) / 2

nn_model = NN([9, 9, 1])  # 10, 10, 10, 10, 1])
nn_model.import_set(X, Y, normalization=True, ratio=[7, 2, 1])
theta, cost_hist = nn_model.train_model((3, 1), max_iter=10000)

training_score = nn_model.cost_function(theta, nn_model.sets['train'])
cv_score = nn_model.cost_function(theta, nn_model.sets['cv'])
test_score = nn_model.cost_function(theta, nn_model.sets['test'])

print(nn_model.f1_score(theta, nn_model.sets['train']))
print(nn_model.f1_score(theta, nn_model.sets['cv']))
print(nn_model.f1_score(theta, nn_model.sets['test']))

print("training cost:", training_score)
print("cross validation cost:", cv_score)
print("test cost:", test_score)

# plt.plot(cost_hist[1:])
# plt.show()
