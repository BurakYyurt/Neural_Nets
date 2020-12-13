from model import NN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


df = pd.read_csv("iris.data", header=None)
df.columns = [1, 2, 3, 4, "class"]

mapping = {"Iris-setosa": 0, "Iris-versicolor": 1,"Iris-virginica":2}

df = df.replace({"class": mapping})

X = df[[1, 2, 3, 4]].to_numpy()
Y_values = df[["class"]].to_numpy()
Y = np.array([[1 if j == i else 0 for j in range(3)]for i in Y_values])

nn_model = NN([4, 4, 3])  # 10, 10, 10, 10, 1])
nn_model.import_set(X, Y, normalization=True, ratio=[7, 2, 1])
theta, cost_hist = nn_model.train_model((3, 1), max_iter=10000)

training_score = nn_model.cost_function(theta, nn_model.sets['train'])
cv_score = nn_model.cost_function(theta, nn_model.sets['cv'])
test_score = nn_model.cost_function(theta, nn_model.sets['test'])

print("training cost:", training_score)
print("cross validation cost:", cv_score)
print("test cost:", test_score)

# plt.plot(cost_hist[1:])
# plt.show()
