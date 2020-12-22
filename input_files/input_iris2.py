from model import Model
from network import Network

model = Model()
model.import_data("../data_sets/breast-cancer-wisconsin.data", 9, delimiter=",", index=True)
model.shuffle_sets()
model.normalize()
model.split_data()

