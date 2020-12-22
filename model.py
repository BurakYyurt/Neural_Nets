import numpy as np


def min_max(sets, inverse=False):
    if not inverse:
        j = []
        for i in sets:
            min_val = np.amin(i, axis=0)
            max_val = np.amax(i, axis=0)
            diff = max_val - min_val
            j.append((i - min_val) / diff)
        return j


def shuffle(sets, seed=None):
    np.random.seed(seed)
    rnd = np.arange(sets[0].shape[0])
    j = []
    for i in sets:
        j.append(i[rnd])
    return j


def split(data_set, ratio):
    """Returns len(ratio) number of sets. """

    set_size = data_set.shape[0]
    sum_ratio = sum(ratio)

    k = 0
    list1 = [k]
    list2 = []
    for i in ratio[:-1]:
        k += int(set_size * i / sum_ratio)
        list1.append(k)
        list2.append(k)
    list2.append(None)

    return [data_set[i:j] for i, j in zip(list1, list2)]


class Model:
    def __init__(self):
        self.features = None
        self.targets = None
        self.index = None
        self.set_size = None

        self.features_raw = None
        self.targets_raw = None

        self.train = None
        self.cross_validate = None
        self.test = None

        self.train_index = None
        self.cross_validate_index = None
        self.test_index = None

    def import_data(self, file_name, feature_count, delimiter, index=False):
        """imports provided dataset using numpy.genfromtxt method
        Separates data set into 3 -index, features and targets
        if index does not exist, creates an index.
        """
        data_set = np.genfromtxt(file_name, delimiter=delimiter)
        self.set_size, _ = data_set.shape
        if index:
            self.index = data_set[:, 0]
            self.features = data_set[:, 1:feature_count + 1]
            self.targets = data_set[:, feature_count + 1:]
        else:
            self.features = data_set[:, :feature_count]
            self.targets = data_set[:, feature_count:]
            self.index = np.arange(self.set_size)

    def shuffle_sets(self, seed=False):
        """Shuffles main data set"""
        self.features, self.targets, self.index = shuffle((self.features, self.targets, self.index))

    def normalize(self, function=min_max):
        self.features_raw = np.copy(self.features)
        self.targets_raw = np.copy(self.targets)
        self.features, self.targets = function((self.features, self.targets))

    def split_data(self, ratio=None):
        if ratio is None:
            ratio = [4, 1, 1]

        split_features = split(self.features, ratio)
        split_targets = split(self.targets, ratio)
        split_index = split(self.index, ratio)

        self.train = (split_features[0], split_targets[0])
        self.cross_validate = (split_features[1], split_targets[1])
        self.test = (split_features[2], split_targets[2])
        self.train_index = split_index[0]
        self.cross_validate_index = split_index[1]
        self.test_index = split_index[2]
