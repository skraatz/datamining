#!/usr/local/bin/python

import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import warnings
# suppresses this message : FutureWarning: From version 0.21, test_size will always
# complement train_size unless both are specified.FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.getcwd()+"/module")
from data_functions import *

DEFAULT_NEIGHBOUR_COUNT = 3
DEFAULT_CLASSIFIER = neighbors.KNeighborsClassifier(DEFAULT_NEIGHBOUR_COUNT, weights='distance')


def testrun(data, clf=DEFAULT_CLASSIFIER, class_rep_count=3, pca = None):
    # get count class members for training and the rest as test data, along with their classification (target)
    feature_data, target, target_names, indices = data

    test_data, test_target, training_data, training_target = \
        prepare_data(feature_data, target, target_names, indices, class_rep_count)

    sc = StandardScaler()
    training_data = sc.fit_transform(training_data)
    test_data = sc.transform(test_data)

    # run principal component analysis and only chose the n_components most important features
    if pca is not None:
        training_data = pca.fit_transform(training_data)
        test_data = pca.transform(test_data)

    clf.fit(training_data, training_target)

    y_pred = clf.predict(test_data)

    return accuracy_score(test_target, y_pred)


def experiment(data):
    pass

# test parameter
n_neighbors = 15

image_path = os.getcwd() + "/data/Images.csv"
hist_path = os.getcwd() + "/data/EdgeHistogram.csv"

# load data from files
data = load_data(image_path, hist_path, with_enn=False)

pca = PCA(n_components=45)

pca = None
# run test
training_members_per_class = [3, 5, 10]
neighbour_axis = range(2, 15, 2)

repeats = 1

results = list()
for class_rep_count in training_members_per_class:
    acc_list = list()
    for n in neighbour_axis:
        total = 0
        for r in range(repeats):
            print("test run #: " + str(r))
            classifier = neighbors.KNeighborsClassifier(n, weights='distance')
            result = class_rep_count, n, testrun(data, classifier, class_rep_count, pca)
            print(result)
            _, _, acc = result
            total += acc
        acc = acc / repeats
        acc_list.append(acc)
    results.append(acc_list)

# print(results)


for result in results:
    plt.plot(neighbour_axis, result, 'bs')


plt.ylabel('accuracy')
plt.xlabel('neighbours')
# plt.legend(training_members_per_class)
plt.show()
