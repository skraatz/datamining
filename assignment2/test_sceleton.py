#!/usr/local/bin/python

import sys
import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

import warnings
# suppresses this message : FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.getcwd()+"/module")
from data_functions import *

DEFAULT_NEIGHBOUR_COUNT = 3
DEFAULT_CLASSIFIER = neighbors.KNeighborsClassifier(DEFAULT_NEIGHBOUR_COUNT, weights='distance')


def testrun(test_samples, clf=DEFAULT_CLASSIFIER, class_rep_count=3, pca = None):
    # get count class members for training and the rest as test data, along with their classification (target)
    feature_data, target, target_names = test_samples
    test_data, test_target, training_data, training_target = \
        prepare_data(feature_data, target, target_names, class_rep_count)

    # X_train, X_test, y_train, y_test = train_test_split(feature_data, target, train_size=count, random_state=0)

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    # run principal component analysis and only chose the n_components most important features
    if pca is not None:
        training_data = pca.fit_transform(training_data)
        test_data = pca.transform(test_data)

    clf.fit(training_data, training_target)

    # y_pred = clf.predict(X_test)
    y_pred = clf.predict(test_data)
    # cm = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(test_target, y_pred)

    # print('Accuracy ' + str(accuracy_score(test_target, y_pred)))
    # print('Precision ' + str(precision_score(test_target, y_pred, average='micro')))
    # print('Recall ' + str(recall_score(test_target, y_pred, average='micro')))
    return accuracy_score(test_target, y_pred)


# test parameter
n_neighbors = 15


image_path = os.getcwd() + "/data/Images.csv"
hist_path = os.getcwd() + "/data/EdgeHistogram.csv"

# load data from files
data = load_data(image_path, hist_path)

pca = PCA(n_components=65)
# clf = RandomForestClassifier(max_depth=65, random_state=0)
# clf = neighbors.KNeighborsClassifier(neighbour_count, weights='distance')

# pca = None
# run test
supilist = [3, 5, 15]
neighbour_axis = range(2, 21, 2)

results = list()
for class_rep_count in supilist:
    acc_list = list()
    for n in neighbour_axis:
        classifier = neighbors.KNeighborsClassifier(n, weights='distance')
        result = class_rep_count, n, testrun(data, classifier, class_rep_count, pca)
        print(result)
        _, _, acc = result
        acc_list.append(acc)
    results.append(acc_list)

print(results)

import matplotlib.pyplot as plt
for result in results:
    plt.plot(neighbour_axis, result)

plt.ylabel('accuracy')
plt.xlabel('neighbours')
plt.show()
