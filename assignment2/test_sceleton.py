#!/usr/local/bin/python

import sys
import os
from sklearn import neighbors

sys.path.append(os.getcwd()+"/module")
from data_functions import *

# test parameter
count = 25
n_neighbors = 10

# load data from files
target, class_dict, target_names, feature_data = load_data(os.getcwd()+"/data/Images.csv", os.getcwd()+"/data/EdgeHistogram.csv")

# get count class members for training and the rest as test data, along with their classification (target)
test_data, test_target, training_data, training_target = prepare_data(feature_data, target, target_names, count)

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(training_data, training_target)

result = clf.predict(test_data)
comparison = zip(result, test_target)

print("Anzahl korrekte Vorhersagen: " + str(len(filter(lambda (x, y): x == y, comparison))))
print("Anzahl genutzte Testdaten: " + str(len(test_target)))
