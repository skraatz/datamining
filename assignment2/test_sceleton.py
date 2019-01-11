#!/usr/local/bin/python

import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.metrics import accuracy_score

import warnings
# suppresses this message : FutureWarning: From version 0.21, test_size will always
# complement train_size unless both are specified.FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.getcwd()+"/module")
from data_functions import *
from plotting import *


def printhelp():
    helpstring = "script usage: python test_sceleton.py [plot]\n"
    helpstring = helpstring + "example 1 (no plotting): python test_sceleton.py \n"
    helpstring = helpstring + "example 2 (with plotting: python test_sceleton.py plot\n"
    print(helpstring)


image_path = os.getcwd() + "/data/Images.csv"
hist_path = os.getcwd() + "/data/EdgeHistogram.csv"
out_path = os.getcwd() + "/output/"

DEFAULT_NEIGHBOUR_COUNT = 3
DEFAULT_CLASSIFIER = neighbors.KNeighborsClassifier(DEFAULT_NEIGHBOUR_COUNT, weights='distance')
DEFAULT_PLOT_ENABLED = False
DEFAULT_PCA_NCOMPONENTS = 45

training_members_per_class = [3, 5, 10]
neighbour_axis = range(2, 15, 2)
data_out_header = "2, 4, 6, 8, 10, 12, 14"


def testrun(data, clf=DEFAULT_CLASSIFIER, class_rep_count=3, pca=None):
    # get count class members for training and the rest as test data, along with their classification (target)
    feature_data, target, target_names, indices = data

    test_data, test_target, training_data, training_target = \
        prepare_data(feature_data, target, target_names, indices, class_rep_count)

    sc = StandardScaler()
    training_data = sc.fit_transform(training_data)
    test_data = sc.transform(test_data)

    # run principal component analysis and only chose the n_components most important features
    if pca is not None:
        print("PCA is enabled")
        training_data = pca.fit_transform(training_data)
        test_data = pca.transform(test_data)

    clf.fit(training_data, training_target)

    y_pred = clf.predict(test_data)

    return accuracy_score(test_target, y_pred)


def experiment(data, pca, repeats, exp_name):
    print("################################")
    print("running experiment " + exp_name)
    print("################################")
    test_results = list()
    for crp in training_members_per_class:
        acc_list = list()
        for n in neighbour_axis:
            total = 0
            for r in range(repeats):
                cls = neighbors.KNeighborsClassifier(n, weights='distance')
                test_result = crp, n, testrun(data, cls, crp, pca)
                print(test_result)
                _, _, acc = test_result
                total += acc
            acc = round(acc / repeats, 3)
            acc_list.append(acc)
        test_results.append(acc_list)

    numpy.savetxt(out_path + exp_name + ".csv", numpy.array(test_results), header=data_out_header, delimiter=';')


def get_optimal_pca_parm(data, exp_name):
    print("################################")
    print("running experiment " + exp_name)
    print("################################")
    test_results = list()
    header = ""
    for num_pca in range(2, 80, 2):
        use_semicolon = ""
        if num_pca > 2:
            use_semicolon = ";"
        header += (use_semicolon + str(num_pca))
        pca = PCA(n_components=num_pca)
        cls = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')
        test_result = num_pca, testrun(data, cls, 10, pca)
        print(test_result)
        _, acc = test_result
        acc = round(acc, 3)
        test_results.append(acc)

    numpy.savetxt(out_path + exp_name + ".csv", numpy.array(test_results), header=header, delimiter=';')


# main program
if __name__ == "__main__":
    plot_enabled = DEFAULT_PLOT_ENABLED

    if len(sys.argv) < 2 or sys.argv[1] in ["h", "-h", "-help"]:
        printhelp()
    else:
        if sys.argv[1] == "plot":
            plot_enabled = True

        data = load_data(image_path, hist_path, with_enn=False, denoise=False)
        get_optimal_pca_parm(data, "pca_tune")

        if plot_enabled:
            create_output(out_path)

"""
        repeats = 1
        # without edited nearest neighbours
        data = load_data(image_path, hist_path, with_enn=False)
        pca = None
        experiment(data, pca, repeats, "no_tuning")

        data = load_data(image_path, hist_path, with_enn=False)
        pca = PCA(n_components=DEFAULT_PCA_NCOMPONENTS)
        experiment(data, pca, repeats, "pca_enabled")

        # with edited nearest neighbours enabled
        data = load_data(image_path, hist_path, with_enn=True)
        pca = None
        experiment(data, pca, repeats, "with_enn_without_pca")

        data = load_data(image_path, hist_path, with_enn=True)
        pca = PCA(n_components=DEFAULT_PCA_NCOMPONENTS)
        experiment(data, pca, repeats, "with_enn_with_pca")

        data = load_data(image_path, hist_path, with_enn=True, denoise=True)
        pca = PCA(n_components=DEFAULT_PCA_NCOMPONENTS)
        experiment(data, pca, repeats, "with_enn_with_pca_denoised")
"""
