import numpy
import pandas as pd


def load_data(filename_classes, filename_features):
    print("loading classes")
    class_data = numpy.loadtxt(open(filename_classes, "rb"), delimiter=";", dtype=numpy.dtype('U20'), skiprows=1,
                               usecols=1)

    print("loading features")
    # dataset = pd.read_csv(filename_features, delimiter=';', skiprows=1, )
    feature_data = numpy.loadtxt(open(filename_features, "rb"), delimiter=";", dtype=int, skiprows=1, usecols=range(1, 81))

    # create set of class labels
    class_labels = list(set(class_data.tolist()))
    class_dict = dict()
    for enumerated_class_label in enumerate(class_labels):
        index, name = enumerated_class_label
        class_dict[name] = index

    # convert list of text labels to list of indices
    # this gives us the class assignments as list of indices
    class_assignments = list()
    for entry in class_data:
        class_assignments.append(class_dict[entry])

    # return class_assignments, class_dict, class_labels, feature_data
    return feature_data, class_assignments, class_labels


def prepare_data(data, target, target_names, count):
    """
    prepares count data members per class as training data, the rest as
    :param data:
    :param target:
    :param target_names:
    :param count:
    :return:
    """
    print ("preparing the data")

    training_data = list()
    training_target = list()
    test_data = list()
    test_target = list()
    position_counter = 0

    class_counter = [0] * len(target_names)
    for feature in data:
        datalen = len(feature)
        element_class = target[position_counter]
        if class_counter[element_class] < count:
            training_data.append(feature[1: datalen])  # append without row index
            training_target.append(element_class)
            class_counter[element_class] += 1
        else:
            test_data.append(feature[1:datalen])               # append without row index
            test_target.append(element_class)
        position_counter += 1
    return test_data, test_target, training_data, training_target
