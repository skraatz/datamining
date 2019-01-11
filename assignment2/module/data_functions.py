import numpy
from imblearn.under_sampling import EditedNearestNeighbours


def load_data(filename_classes, filename_features, with_enn=False, denoise=False):
    """
    loads data from data files
    :param filename_classes:
    :param filename_features:
    :param with_enn:
    :return: data rows, targets for training and test
    """
    print("loading classes")
    class_data = numpy.loadtxt(open(filename_classes, "rb"), delimiter=";", dtype=numpy.dtype('U20'), skiprows=1,
                               usecols=1)

    print("loading features")
    # remove first (index) column when loading the csv
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

    if with_enn:
        # clean up the data to limit chances of noisy training samples
        print("ENN: cleaning up data for instance selection")
        enn = EditedNearestNeighbours(return_indices=True)
        data_resampled, target_resampled, sample_indices = enn.fit_resample(feature_data, class_assignments)
        if denoise:
            # do not return any data points considered noise by enn
            return data_resampled, target_resampled, class_labels, sample_indices
        else:
            return feature_data, class_assignments, class_labels, sample_indices
    else:
        return feature_data, class_assignments, class_labels, list()


def prepare_data(data, target, target_names, indices, count):
    """
    prepares count data members per class as training data, the rest as test data
    if there is a nonempty list of indices present, only members of this list will
    be considered as "good" training data
    :param data: the feature data
    :param target: the class assignments
    :param target_names: list of class labels
    :param indices: the list of indices of representative candidates
    :param count: the number of members per class to be selected as training data
    :return:
    """
    print ("preparing the data")

    training_data = list()
    training_target = list()
    test_data = list()
    test_target = list()
    position_counter = 0

    class_counter = [0] * len(target_names)
    if len(indices) > 0:
        print ("indices list is present")
        for feature in data:
            element_class = target[position_counter]
            # only consider entry as training representative, if it was previously in the cleaned up section
            if position_counter in indices and class_counter[element_class] < count:
                training_data.append(feature)
                training_target.append(element_class)
                class_counter[element_class] += 1
            else:
                test_data.append(feature)
                test_target.append(element_class)
            position_counter += 1
        return test_data, test_target, training_data, training_target
    else:
        print ("indices list is empty")
        for feature in data:
            element_class = target[position_counter]
            if class_counter[element_class] < count:
                training_data.append(feature)
                training_target.append(element_class)
                class_counter[element_class] += 1
            else:
                test_data.append(feature)
                test_target.append(element_class)
            position_counter += 1
        return test_data, test_target, training_data, training_target
