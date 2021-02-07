import itertools

import numpy as np

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

# ML Functions
import read_write_data
from ML_Algorithms import knn, svm, random_forest

from sklearn.model_selection import train_test_split

# Audio
from audio_analysis import get_feature_vector
from audio_analysis import label_encoder
from audio_analysis import label_encoder_for_test

from load_files import load_train_set, load_test_set


def run():
    train_files, train_labels = load_train_set('./data-set/')
    test_files, test_labels = load_test_set('./IRMAS-TrainingData/')

    train_indexes = []
    for label_i in range(len(train_labels)):
        if train_labels[label_i] not in test_labels:
            train_labels[label_i] = "other"
            # train_indexes.append(label_i)

    test_indexes = []
    for label_i in range(len(test_labels)):
        if test_labels[label_i] not in train_labels:
            # test_labels[label_i] = "other"
            test_indexes.append(label_i)

    for i in range(len(train_indexes)):
        del train_files[train_indexes[i]]
        del train_labels[train_indexes[i]]
        train_indexes = [x - 1 for x in train_indexes]

    for i in range(len(test_indexes)):
        del test_files[test_indexes[i]]
        del test_labels[test_indexes[i]]
        test_indexes = [x - 1 for x in test_indexes]

    print("test_labels length: ", len(test_labels))
    print("test_files length: ", len(test_files))
    print()
    print("train_labels length: ", len(train_labels))
    print("train_files length: ", len(train_files))

    train_classes, encoded_classes = label_encoder(train_labels)
    test_classes = label_encoder_for_test(encoded_classes, test_labels)

    train_set = get_feature_vector(train_files)
    test_set = get_feature_vector(test_files)

    knn(train_set, train_classes, test_set, test_classes)
    svm(train_set, train_classes, test_set, test_classes)
    random_forest(train_set, train_classes, test_set, test_classes, encoded_classes)
