import itertools

import numpy as np

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

# ML Functions
from ML_Algorithms import knn, svm, random_forest

from sklearn.model_selection import train_test_split

# Audio
from audio_analysis import get_feature_vector
from audio_analysis import label_encoder
from audio_analysis import label_encoder_for_test

from load_files import load_train_set, read_data_file

from side_funcs import fit_train_test


def run():
    # data_set, data_labels = load_train_set('./data-set/')
    data_set, data_labels = read_data_file()

    train_files, test_files, train_labels, test_labels = train_test_split(data_set, data_labels, test_size=0.25)

    # for label in range(len(train_labels)):
    #     if train_labels[label] not in test_labels:
    #         train_labels[label] = "other"
    #
    # for label in range(len(test_labels)):
    #     if test_labels[label] not in train_labels:
    #         test_labels[label] = "other"

    train_files, train_labels, test_files, test_labels = fit_train_test(train_files, train_labels, test_files,
                                                                        test_labels)

    train_classes, encoded_classes = label_encoder(train_labels)
    test_classes = label_encoder_for_test(encoded_classes, test_labels)

    # train_set = get_feature_vector(train_files)
    train_set = train_files
    test_set = test_files
    # test_set = get_feature_vector(test_files)

    knn(train_set, train_classes, test_set, test_classes)
    svm(train_set, train_classes, test_set, test_classes)
    random_forest(train_set, train_classes, test_set, test_classes, encoded_classes)
