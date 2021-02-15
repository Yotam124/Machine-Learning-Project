import itertools

import numpy as np

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

# ML Functions
from sklearn.model_selection import train_test_split

from ML_Algorithms import knn, svm, random_forest, linear_regression

# Audio
from audio_analysis import get_feature_vector, get_feature_vector_2
from audio_analysis import label_encoder
from audio_analysis import label_encoder_for_test

from load_files import load_train_set, load_test_set, read_data_file

from side_funcs import fit_train_test, cut_a_single_song


def run():
    # train_files, train_labels = read_data_file('./train_test_data_combined.txt')
    data_set, data_labels = load_test_set('./IRMAS-TrainingData/')

    train_files, test_files, train_labels, test_labels = train_test_split(data_set, data_labels, test_size=0.25)

    train_set, train_labels, test_set, test_labels = fit_train_test(train_files, train_labels, test_files,
                                                                    test_labels)

    print("test_labels length: ", len(test_labels))
    print("test_files length: ", len(test_files))
    print()
    print("train_labels length: ", len(train_labels))
    print("train_files length: ", len(train_files))
    print()

    train_classes, encoded_classes = label_encoder(train_labels)
    test_classes = label_encoder_for_test(encoded_classes, test_labels)

    train_set = get_feature_vector_2(train_files)
    test_set = get_feature_vector_2(test_files)

    # linear_regression(train_set, train_classes, test_set, test_classes, encoded_classes)
    knn(train_set, train_classes, test_set, test_classes, encoded_classes)
    svm(train_set, train_classes, test_set, test_classes, encoded_classes)
    random_forest(train_set, train_classes, test_set, test_classes, encoded_classes)
