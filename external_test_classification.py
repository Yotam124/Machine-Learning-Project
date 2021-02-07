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

from load_files import load_train_set, load_test_set, read_data_file

from side_funcs import fit_train_test



def run():
    # train_files, train_labels = load_train_set('./data-set/')
    train_files, train_labels = read_data_file()

    test_files, test_labels = load_test_set('./IRMAS-TrainingData/')

    train_files, train_labels, test_files, test_labels = fit_train_test(train_files, train_labels, test_files,
                                                                        test_labels)

    print("test_labels length: ", len(test_labels))
    print("test_files length: ", len(test_files))
    print()
    print("train_labels length: ", len(train_labels))
    print("train_files length: ", len(train_files))

    train_classes, encoded_classes = label_encoder(train_labels)
    test_classes = label_encoder_for_test(encoded_classes, test_labels)

    # train_set = get_feature_vector(train_files)
    train_set = train_files
    test_set = get_feature_vector(test_files)

    # read_write_data.create_data_file(train_set, train_labels)

    knn(train_set, train_classes, test_set, test_classes)
    svm(train_set, train_classes, test_set, test_classes)
    random_forest(train_set, train_classes, test_set, test_classes, encoded_classes)
