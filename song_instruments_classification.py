import itertools

import numpy as np

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

# ML Functions
from ML_Algorithms import knn, svm, random_forest, knn_song_clf

# Audio
from audio_analysis import get_feature_vector
from audio_analysis import label_encoder
from audio_analysis import label_encoder_for_test

from load_files import load_train_set, load_test_set, read_data_file

from side_funcs import fit_train_test, cut_a_single_song


def run():
    train_files, train_labels = read_data_file()

    test_files, test_labels = cut_a_single_song('./IRMAS-TestingData/Part1/01 Just One Of Those Things-14')

    print("test_labels length: ", len(test_labels))
    print("test_files length: ", len(test_files))
    print()
    print("train_labels length: ", len(train_labels))
    print("train_files length: ", len(train_files))
    #
    train_classes, encoded_classes = label_encoder(train_labels)
    # test_classes = label_encoder_for_test(encoded_classes, test_labels)
    #
    # train_set = get_feature_vector(train_files)
    train_set = train_files
    test_set = get_feature_vector(test_files)

    knn_song_clf(train_set, train_classes, test_set, test_labels, encoded_classes)
