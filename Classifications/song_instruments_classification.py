import itertools

import numpy as np

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

# ML Functions
from ML_Algorithms import knn_song_clf, svm_song_clf, rf_song_clf, knn, svm, random_forest

# Audio
from audio_analysis import get_feature_vector, get_feature_vector_2
from audio_analysis import label_encoder
from audio_analysis import label_encoder_for_test

from load_files import load_train_set, load_test_set, read_data_file

from side_funcs import fit_train_test, cut_a_single_song


def run():
    # train_files, train_labels = read_data_file('./train_test_data_combined.txt')
    train_files, train_labels = load_train_set('./IRMAS-TrainingData/', single_instrument=True)

    train_set = get_feature_vector(train_files)

    song_name = "(02) dont kill the whale-1"
    test_files, test_labels = cut_a_single_song(f'./IRMAS-TestingData/Part1/{song_name}')

    print("test_labels length: ", len(test_labels))
    print("test_files length: ", len(test_files))
    print()
    print("train_labels length: ", len(train_labels))
    print("train_files length: ", len(train_files))
    #
    train_classes, encoded_classes = label_encoder(train_labels)
    test_classes = label_encoder_for_test(encoded_classes, test_labels)

    test_set = get_feature_vector_2(test_files)

    knn(train_set, train_classes, test_set, test_classes, encoded_classes)
    svm(train_set, train_classes, test_set, test_classes, encoded_classes)
    random_forest(train_set, train_classes, test_set, test_classes, encoded_classes)
