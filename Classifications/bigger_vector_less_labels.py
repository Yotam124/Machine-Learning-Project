from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import itertools

import numpy as np

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

# ML Functions
from ML_Algorithms import knn, svm, random_forest, linear_regression, adaboost, svm_with_pca

from sklearn.model_selection import train_test_split

# Audio
from audio_analysis import get_feature_vector, get_feature_vector_2
from audio_analysis import label_encoder
from audio_analysis import label_encoder_for_test

from load_files import load_train_set, read_data_file, load_test_set

from side_funcs import fit_train_test


def run():
    data_set, data_labels = load_test_set()

    X = get_feature_vector_2(data_set)

    # pca = PCA().fit(X)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance')
    # plt.show()

    train_set, test_set, train_labels, test_labels = train_test_split(X, data_labels, test_size=0.25)

    train_set, train_labels, test_set, test_labels = fit_train_test(train_set, train_labels, test_set,
                                                                        test_labels)

    train_classes, encoded_classes = label_encoder(train_labels)
    test_classes = label_encoder_for_test(encoded_classes, test_labels)

    knn(train_set, train_classes, test_set, test_classes, encoded_classes)
    svm(train_set, train_classes, test_set, test_classes, encoded_classes)
    random_forest(train_set, train_classes, test_set, test_classes, encoded_classes)
    adaboost(train_set, train_classes, test_set, test_classes, encoded_classes)
