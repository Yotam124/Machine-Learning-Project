from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report

import numpy as np
# Audio
from audio_analysis import get_feature_vector
from audio_analysis import label_encoder
from audio_analysis import label_encoder_for_test

from load_files import load_train_set

if __name__ == '__main__':
    data_set, data_labels = load_train_set()
    train_files, test_files, train_labels, test_labels = train_test_split(data_set, data_labels, test_size=0.25)

    # classes_num_train = label_encoder(train_labels)
    # classes_num_test = label_encoder(test_labels)
    train_classes, encoded_classes = label_encoder(train_labels)
    test_classes = label_encoder_for_test(encoded_classes, test_labels)

    train_set = get_feature_vector(train_files)
    test_set = get_feature_vector(test_files)

    # ------------------------ KNN ------------------------
    # Machine Learning Parameters
    n_neighbors = 1  # Number of neighbors for kNN Classifier

    # KNN Classifier
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # kNN
    model_knn.fit(train_set, train_classes)

    # Predict using the Test Set
    predicted_labels = model_knn.predict(test_set)

    # ------------------------ Evaluation - KNN ------------------------

    # Recall - the ability of the classifier to find all the positive samples
    print("Recall: ", recall_score(test_classes, predicted_labels, average=None, zero_division=1))

    # Precision - The precision is intuitively the ability of the classifier not to
    # label as positive a sample that is negative
    print("Precision: ", precision_score(test_classes, predicted_labels, average=None, zero_division=1))

    # F1-Score - The F1 score can be interpreted as a weighted average of the precision
    # and recall
    print("F1-Score: ", f1_score(test_classes, predicted_labels, average=None, zero_division=1))

    # Accuracy - the number of correctly classified samples
    print("Accuracy: %.2f  ," % accuracy_score(test_classes, predicted_labels, normalize=True),
          accuracy_score(test_classes, predicted_labels, normalize=False))
    print("Number of samples:", test_classes.shape[0])
