import itertools
import operator

import numpy as np
import math
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# KNN
from sklearn.neighbors import KNeighborsClassifier

# SVM
from sklearn.svm import LinearSVC, SVC
import joblib

# Random Forest
from sklearn.ensemble import RandomForestClassifier


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    """
    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# ------------------------------------------------ KNN ------------------------------------------------
def knn(train_set, train_classes, test_set, test_classes, encoded_classes):
    # Machine Learning Parameters
    n_neighbors = 1  # Number of neighbors for kNN Classifier

    # KNN Classifier
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # kNN
    model_knn.fit(train_set, train_classes)

    # Predict using the Test Set
    predicted_labels = model_knn.predict(test_set)

    print('------------------------ KNN - Evaluation ------------------------')
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
    print()

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_classes, predicted_labels)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(18, 13))
    plot_confusion_matrix(cnf_matrix, classes=encoded_classes,
                          title='KNN')


# ------------------------------------------------ SVM ------------------------------------------------
def svm(train_set, train_classes, test_set, test_classes, encoded_classes):
    # model_svm = LinearSVC(random_state=0, tol=1e-5, max_iter=5000)
    svclassifier = SVC(kernel='rbf', C=10.0, gamma=0.1)

    # SVM
    # model_svm.fit(train_set, train_classes);
    svclassifier.fit(train_set, train_classes)

    # Save
    joblib.dump(svclassifier, 'trainedSVM.joblib')
    # Load
    # svclassifier = joblib.load('filename.joblib')

    # Predict using the Test Set
    # predicted_labels = model_svm.predict(test_set)
    predicted_labels = svclassifier.predict(test_set)

    print('------------------------ SVM - Evaluation ------------------------')
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
    print()
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_classes, predicted_labels)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(18, 13))
    plot_confusion_matrix(cnf_matrix, classes=encoded_classes,
                          title='SVM')


# ------------------------------------------------ Random Forest ------------------------------------------------
def random_forest(train_set, train_classes, test_set, test_classes, encoded_classes):
    n_estimators = 100

    model = RandomForestClassifier(n_estimators=n_estimators)

    model.fit(train_set, train_classes)

    predicted_labels = model.predict(test_set)

    print('------------------------ Random Forest - Evaluation ------------------------')
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
    print()

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_classes, predicted_labels)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(18, 13))
    plot_confusion_matrix(cnf_matrix, classes=encoded_classes,
                          title='Random Forest')


def knn_song_clf(train_set, train_classes, test_set, test_classes, encoded_classes):
    # Machine Learning Parameters
    n_neighbors = 1  # Number of neighbors for kNN Classifier

    # KNN Classifier
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # kNN
    model_knn.fit(train_set, train_classes)

    # Predict using the Test Set
    predicted_labels = model_knn.predict(test_set)

    predicted_decoded_labels = []
    for label in predicted_labels:
        predicted_decoded_labels.append(encoded_classes[label])

    label_count = Counter(predicted_decoded_labels)
    threshold_label = []
    for label_i in label_count:
        if label_count[label_i] >= math.sqrt(len(label_count)) - 1:
            threshold_label.append(label_i)

    print('Threshold: ', math.sqrt(len(label_count)) - 1)
    print('Predicted labels: ', predicted_decoded_labels)
    print('Predicted labels: ', label_count)
    print('Threshold label: ', threshold_label)
    print('True labels: ', test_classes)
