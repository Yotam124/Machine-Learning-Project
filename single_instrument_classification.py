import itertools

import numpy as np

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# KNN
from sklearn.neighbors import KNeighborsClassifier

# SVM
from sklearn.svm import LinearSVC, SVC
import joblib

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
    print('------------------------ KNN ------------------------')
    # Machine Learning Parameters
    n_neighbors = 1  # Number of neighbors for kNN Classifier

    # KNN Classifier
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # kNN
    model_knn.fit(train_set, train_classes)

    # Predict using the Test Set
    predicted_labels = model_knn.predict(test_set)

    # ------------------------ KNN - Evaluation ------------------------
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

    # ------------------------ SVM ------------------------
    print('------------------------ SVM ------------------------')
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

    # ------------------------ SVM - Evaluation ------------------------

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

    # ------------------------ RandomForest ------------------------
    print('------------------------ RandomForest ------------------------')

    # ------------------------ RandomForest - Evaluation ------------------------

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_classes, predicted_labels)
    np.set_printoptions(precision=2)


    # Function to Plot Confusion Matrix
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
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

        # Plot non-normalized confusion matrix
        plt.figure(figsize=(18, 13))
        plot_confusion_matrix(cnf_matrix, classes=encoded_classes,
                              title='Confusion matrix, without normalization')