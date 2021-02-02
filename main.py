# Load Files
from load_files import load_train_set
from load_files import load_test_set

# Audio
from audio_analysis import get_feature_vector
from audio_analysis import label_encoder

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report

if __name__ == '__main__':
    train_files, train_labels = load_train_set()
    test_files, test_labels = load_test_set()

    classes_num_train = label_encoder(train_labels)
    classes_num_test = label_encoder(test_labels)

    train_set = get_feature_vector(train_files)
    train_classes = classes_num_train

    test_set = get_feature_vector(test_files)
    test_classes = classes_num_test

    # testset_size = 0.25  # Percentage of data for Testing

    # # Create Train and Test Set
    # splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
    # splits = splitter.split(scaled_feature_vectors, classes_num)
    # for train_index, test_index in splits:
    #     train_set = scaled_feature_vectors[train_index]
    #     test_set = scaled_feature_vectors[test_index]
    #     train_classes = classes_num[train_index]
    #     test_classes = classes_num[test_index]

    # Check Set Shapes
    print()
    print("train_set shape:", train_set.shape)
    print("test_set shape:", test_set.shape)
    print("train_classes shape:", train_classes.shape)
    print("test_classes shape:", test_classes.shape)
    print()

    # ------------------------ KNN ------------------------

    # Machine Learning Parameters
    n_neighbors = 1  # Number of neighbors for kNN Classifier

    # KNN Classifier
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # kNN
    model_knn.fit(train_set, train_classes)

    # Predict using the Test Set
    predicted_labels = model_knn.predict(test_set)

    # ------------------------ Evaluation ------------------------

    # Recall - the ability of the classifier to find all the positive samples
    print("Recall: ", recall_score(test_classes, predicted_labels, average=None))

    # Precision - The precision is intuitively the ability of the classifier not to
    # label as positive a sample that is negative
    print("Precision: ", precision_score(test_classes, predicted_labels, average=None))

    # F1-Score - The F1 score can be interpreted as a weighted average of the precision
    # and recall
    print("F1-Score: ", f1_score(test_classes, predicted_labels, average=None))

    # Accuracy - the number of correctly classified samples
    print("Accuracy: %.2f  ," % accuracy_score(test_classes, predicted_labels, normalize=True),
          accuracy_score(test_classes, predicted_labels, normalize=False))
    print("Number of samples:", test_classes.shape[0])
