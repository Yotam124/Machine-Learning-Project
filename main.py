# General
import numpy as np
import pickle
import itertools
import re

# System
import os, fnmatch

# Visualization
import seaborn
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# Random Seed
from numpy.random import seed

seed(1)

# Audio
import librosa.display, librosa

# Parameters
# Signal Processing Parameters
fs = 44100  # Sampling Frequency
n_fft = 2048  # length of the FFT window
hop_length = 512  # Number of samples between successive frames
n_mels = 128  # Number of Mel bands
n_mfcc = 13  # Number of MFCCs


# Define Function to Calculate MFCC, Delta_MFCC and Delta2_MFCC
def get_features(y, sr=fs):
    S = librosa.feature.melspectrogram(y, sr=fs, n_mels=n_mels)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    feature_vector = np.mean(mfcc, 1)
    # feature_vector = (feature_vector-np.mean(feature_vector))/np.std(feature_vector)
    return feature_vector


# Encode Labels
def label_encoder(labels):
    print(labels)
    labelencoder = LabelEncoder()
    labelencoder.fit(labels)
    print(len(labelencoder.classes_), "classes:", ", ".join(list(labelencoder.classes_)))
    classes_num = labelencoder.transform(labels)
    return classes_num


# Load audio files, calculate features and create feature vectors
def get_feature_vector(files):
    feature_vectors = []
    sound_paths = []
    for i, f in enumerate(files):
        print("get %d of %d = %s" % (i + 1, len(files), f))
        try:
            y, sr = librosa.load(f, sr=fs)
            y /= y.max()  # Normalize

            if len(y) < 2:
                print("Error loading %s" % f)
                continue
            feat = get_features(y, sr)
            feature_vectors.append(feat)
            sound_paths.append(f)
        except Exception as e:
            print("Error loading %s. Error: %s" % (f, e))

    print("Calculated %d feature vectors" % len(feature_vectors))

    # Scale features using Standard Scaler
    scaler = StandardScaler()
    scaled_feature_vectors = scaler.fit_transform(np.array(feature_vectors))
    print("Feature vectors shape:", scaled_feature_vectors.shape)
    return scaled_feature_vectors


if __name__ == '__main__':

    # Get files in data path
    train_path = './IRMAS-TrainingData/'
    test_path = './IRMAS-TestingData/Part1/'

    # Get Audio Files for train
    train_files = []
    for root, dirnames, filenames in os.walk(train_path):
        for filename in fnmatch.filter(filenames, '*.wav'):
            train_files.append(os.path.join(root, filename))

    print("found %d audio files in %s" % (len(train_files), train_path))

    # Get Labels for train-set
    train_labels = []
    classes = []
    for filename in train_files:
        result = re.search(r'\[[a-z]+\](\[[a-z]+\])*', filename).group()
        train_labels.append(result)
        if not classes.__contains__(result):
            classes.append(result)

    # Get Audio Files for test
    test_files = []
    test_labels = []
    for root, dirnames, filenames in os.walk(test_path):
        # Audio file
        for filename in fnmatch.filter(filenames, '*.wav'):
            test_files.append(os.path.join(root, filename))
        # Text file
        for filename in fnmatch.filter(filenames, '*.txt'):
            f = open(os.path.join(root, filename), 'r')
            data = f.read().split()
            instruments = ''
            for s in data:
                instruments += '[' + s + ']'
            test_labels.append(instruments)
    print("found %d audio files in %s" % (len(test_files), test_path))

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
