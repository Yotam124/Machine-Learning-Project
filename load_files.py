# System
import fnmatch
import os

import re

train_path = './data-set/'
test_path = './IRMAS-TestingData/Part1/'


def load_train_set(single_instrument=True):
    # Get Audio Files for train
    train_files = []
    # regex_pattern = re.compile(r'^\[[a-z]+\](\[[a-z]+_[a-z]+\])+')
    # regex_pattern = re.compile(r'^\[[a-z]+\](\[pop_roc\])')
    for root, dirnames, filenames in os.walk(train_path):
        for filename in fnmatch.filter(filenames, '*.wav'):
            print(os.path.join(root, filename))
            # if single_instrument:
            #     if re.search(regex_pattern, filename):
            #         train_files.append(os.path.join(root, filename))
            # else:
            train_files.append(os.path.join(root, filename))

    print("found %d audio files in %s" % (len(train_files), train_path))

    # Get Labels for train-set
    train_labels = []
    classes = []
    for filename in train_files:
        if single_instrument:
            # result = re.search(r'\[[a-z]+\](\[[a-z]+_[a-z]+\])*', filename).group()
            result = filename.split('_')
            result = result[0]
            result = result.split('\\')
            result = result[len(result)-1]

        else:
            result = re.search(r'(\[[a-z]+\](\[[a-z]+\])*)', filename).group()
        train_labels.append(result)
        if not classes.__contains__(result):
            classes.append(result)
    return [train_files, train_labels]


def load_test_set():
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
    return [test_files, test_labels]
