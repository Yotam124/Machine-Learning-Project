# System
import fnmatch
import os

import re


def load_train_set(train_path='./data-set/', single_instrument=True):
    # Get Audio Files for train
    train_files = []
    # regex_pattern = re.compile(r'^\[[a-z]+\](\[[a-z]+_[a-z]+\])+')
    # regex_pattern = re.compile(r'^\[[a-z]+\](\[pop_roc\])')
    for root, dirnames, filenames in os.walk(train_path):
        for filename in fnmatch.filter(filenames, '*.wav'):
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
            result = result[len(result) - 1]

        else:
            result = re.search(r'(\[[a-z]+\](\[[a-z]+\])*)', filename).group()
        train_labels.append(result)
        if not classes.__contains__(result):
            classes.append(result)
    return [train_files, train_labels]


def load_test_set(genre, by_genre=False, test_path='./IRMAS-TrainingData/'):
    # Get Audio Files for test
    test_files = []
    test_labels = []
    classes = [
        'cello',
        'clarinet',
        'flute',
        'piano',
        'saxophone',
        'trumpet',
        'violin',
        'voice'
    ]

    regex_pattern = re.compile(fr'^\[[a-z]+\](\[{genre}\])') if by_genre else ''
    for root, dirnames, filenames in os.walk(test_path):
        # Audio file
        for filename in fnmatch.filter(filenames, '*.wav'):
            pattern = re.search(regex_pattern, filename)
            if pattern is not None:
                test_files.append(os.path.join(root, filename))
                filename_class = re.search(r'\[[a-z]+\]', filename).group()
                filename_class = filename_class[1:4]

                for name in classes:
                    if fnmatch.fnmatchcase(filename_class, '*' + name[0:3] + '*'):
                        test_labels.append(name)
                        break
                else:
                    test_labels.append('other')
        # Text file
        # for filename in fnmatch.filter(filenames, '*.txt'):
        #     f = open(os.path.join(root, filename), 'r')
        #     data = f.read().split()
        #     instruments = ''
        #     for s in data:
        #         instruments += '[' + s + ']'
        #     test_labels.append(instruments)
    print("found %d audio files in %s" % (len(test_files), test_path))
    return [test_files, test_labels]
