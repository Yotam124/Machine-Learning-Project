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
            result = re.search(r'\[[a-z]+\](\[[a-z]+_[a-z]+\])*', filename).group()
            # result = filename.split('_')
            # result = result[0]
            # result = result.split('\\')
            # result = result[len(result)-1]
        else:
            result = re.search(r'(\[[a-z]+\](\[[a-z]+\])*)', filename).group()
        train_labels.append(result)
        if not classes.__contains__(result):
            classes.append(result)
    return [train_files, train_labels]


def load_test_set(genre='', by_genre=False, test_path='./IRMAS-TrainingData/'):
    # Get Audio Files for test
    test_files = []
    test_labels = []
    classes = [
        'cello',
        'clarinet',
        'flute',
        'piano',
        'gac',
        'gel',
        'org',
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
    print("found %d audio files in %s" % (len(test_files), test_path))
    return [test_files, test_labels]


def load_by_genre(path='./IRMAS-TrainingData/'):
    # Get Audio Files for train
    train_files = []
    # regex_pattern = re.compile(r'^\[[a-z]+\](\[[a-z]+_[a-z]+\])+')
    # regex_pattern = re.compile(r'^\[[a-z]+\](\[pop_roc\])')
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.wav'):
            train_files.append(os.path.join(root, filename))

    print("found %d audio files in %s" % (len(train_files), path))

    # Get Labels for train-set
    train_labels = []
    classes = []
    for filename in train_files:
        result = re.search(r'\[[a-z]+\](\[[a-z]+_[a-z]+\])*', filename).group()
        result = result.replace('[', '-').replace(']', '')
        result = result.split('-')
        label = 'no_genre'
        for r in result:
            if r.__contains__('_'):
                label = r
                break
        train_labels.append(label)
        if not classes.__contains__(result):
            classes.append(result)
    return [train_files, train_labels]


def read_data_file(path='./train_test_data.txt'):
    train_set = []
    train_labels = []

    with open(path, 'r') as file_handle:
        start_read_labels = False
        for line in file_handle:
            vector = []
            if line.strip() == 'train_set':
                vector = []
                for v in file_handle:
                    if v.strip() == 'train_classes':
                        start_read_labels = True
                        break
                    if v.strip() == '':
                        train_set.append(vector)
                        vector = []
                    else:
                        sp = v.split()
                        for num in sp:
                            num = num.replace('[', '').replace(']', '')
                            if num == '':
                                continue
                            vector.append(float(num))

            if start_read_labels:
                for label in file_handle:
                    train_labels.append(label.strip())

    return [train_set, train_labels]


def create_data_file(train_set, train_classes, file_name='train_test_data.txt'):
    with open(file_name, 'w') as file_handle:

        file_handle.write('train_set\n')
        for s in train_set:
            file_handle.write('%s\n' % s)
            file_handle.write('\n')

        file_handle.write('train_classes\n')
        for s in train_classes:
            file_handle.write('%s\n' % s)
