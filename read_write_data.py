import fnmatch
import os


def read_data_file(path='./train_test_data.txt'):
    train_set = []
    train_labels = []

    with open(path, 'r') as file_handle:
        for line in file_handle:
            vector = []
            if line == 'train_set':
                for v in file_handle:
                    print(v)
                    # train_set.append(line)

            if line.__eq__('train_classes'):
                train_labels.append(line)

    return [train_set, train_labels]


def create_data_file(train_set, train_classes):
    with open('train_test_data.txt', 'w') as file_handle:

        file_handle.write('train_set\n')
        for s in train_set:
            file_handle.write('%s\n' % s)

        file_handle.write('train_classes\n')
        for s in train_classes:
            file_handle.write('%s\n' % s)


if __name__ == '__main__':
    read_data_file()