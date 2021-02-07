import fnmatch
import os


def read_data_file(path='./train_test_data.txt'):
    train_files = []
    train_labels = []
    test_files = []
    test_labels = []
    with open(path, 'r') as file_handle:
        for line in file_handle:
            if line.__eq__('train_set'):
                train_files.append(line)

            if line.__eq__('train_classes'):
                train_labels.append(line)

            if line.__eq__('test_set'):
                test_files.append(line)

            if line.__eq__('test_classes'):
                test_labels.append(line)

    return [train_files, train_labels, test_files, test_labels]


def create_data_file(train_set, train_classes):
    with open('train_test_data.txt', 'w') as file_handle:

        file_handle.write('train_set\n')
        for s in train_set:
            file_handle.write('%s\n' % s)

        file_handle.write('train_classes\n')
        for s in train_classes:
            file_handle.write('%s\n' % s)

# if __name__ == '__main__':
#     path = './sing/'
#     for root, dirnames, filenames in os.walk(path):
#         for filename in fnmatch.filter(filenames, '*.wav'):
#             os.renames(os.path.join(root, filename), os.path.join(root, f'voice_{filename}'))
