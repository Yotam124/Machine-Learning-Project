import fnmatch
import os

from pydub import AudioSegment
from pydub.utils import make_chunks


def fit_train_test(train_files, train_labels, test_files, test_labels):
    train_indexes = []
    for label_i in range(len(train_labels)):
        if train_labels[label_i] not in test_labels:
            # train_labels[label_i] = "other"
            train_indexes.append(label_i)

    test_indexes = []
    for label_i in range(len(test_labels)):
        if test_labels[label_i] not in train_labels:
            # test_labels[label_i] = "other"
            test_indexes.append(label_i)

    for i in range(len(train_indexes)):
        del train_files[train_indexes[i]]
        del train_labels[train_indexes[i]]
        train_indexes = [x - 1 for x in train_indexes]

    for i in range(len(test_indexes)):
        del test_files[test_indexes[i]]
        del test_labels[test_indexes[i]]
        test_indexes = [x - 1 for x in test_indexes]
    return [train_files, train_labels, test_files, test_labels]


def cut_to_chunks(path, label):
    for root, dirnames, filenames in os.walk(path):
        for i, filename in enumerate(fnmatch.filter(filenames, '*.wav')):
            print(filename)
            my_audio = AudioSegment.from_file(os.path.join(root, filename), "wav")
            chunk_length_ms = 1000  # pydub calculates in millisec
            chunks = make_chunks(my_audio, chunk_length_ms)  # Make chunks of one sec

            # Export all of the individual chunks as wav files

            for j, chunk in enumerate(chunks):
                chunk_name = f'{path}/{label}_chunk{i}-{j}.wav'.format(i)
                print("exporting", chunk_name)
                chunk.export(chunk_name, format="wav")


def cut_a_single_song(path):
    song = os.walk(path)
    my_audio = AudioSegment.from_file(song, "wav")
    chunk_length_ms = 1000  # pydub calculates in millisec
    chunks = make_chunks(my_audio, chunk_length_ms)  # Make chunks of one sec

    return chunks
    # chunks_list = []
    # for i, chunk in enumerate(chunks):
    #     chunk_name = f'{path}_chunk{0}.wav'.format(i)
    #     print("cut", chunk_name)
    #     chunk.export(chunk_name, format="wav")
