import numpy as np

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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
    labelencoder = LabelEncoder()
    labelencoder.fit(labels)
    print(len(labelencoder.classes_), "classes:", ", ".join(list(labelencoder.classes_)))
    classes_num = labelencoder.transform(labels)
    return [classes_num, list(labelencoder.classes_)]


def label_encoder_for_test(encoded_classes, test_labels):
    classes_num = []
    for label in test_labels:
        if label not in encoded_classes:
            classes_num.append(len(test_labels)*2)
        else:
            classes_num.append(encoded_classes.index(label))

    labelencoder = LabelEncoder()
    labelencoder.fit(test_labels)
    print(len(labelencoder.classes_), "classes:", ", ".join(list(labelencoder.classes_)))

    return np.array(classes_num)


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
