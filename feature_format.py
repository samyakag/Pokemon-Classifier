import tensorflow.python.platform
import csv
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
np.set_printoptions(threshold=sys.maxint)

def featureFormat(filename):
    # List contains features to be selected from the csv file
    features_list = ["Name", "Type 1", "Type 2",
                     "Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    # Arrays to hold the labels and feature vectors.
    features = []
    labels = []
    all_types = []
    # read the above columns from the csv file
    df = pd.read_csv(filename, usecols=features_list)
    labels = df[df.columns[1]]
    # print(df[df.columns[1]])
    features = df[df.columns[3:]].values.astype(float)
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(features)
    features = pd.DataFrame(x_scaled)
    # features.plot(kind='bar')
    # features = (features - features.mean()) / \
    #     (features.max() - features.min())
    all_types = np.unique(labels)
    # print(all_types.shape)
    # perform one hot encoding on labels

    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    labels = one_hot_encode(labels, len(all_types))

    # Convert the array of float arrays into a numpy float matrix.
    features = np.matrix(features).astype(np.float32)
    # labels = np.matrix(labels).astype(str)
    # all_types = np.transpose(np.matrix(all_types).astype(str))
    # # Convert the array of int labels into a numpy array.
    # labels_np = np.array(labels).astype(dtype=np.uint8)
    # print(labels)
    # tf.Print(labels, [labels])
    return features, labels, all_types


def one_hot_encode(labels, n_unique_labels):
    n_labels = len(labels)
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
