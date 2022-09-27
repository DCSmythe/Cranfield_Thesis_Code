"""
These source files utilise functions from the 'MNE', 'scikit-learn' and 'Tensorflow' external python libraries.
As such, links are placed here to give credit to the original distributors of these libraries:
MNE:            https://mne.tools/stable/index.html#
Tensorflow:     https://www.tensorflow.org/
scikit-learn:   https://scikit-learn.org/stable/index.html#
"""

import numpy as np
import csv
import tensorflow as tf
import mne
import sys
import os
from sklearn.preprocessing import scale

"""
Takes one of the .data files as an input and performs machine learning on each of the features of each individual 
channel, using a DNN. It then writes the results of 10-cross validation to a .txt file

Inputs:
    - input_file:  
    The path to the .data file input library. It can be any of 'urban.data' 'motorway.data' or 'high_low.data'. 
    'four-class.data' will not run in this file as the input file must only have two classes

    - output_directory_path:                     
    The directory path in which the .txt file is written to

    - epoch_count:
    The number of iterations the model should perform while learning
"""

if __name__ == "__main__":
    # Reads the command line arguments
    if len(sys.argv) != 4:
        exit("Incorrect number of arguments. Arguments should be of the form: <input_file> <output_directory_path> <epoch_count> \nExiting Program")
    input_file_path = str(sys.argv[1])
    output_path = str(sys.argv[2])
    epochs = int(sys.argv[3])

    # Assumes that ICA_out is in the same directory to get the channel names
    raw = mne.io.read_raw_fif("ICA_out/90390/2_raw.fif", preload=True, verbose=False)
    channel_names = raw.info["ch_names"]

    # Reads the input data file
    print("Reading Data File...")
    with open(input_file_path, newline='') as f_train:
        data = np.array(list(csv.reader(f_train)))
        np.random.shuffle(data)

        # Extracts the labels and feature inputs
        labels = data[:, -2:-1].astype(np.float64)
        data = data[:, :-2].astype(np.float64)

        # Writes the labels in a one-hot encoding
        labels2 = []
        for entry in labels:
            if entry[0] == 0:
                labels2.append([1, 0])
            else:
                labels2.append([0, 1])
        labels = np.array(labels2)

    # Scales the data using the sklearn function
    data = scale(data)

    res = []
    # Iterates over the channels
    for i in range(24):
        res2 = []
        for j in range(10):
            # Declares the index ranges for each channel's features
            upper = int(round(float(len(data[0])) * float(i) / 24.0) + 4)
            lower = int(round(float(len(data[0])) * float(i) / 24.0))
            upper2 = int(round(float(len(data)) * float(j + 1) / 10.0))
            lower2 = int(round(float(len(data)) * float(j) / 10.0))

            # Declares the training and validation sets
            data_train = np.append(data[0:lower2, lower:upper], data[upper2:, lower:upper], axis=0)
            labels_train = np.append(labels[0:lower2, :], labels[upper2:, :], axis=0)
            data_test = np.array(data[lower2:upper2, lower:upper])
            labels_test = np.array(labels[lower2:upper2, :])

            # Appends the remaining features according to the index ranges and .data format
            for j in range(1, 4):
                data_train = np.append(data_train,
                                       np.append(data[0:lower2, lower + j * 4 * 24:upper + j * 4 * 24],
                                                 data[upper2:, lower + j * 4 * 24:upper + j * 4 * 24], axis=0),
                                       axis=1)
                data_test = np.append(data_test, data[lower2:upper2, lower + j * 4 * 24:upper + j * 4 * 24], axis=1)

            X = tf.constant(data_train, dtype=tf.float64)
            Y = tf.constant(labels_train, dtype=tf.float64)
            X2 = tf.constant(data_test, dtype=tf.float64)
            Y2 = tf.constant(labels_test, dtype=tf.float64)

            # Declares the keras model
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(10,
                                            input_dim=len(data_train[0]),
                                            activation='relu'))
            model.add(tf.keras.layers.Dense(10,
                                            activation='relu'))
            model.add(tf.keras.layers.Dense(2,
                                            activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

            # Fits the data to the model
            history = model.fit(
                X,
                Y,
                epochs=epochs,
                batch_size=200,
                verbose=1,
                validation_data=(X2, Y2),
            )

            # Evaluates the data on the validation set
            results = model.evaluate(X2, Y2)
            res2.append(results)

        # Determines the mean and standard deviations of the 10-cross validation
        res.append([np.mean(res2, axis=0)[1], np.std(res2, axis=0)[1]])

    # Creates the output directory if it did not already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Writes each channel's results to the output file
    with open(output_path + "/values.txt", "w", newline='') as f_res:
        for i in range(len(res)):
            f_res.write(channel_names[i] + ": " + str(100 * np.array(res[i])[0]) + " +- " + str(100 * np.array(res[i])[1]) + "\n")
