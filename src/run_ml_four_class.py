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
import seaborn as sn
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import scale

"""
Takes one of the .data files as an input and performs machine learning on the 4 class data within that file. It then
writes the results of 10-cross or cross-participant validation to a .txt file, and draws a number of graphs representing
the accuracy, precision and recall during training, in addition to a classification matrix

Inputs:
    - input_file:  
    The path to the .data file input library. It can only be the 'four-class.data' file, as it requires there to be 4 
    classes.

    - output_directory_path:                     
    The directory path in which the .txt file, graphs and classification matrix are written to

    - epoch_count:
    The number of iterations the model should perform while learning

    - cross_participant_bool:
    A boolean representing whether cross-participant validation is performed. 0 performs 10-fold cross validation, 1 
    performs cross-participant validation
"""

if __name__ == "__main__":
    # Reads the command line arguments
    if len(sys.argv) != 5:
        exit("Incorrect number of arguments. Arguments should be of the form: <input_file> <output_directory_path> <epoch_count> <cross_paritipant_bool> \nExiting Program")
    input_file_path = str(sys.argv[1])
    output_path = str(sys.argv[2])
    epochs = int(sys.argv[3])
    cross_part = bool(int(sys.argv[4]))

    # Reads the input data file
    print("Reading Data File...")
    with open(input_file_path, 'r', newline='') as f_train:
        data = np.array(list(csv.reader(f_train)))
        if not cross_part:
            np.random.shuffle(data)

        # Extracts the labels and feature inputs
        labels = data[:, -2:-1].astype(np.float64)
        participants = data[:, -1:].flatten()
        data = data[:, :-2].astype(np.float64)

        # Writes the labels in a one-hot encoding
        labels2 = []
        for entry in labels:
            if entry[0] == 0:
                labels2.append([1, 0, 0, 0])
            elif entry[0] == 1:
                labels2.append([0, 1, 0, 0])
            elif entry[0] == 2:
                labels2.append([0, 0, 1, 0])
            else:
                labels2.append([0, 0, 0, 1])
        labels = np.array(labels2)

    # Determines the indices of each participant for cross-participant validation
    upper_lower_participant_idxs = []
    for participant in set(participants):
        upper_lower_participant_idxs.append((list(participants).index(participant),
                                             len(participants) - 1 - list(participants)[::-1].index(participant)))

    # Scales the data using the sklearn function
    data = scale(data)

    full_results = []
    histories = []
    res = []

    # Determines the number of iterations necessary of the validation method used
    if cross_part:
        itrs = 20
    else:
        itrs = 10

    # Performs the validation of the training
    for i in range(itrs):
        # Declares the index ranges for the training and validation sets
        if cross_part:
            lower, upper = upper_lower_participant_idxs[i]
        else:
            upper = int(round(float(len(data)) * float(i + 1) / 10.0))
            lower = int(round(float(len(data)) * float(i) / 10.0))

        # Declares the training and validation sets
        X = tf.constant(np.append(data[0:lower, :], data[upper:, :], axis=0), dtype=tf.float64)
        Y = tf.constant(np.append(labels[0:lower, :], labels[upper:, :], axis=0), dtype=tf.float64)
        X2 = tf.constant(data[lower:upper, :], dtype=tf.float64)
        Y2 = tf.constant(labels[lower:upper, :], dtype=tf.float64)

        # Declares the keras model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(200,
                                        input_dim=len(data[0]),
                                        activation='relu'))
        model.add(tf.keras.layers.Dense(100,
                                        activation='linear'))
        model.add(tf.keras.layers.Dense(4,
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
        res.append(results)
        Y2 = Y2.numpy()
        Y3 = model(X2).numpy()

        # Counts the occurrences of each class for the classification matrix
        results_array = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for i in range(len(Y3)):
            results_array[np.argmax(Y2[i])][np.argmax(Y3[i])] += 1

        full_results.append(results_array)
        histories.append(history)

    # Creates the output directory if it did not already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Writes the mean accuracy, precision and recalls to the .txt file in the output directory
    with open(output_path + "/values.txt", "w", newline='') as f_res:
        f_res.write(
            "Accuracy:  " + str(np.mean(np.array(res), axis=0)[1]) + " +- " + str(
                np.std(np.array(res), axis=0)[1]) + "\n")
        f_res.write(
            "Precision: " + str(np.mean(np.array(res), axis=0)[2]) + " +- " + str(
                np.std(np.array(res), axis=0)[2]) + "\n")
        f_res.write(
            "Recall:    " + str(np.mean(np.array(res), axis=0)[3]) + " +- " + str(
                np.std(np.array(res), axis=0)[3]) + "\n")

    # Draws the accuracy graph
    plt.figure(figsize=(20, 15))
    for history in histories:
        acc = history.history['val_accuracy']
        acc.insert(0, 0)
        plt.plot(range(epochs+1), acc)
    plt.xlabel("Epochs"), plt.ylabel("Accuracy")
    plt.title('Accuracy')
    plt.ylim([0.0, 1.0])
    plt.savefig(output_path + "/accuracy.png")
    plt.show()

    # Draws the precision graph
    plt.figure(figsize=(20, 15))
    i = 0
    for history in histories:
        if i != 0:
            prec = history.history['val_precision_' + str(i)]
        else:
            prec = history.history['val_precision']
        prec.insert(0, 0)
        plt.plot(range(epochs+1), prec)
        i += 1
    plt.xlabel("Epochs"), plt.ylabel("Precision")
    plt.title('Precision')
    plt.ylim([0.0, 1.0])
    plt.savefig(output_path + "/precision.png")
    plt.show()

    # Draws the recall graph
    plt.figure(figsize=(20, 15))
    i = 0
    for history in histories:
        if i != 0:
            rec = history.history['val_recall_' + str(i)]
        else:
            rec = history.history['val_recall']
        rec.insert(0, 0)
        plt.plot(range(epochs+1), rec)
        i += 1
    plt.xlabel("Epochs"), plt.ylabel("Recall")
    plt.title('Recall')
    plt.ylim([0.0, 1.0])
    plt.savefig(output_path + "/recall.png")
    plt.show()

    # Determines the means for the classification matrix
    mean_results = np.mean(full_results, axis=0)
    summ = np.sum(np.array(mean_results), axis=1)
    for i in range(4):
        for j in range(4):
            mean_results[i][j] = mean_results[i][j] / summ[i]

    # Creates the classification matrix
    df_cm = pd.DataFrame(mean_results,
                         ["Motorway No Cars", "Motorway Cars", "Urban No Cars", "Urban Cars"],
                         ["Motorway No Cars", "Motorway Cars", "Urban No Cars", "Urban Cars"])
    plt.figure(figsize=(20, 15))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')  # font size
    plt.xlabel("Predicted Classification")
    plt.ylabel("True Classification")
    plt.yticks(rotation=0)
    plt.savefig(output_path + "/class_matrix.png")
    plt.show()
