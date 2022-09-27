"""
These source files utilise functions from the 'MNE', 'scikit-learn' and 'Tensorflow' external python libraries.
As such, links are placed here to give credit to the original distributors of these libraries:
MNE:            https://mne.tools/stable/index.html#
Tensorflow:     https://www.tensorflow.org/
scikit-learn:   https://scikit-learn.org/stable/index.html#
"""

import numpy as np
import csv
import sklearn
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn import svm
from sklearn.preprocessing import scale

"""
Takes one of the .data files as an input and performs machine learning on the data within that file. It then writes the
results of 10-cross or cross-participant validation to a .txt file in addition to a classification matrix

Inputs:
    - input_file:  
    The path to the .data file input library. I can be any one of the .data files produced by the generate_ml scripts

    - output_directory_path:                     
    The directory path in which the .txt file and classification matrix are written to

    - cross_participant_bool:
    A boolean representing whether cross-participant validation is performed. 0 performs 10-fold cross validation, 1 
    performs cross-participant validation
"""

if __name__ == "__main__":
    # Reads the command line arguments
    if len(sys.argv) != 4:
        exit("Incorrect number of arguments. Arguments should be of the form: <input_file> <output_directory_path> <cross_paritipant_bool> \nExiting Program")
    input_file_path = str(sys.argv[1])
    output_path = str(sys.argv[2])
    cross_part = bool(int(sys.argv[3]))

    accs = []
    precs = []
    recs = []
    full_results = []

    # Reads the input data file
    print("Reading Data File...")
    with open(input_file_path, newline='') as f_train:
        data = np.array(list(csv.reader(f_train)))
        if not cross_part:
            np.random.shuffle(data)

    # Extracts the labels and feature inputs
    labels = data[:, -2:-1].astype(np.float64).flatten()
    participants = data[:, -1:].flatten()
    data = data[:, :-2].astype(np.float64)

    # Scales the data using sklearn functions
    scaler1 = sklearn.preprocessing.MinMaxScaler()
    data = scaler1.fit_transform(data)

    # Determines the indices of each participant for cross-participant validation
    upper_lower_participant_idxs = []
    for participant in set(participants):
        upper_lower_participant_idxs.append((list(participants).index(participant),
                                             len(participants) - 1 - list(participants)[::-1].index(participant)))

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
        X = np.append(data[0:lower, :], data[upper:, :], axis=0)
        Y = np.append(labels[0:lower], labels[upper:], axis=0)
        X2 = data[lower:upper, :]
        Y2 = labels[lower:upper]

        # Declares the SVM model
        mdl = sklearn.svm.LinearSVC(verbose=2, max_iter=200000, class_weight='balanced', tol=1e-11)

        # Fits the data to the model
        mdl.fit(X, Y)

        # Evaluates the data on the validation set
        Y3 = mdl.predict(X2)
        accs.append(sklearn.metrics.accuracy_score(Y2, Y3))
        precs.append(sklearn.metrics.precision_score(Y2, Y3, average="macro"))
        recs.append(sklearn.metrics.recall_score(Y2, Y3, average="macro"))

        # Counts the occurrences of each class for the classification matrix
        results_array = [[0 for _ in range(int(np.max(labels) - np.min(labels) + 1))] for _ in range(int(np.max(labels) - np.min(labels) + 1))]
        for j in range(len(Y3)):
            results_array[int(Y2[j])][int(Y3[j])] += 1

        full_results.append(results_array)

    # Creates the output directory if it did not already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Writes the mean accuracy, precision and recalls to the .txt file in the output directory
    with open(output_path + "/values.txt", "w", newline='') as f_res:
        f_res.write("Accuracy:  " + str(np.mean(accs)) + " +- " + str(np.std(accs)) + "\n")
        f_res.write("Precision: " + str(np.mean(precs)) + " +- " + str(np.std(precs)) + "\n")
        f_res.write("Recall:    " + str(np.mean(recs)) + " +- " + str(np.std(recs)) + "\n")

    # Determines the means for the classification matrix
    mean_results = np.mean(full_results, axis=0)
    summ = np.sum(np.array(mean_results), axis=1)
    for i in range(len(mean_results)):
        for j in range(len(mean_results)):
            mean_results[i][j] = mean_results[i][j] / summ[i]

    # Creates the classification matrix
    if len(mean_results) == 4:
        df_cm = pd.DataFrame(mean_results,
                             ["Motorway No Cars", "Motorway Cars", "Urban No Cars", "Urban Cars"],
                             ["Motorway No Cars", "Motorway Cars", "Urban No Cars", "Urban Cars"])
    elif len(mean_results) == 2:
        df_cm = pd.DataFrame(mean_results, ["Low Workload", "High Workload"], ["Low Workload", "High Workload"])
    plt.figure(figsize=(20, 15))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')  # font size
    plt.xlabel("Predicted Classification")
    plt.ylabel("True Classification")
    plt.yticks(rotation=0)
    plt.savefig(output_path + "/class_matrix.png")
    plt.show()
