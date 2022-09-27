"""
These source files utilise functions from the 'MNE', 'scikit-learn' and 'Tensorflow' external python libraries.
As such, links are placed here to give credit to the original distributors of these libraries:
MNE:            https://mne.tools/stable/index.html#
Tensorflow:     https://www.tensorflow.org/
scikit-learn:   https://scikit-learn.org/stable/index.html#
"""

import mne
import os
import matplotlib.pyplot as plt
import csv
import sys

"""
This script performs the pre-processing and segmentation of the data, before a sliding window is applied. First it 
applies the band-pass filter, then segments the data. For each data segment it then generates the results of ICA and 
displays them for the user to manually select to exclude components perceived as artifacts

Inputs:
    - input_directory_path:  
    The path to the directory containing the raw .cnt files

    - output_directory_path:                     
    The directory path in which the .fif files will be written for each participant and segment

    - times_csv_path:
    The path to the .csv file containing the manually extracted time-stamps to segment the raw data
"""

if __name__ == "__main__":
    # Reads the command line arguments
    if len(sys.argv) != 4:
        exit("Incorrect number of arguments. Arguments should be of the form: <input_directory_path> <output_directory_path> <times_csv_path> \nExiting Program")
    random_state = 97
    input_path = str(sys.argv[1])
    output_path = str(sys.argv[2])
    times_csv_path = str(sys.argv[3])

    # Opens and iterates through entries in the time stamps .csv file
    with open(times_csv_path, 'r', newline='') as time_file:
        reader = csv.reader(time_file, delimiter=',')

        # Creates the output directory if it did not already exist
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Iterates through the participants
        for row in reader:
            dir_path = output_path + "/" + row[0]

            # Creates a subdirectory for the participant if it does not already exist
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            # Reads the raw file and generates a PSD for it, which is shown to the user
            raw = mne.io.read_raw_cnt(input_path + "/" + row[0] + ".cnt", preload=True, verbose=0)
            raw.plot_psd(average=True)

            # Applies the band-pass filter and generates a PSD for it, which is shown to the user
            raw.filter(l_freq=0.2, h_freq=40, verbose=0)
            raw.plot_psd(average=True)

            # Extracts the first segment
            if row[3] != '-' and row[4] != '-':
                sub_2 = raw.copy().crop(float(row[3]), float(row[4]), verbose=0)

                # Applies ICA, displaying a window for the user to select channels which should be excluded
                ica = mne.preprocessing.ICA(max_iter='auto', random_state=random_state, verbose=0, n_components=18)
                ica.fit(sub_2, verbose=0)
                sub_2.load_data(verbose=0)
                ica.plot_sources(sub_2)
                plt.show()
                ica.apply(sub_2)

                # Writes the resulting segment
                sub_2.save(dir_path + "/2_raw.fif", overwrite=True)

            # Applies the same for the second segment
            if row[5] != '-' and row[6] != '-':
                sub_3 = raw.copy().crop(float(row[5]), float(row[6]))

                # Applies ICA, displaying a window for the user to select channels which should be excluded
                ica = mne.preprocessing.ICA(max_iter='auto', random_state=random_state, verbose=0, n_components=18)
                ica.fit(sub_3, verbose=0)
                sub_3.load_data(verbose=0)
                ica.plot_sources(sub_3)
                plt.show()
                ica.apply(sub_3)

                # Writes the resulting segment
                sub_3.save(dir_path + "/3_raw.fif", overwrite=True)

            # Applies the same for the third segment
            if row[7] != '-' and row[8] != '-':
                sub_4 = raw.copy().crop(float(row[7]), float(row[8]))

                # Applies ICA, displaying a window for the user to select channels which should be excluded
                ica = mne.preprocessing.ICA(max_iter='auto', random_state=random_state, verbose=0, n_components=18)
                ica.fit(sub_4, verbose=0)
                sub_4.load_data(verbose=0)
                ica.plot_sources(sub_4)
                plt.show()
                ica.apply(sub_4)

                # Writes the resulting segment
                sub_4.save(dir_path + "/4_raw.fif", overwrite=True)

            # Applies the same for the fourth segment
            if row[9] != '-' and row[10] != '-':
                sub_5 = raw.copy().crop(float(row[9]), float(row[10]))

                # Applies ICA, displaying a window for the user to select channels which should be excluded
                ica = mne.preprocessing.ICA(max_iter='auto', random_state=random_state, verbose=0, n_components=18)
                ica.fit(sub_5, verbose=0)
                sub_5.load_data(verbose=0)
                ica.plot_sources(sub_5)
                plt.show()
                ica.apply(sub_5)

                # Writes the resulting segment
                sub_5.save(dir_path + "/5_raw.fif", overwrite=True)