"""
These source files utilise functions from the 'MNE', 'scikit-learn' and 'Tensorflow' external python libraries.
As such, links are placed here to give credit to the original distributors of these libraries:
MNE:            https://mne.tools/stable/index.html#
Tensorflow:     https://www.tensorflow.org/
scikit-learn:   https://scikit-learn.org/stable/index.html#
"""

import mne
import os
import numpy as np
import scipy.integrate
import csv
import sys

"""
This script generates the machine learning dataset to classify the data into four classes

Inputs:
    - input_directory_path:  
    The path to the directory containing the .fif files, representing the segmented and processed data intervals after 
    ICA

    - output_directory_path:                     
    The directory path in which the dataset will be stored as 'four_class.data'
    
    - questionnaire_csv_path:
    The path to the questionnaire results, to extract the order in which the experiments were performed in
"""

if __name__ == "__main__":
    # Reads the command line arguments
    if len(sys.argv) != 4:
        exit("Incorrect number of arguments. Arguments should be of the form: <input_directory_path> <output_directory_path> <questionnaire_csv_path> \nExiting Program")
    input_path = str(sys.argv[1])
    output_path = str(sys.argv[2])
    questionnaire_csv_path = str(sys.argv[3])

    # Creates the output directory if it did not already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Declares the frequency band regions
    theta_band = 4, 8
    alpha_band = 8, 12
    beta_band = 12, 30
    gamma_band = 30, 40
    band_freqs = [theta_band, alpha_band, beta_band, gamma_band]

    # Reads in the experiment order from the questionnaire file
    phase_map = {}
    with open(questionnaire_csv_path, 'r', newline='') as time_file:
        reader = csv.reader(time_file, delimiter=',')
        for row in reader:
            phase_map[row[0]] = row[12]

    # Opens the data files and writes them to a file within the output path
    with open(output_path + "/four_class.data", 'w', newline='') as four_class_f:
        csv_four_class = csv.writer(four_class_f)
        for path in os.listdir("data"):
            x = path.split('.')

            # Iterates over the 4 data segments
            for i in range(0, 4):
                # Loads in the data
                f_path = input_path + "/" + x[0] + "/" + str(i + 2) + "_raw.fif"
                print(f_path)
                raw = mne.io.read_raw_fif(f_path, preload=True, verbose=False)
                channel_names = raw.info["ch_names"]

                # Segments the data into 1 second intervals
                epochs = mne.make_fixed_length_epochs(raw, duration=1, overlap=0.5, preload=True, verbose=False)
                epochs_arr = np.array(epochs.get_data())

                # Drops bad intervals based on the Peak-to-Peak threshold declared as the 95th quantile
                ptp = np.ptp(epochs_arr, axis=2)
                reject_criteria = dict(eeg=np.quantile(ptp, 0.95))
                epochs_copy = epochs.copy().drop_bad(reject=reject_criteria, verbose=False)
                epochs = epochs_copy

                # Obtains the PSD and frequency values for each interval using the welch method
                scalar = mne.decoding.Scaler(raw.info)
                epochs_data = scalar.fit_transform(epochs.get_data())
                psd_mean, freqs_mean = mne.time_frequency.psd_array_welch(epochs_data, sfreq=raw.info['sfreq'], verbose=0,
                                                                          fmin=0.5, fmax=40, n_fft=1024, n_per_seg=1024)

                # Randomly shuffles the Data points
                psd_mean = np.array(psd_mean)
                np.random.shuffle(psd_mean)

                # Gets the experiment segment from the experimental order
                true_phase = i
                freq_resolution = freqs_mean[1] - freqs_mean[0]
                if phase_map[x[0]] == "Urban":
                    if true_phase == 0 or true_phase == 1:
                        true_phase += 2
                    else:
                        true_phase -= 2

                means = []
                variances = []
                maxs = []
                phases = []

                # Iterates over each frequency band and extracts the mean, variance, maximum and band power ratios
                for j in range(4):
                    freq_low, freq_high = band_freqs[j]

                    # Extracts the band power ratio
                    band_idxs = np.logical_and(freqs_mean >= freq_low, freqs_mean <= freq_high)
                    bps = scipy.integrate.simps(psd_mean[:, :, band_idxs], dx=freq_resolution)
                    bps /= scipy.integrate.simps(psd_mean[:, :], dx=freq_resolution)
                    phases.append(bps)

                    low_idx, _ = min(enumerate(freqs_mean), key=lambda x: abs(x[1] - freq_low))
                    high_idx, _ = min(enumerate(freqs_mean), key=lambda x: abs(x[1] - freq_high))
                    cut = np.array(psd_mean[:, :, low_idx:high_idx + 1])

                    # Extracts the mean, variances and maximum values
                    means.append(np.mean(cut, axis=2))
                    variances.append(np.var(cut, axis=2))
                    maxs.append(np.max(cut, axis=2))

                # Writes the data to the output file
                for j in range(len(psd_mean)):
                    row = np.array([])

                    for k in range(4):
                        row = np.append(row, phases[k][j])
                        row = np.append(row, means[k][j])
                        row = np.append(row, variances[k][j])
                        row = np.append(row, maxs[k][j])

                    # Appends the classification label and the participant number for cross-participant validation
                    row = np.append(row, true_phase)
                    row = np.append(row, x[0])

                    csv_four_class.writerow(row)