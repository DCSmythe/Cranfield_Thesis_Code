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
import matplotlib.pyplot as plt
import scipy.integrate
import csv
import sys

"""
This script generates topomaps for the mean and variance of each frequency band and each experimental segment

Inputs:
    - input_directory_path:  
    The path to the directory containing the .fif files, representing the segmented and processed data intervals after 
    ICA

    - output_directory_path:                     
    The directory path in which the topomap figures will be stored

    - questionnaire_csv_path:
    The path to the questionnaire results, to extract the order in which the experiments were performed in
"""

if __name__ == "__main__":
    # Reads the command line arguments
    if len(sys.argv) != 4:
        exit("Incorrect number of arguments. Arguments should be of the form: <input_directory_path> <output_directory_path> <questionnaire_csv_path> \nExiting Program")
    input_file_path = str(sys.argv[1])
    output_path = str(sys.argv[2])
    questionnaire_csv_path = str(sys.argv[3])

    # Declares the frequency band regions
    theta_band = 4, 8
    alpha_band = 8, 12
    beta_band = 12, 30
    gamma_band = 30, 40
    band_freqs = [theta_band, alpha_band, beta_band, gamma_band]

    # Reads in the experiment order from the questionnaire file
    phase_map = {}
    with open(questionnaire_csv_path, newline='') as time_file:
        reader = csv.reader(time_file, delimiter=',')
        for row in reader:
            phase_map[row[0]] = row[12]

    means_arr = [[] for _ in range(4)]
    vars_arr = [[] for _ in range(4)]

    # Iterates through and determines the mean and variance values for each participant
    for path in os.listdir("data"):
        x = path.split('.')

        # Iterates over the 4 data segments
        for i in range(0, 4):
            # Loads in the data
            f_path = input_file_path + "/" + x[0] + "/" + str(i + 2) + "_raw.fif"
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

            # Iterates over each frequency band and extracts the median of the mean and variance values for each band
            for j in range(4):
                freq_low, freq_high = band_freqs[j]

                low_idx, _ = min(enumerate(freqs_mean), key=lambda x: abs(x[1] - freq_low))
                high_idx, _ = min(enumerate(freqs_mean), key=lambda x: abs(x[1] - freq_high))

                cut = np.array(psd_mean[:, :, low_idx:high_idx + 1])
                means.append(np.mean(cut, axis=2))
                variances.append(np.var(cut, axis=2))
            vars_arr[true_phase].append(np.median(variances, axis=1))
            means_arr[true_phase].append(np.median(means, axis=1))

    # Assumes the "ICA_out/90390/2_raw.fif" is in the same directory and uses it to set the channel locations
    raw = mne.io.read_raw_fif("ICA_out/90390/2_raw.fif")
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)

    route_names = ["Motorway No Cars", "Motorway Cars", "Urban No Cars", "Urban Cars"]
    band_names = ["Theta", "Alpha", "Beta", "Gamma"]

    # Creates the output directory if it did not already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterates over each band and draws a topomap for the variances of each experiment segment
    for i in range(4):
        fig, ax = plt.subplots(1, 4)
        for j in range(4):
            im, cm = mne.viz.plot_topomap(np.mean(vars_arr[j], axis=0)[i], raw.info, axes=ax[j], show=False, sensors=True,
                                          vmin=np.min(np.mean(vars_arr[j], axis=0)[i]),
                                          vmax=np.max(np.mean(vars_arr[j], axis=0)[i]), cmap="RdBu_r")
            ax[j].set_title(route_names[j])
        ax_x_start = 0.92
        ax_x_width = 0.02
        ax_y_start = 0.3
        ax_y_height = 0.4
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        fig.set_size_inches(20, 15)

        # Saves the figure and displays it
        plt.savefig(output_path + "/" + band_names[i] + "_variances.png", bbox_inches='tight')
        plt.show()

    # Iterates over each band and draws a topomap for the mean of each experiment segment
    for i in range(4):
        fig, ax = plt.subplots(1, 4)
        for j in range(4):
            im, cm = mne.viz.plot_topomap(np.mean(means_arr[j], axis=0)[i], raw.info, axes=ax[j], show=False, sensors=True,
                                          vmin=np.min(np.mean(means_arr[j], axis=0)[i]),
                                          vmax=np.max(np.mean(means_arr[j], axis=0)[i]), cmap="RdBu_r")
            ax[j].set_title(route_names[j])
        ax_x_start = 0.92
        ax_x_width = 0.02
        ax_y_start = 0.3
        ax_y_height = 0.4
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        fig.set_size_inches(20, 15)

        # Saves the figure and displays it
        plt.savefig(output_path + "/" + band_names[i] + "_means.png", bbox_inches='tight')
        plt.show()
