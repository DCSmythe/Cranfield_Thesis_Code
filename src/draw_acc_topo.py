"""
These source files utilise functions from the 'MNE', 'scikit-learn' and 'Tensorflow' external python libraries.
As such, links are placed here to give credit to the original distributors of these libraries:
MNE:            https://mne.tools/stable/index.html#
Tensorflow:     https://www.tensorflow.org/
scikit-learn:   https://scikit-learn.org/stable/index.html#
"""

import mne
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

"""
This script draws a topomap displaying the accuracy values of each EEG channel.

Inputs:
    - input_file_path:  
    The path to the '.csv' file containing the accuracy values for each of the accuracy values for the 'Urban', 
    'Motorway' and 'High_Low' models
    
    - output_directory_path:                     
    The path of the desired output directory, in which the topomaps are saved.
"""
if __name__ == "__main__":
    # Reads the command line arguments
    if len(sys.argv) != 3:
        exit(
            "Incorrect number of arguments. Arguments should be of the form: <input_file_path:> <output_directory_path> \nExiting Program")
    input_file_path = str(sys.argv[1])
    output_path = str(sys.argv[2])

    # Declares the frequency band regions
    theta_band = 4, 8
    alpha_band = 8, 12
    beta_band = 12, 30
    gamma_band = 30, 40
    band_freqs = [theta_band, alpha_band, beta_band, gamma_band]

    # Reads in the accuracy values
    maps = []
    with open(input_file_path, 'r', newline='') as time_file:
        reader = csv.reader(time_file, delimiter=',')
        for row in reader:
            maps.append(row)

    # Assumes the "ICA_out/90390/2_raw.fif" is in the same directory and uses it to set the channel locations
    raw = mne.io.read_raw_fif("ICA_out/90390/2_raw.fif")
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)

    # Creates the output directory if it did not already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Declares the axes and writes the input values to it
    ax_names = ["Urban", "Motorway", "Combined"]
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        im, cm = mne.viz.plot_topomap(np.float32(maps[i]), raw.info, show=False, sensors=True, axes=ax[i], cmap="RdBu_r", vmin=50,
                                      vmax=70)
        ax[i].set_title(ax_names[i])
    ax_x_start = 0.92
    ax_x_width = 0.02
    ax_y_start = 0.3
    ax_y_height = 0.4
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    fig.set_size_inches(20, 15)

    # Saves the figure and displays it
    plt.savefig(output_path + "/Accuracy_map.png")
    plt.show()
