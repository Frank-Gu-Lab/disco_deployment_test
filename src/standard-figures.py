# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 21 14:39 2021

@author: Samantha Stuart

Manipulates source data into formats for plotting.
Generates standard publication plots from processed DISCO data.

"""
import pandas as pd
import glob
import os
from sklearn.preprocessing import MaxAbsScaler
from discoprocess.data_plot import generate_buildup_curve
from discoprocess.data_plot import generate_fingerprint
from discoprocess.data_plot_helpers import tukey_fence


# for only the peaks with a significant disco effect
polymer_library_binding = set(glob.glob("../data/output/merged/stats_analysis_output_mean_*")) - set(glob.glob("../data/output/merged/stats_analysis_output_mean_all_*"))

# significant and zero peaks
polymer_library_all = glob.glob("../data/output/merged/stats_analysis_output_mean_all_*")

polymer_library_replicates = glob.glob("../data/output/merged/stats_analysis_output_replicate_*")

merged_bind_dataset = pd.read_excel("../data/output/merged/merged_binding_dataset.xlsx")

# Define a custom output directory for formal figures
output_directory = "../data/output/publications"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def grab_polymer_name(full_filepath, common_filepath):
    '''Grabs the polymer name from file path.

    Parameters:
    -----------
    full_filepath: string
        path to the datasheet for that polymer

    common_filepath: string
        portion of the filepath that is shared between all polymer inputs, excluding their custom names

    Returns:
    -------
    polymer_name: string
        the custom portion of the filepath with the polymer name and any other custom info
    '''

    #Necessary for some windows operating systems
    for char in full_filepath:
        if char == "\\":
            full_filepath = full_filepath.replace("\\", "/")

    for char in common_filepath:
        if char == "\\":
            common_filepath = full_filepath.replace("\\", "/")

    polymer_name = full_filepath.split(common_filepath)[1]
    polymer_name = polymer_name[:-5] # remove the .xlsx

    return polymer_name

# plot DISCO Effect build up curves with only significant peaks
for polymer in polymer_library_binding:

    binding_directory = f"{output_directory}/binding"

    if not os.path.exists(binding_directory):
        os.makedirs(binding_directory)

    polymer_name = grab_polymer_name(full_filepath = polymer,
    common_filepath="../data/output/merged/stats_analysis_output_mean_")

    # read polymer datasheet
    polymer_df = pd.read_excel(polymer, index_col=[0, 1, 2, 3], header=[0, 1])

    generate_buildup_curve(polymer_df, polymer_name, binding_directory)


# plot DISCO Effect build up curves with insignificant and significant peaks overlaid
for polymer in polymer_library_all:
    binding_directory2 = f"{output_directory}/all_peaks"

    if not os.path.exists(binding_directory2):
        os.makedirs(binding_directory2)

    polymer_name = grab_polymer_name(full_filepath=polymer,
                                     common_filepath="../data/output/merged/stats_analysis_output_mean_all_")

    # read polymer datasheet
    polymer_df = pd.read_excel(polymer, index_col=[0, 1, 2, 3], header=[0, 1])

    generate_buildup_curve(polymer_df, polymer_name, binding_directory2)


# plot binding fingerprints
binding_directory = f"{output_directory}/binding"

unique_bind_polymers = merged_bind_dataset.loc[merged_bind_dataset['AFo'] != 0, ('polymer_name')].unique()

# iterate through figures per polymer
for polymer in unique_bind_polymers:
    plot_df = merged_bind_dataset.loc[(merged_bind_dataset['polymer_name'] == polymer) & (merged_bind_dataset['AFo'] != 0)].copy()

    # identify univariate outliers in plot_df using Tukey-Fence method
    plot_df_outliers = tukey_fence(plot_df, 'AFo')
    # print(plot_df_outliers)

    # subset df to remove probable outliers
    plot_df_no_outliers = plot_df_outliers[plot_df_outliers['outlier_prob'] == False]

    # fit normalizer to AFo in the plot df after outliers removed
    scaler = MaxAbsScaler()
    plot_df_outliers['AFo_norm'] = scaler.fit(abs(plot_df_no_outliers['AFo'].values).reshape(-1,1))

    # transform all data (including outliers) and mark flagged outliers on plot, so outliers not used as normalization ref point
    plot_df_outliers['AFo_norm'] = scaler.transform(abs(plot_df_outliers['AFo'].values).reshape(-1,1))
    plot_df_outliers['SSE_bar_norm'] = scaler.transform(abs(plot_df_outliers['SSE_bar'].values).reshape(-1, 1))
    plot_df_outliers['AFo_bar_norm'] = scaler.transform(abs(plot_df_outliers['AFo_bar'].values).reshape(-1, 1))
    plot_df_outliers['SSE_norm'] = scaler.transform(abs(plot_df_outliers['SSE'].values).reshape(-1, 1))

    generate_fingerprint(plot_df_outliers, polymer, binding_directory)
