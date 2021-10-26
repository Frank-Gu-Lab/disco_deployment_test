# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 21 14:39 2021

@author: Samantha Stuart

Standard publication plots generated from processed DISCO data. 

TO DO:
------
- write plotting wrapper script to read data outputs from data processing code, and 
auto generate DISCO paper figures for all experiments

    - DISCO effect build-up curves w/Std error, overlaid for a given binding site
        - modify v2 from the mean plotting graph - bring out here (mean data)

    - DISCO polymer fingerprints (need replicate AF0 data + AF0_bar)
        - modify Jeffs code, read data from replicate tables for a given polymer
        - automatically indicate outliers ?

"""
import pandas as pd
import glob
import os

from discoprocess.data_plot import generate_buildup_curve
from discoprocess.data_plot import generate_fingerprint

# for only the peaks with a significant disco effect
polymer_library_binding = set(glob.glob("data/output/merged/stats_analysis_output_mean_*")) - set(glob.glob("data/output/merged/stats_analysis_output_mean_all_*"))

# significant and zero peaks
polymer_library_all = glob.glob("data/output/merged/stats_analysis_output_mean_all_*")

polymer_library_replicates = glob.glob(
    "data/output/merged/stats_analysis_output_replicate_*")

# Define a custom output directory for formal figures
output_directory = "data/output/publications"

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

    _, polymer_name = full_filepath.split(common_filepath)
    polymer_name = polymer_name[:-5] # remove the .xlsx

    return polymer_name

# plot DISCO Effect build up curves with only significant peaks
for polymer in polymer_library_binding:
    
    binding_directory = f"{output_directory}/binding"

    if not os.path.exists(binding_directory):
        os.makedirs(binding_directory)

    polymer_name = grab_polymer_name(full_filepath = polymer,
        common_filepath="data/output/merged/stats_analysis_output_mean_")

    # read polymer datasheet
    polymer_df = pd.read_excel(polymer, index_col=[0, 1, 2, 3], header=[0, 1])
    
    generate_buildup_curve(polymer_df, polymer_name, binding_directory)

    # Plot DISCO AFo Fingerprints for the binding peaks in each polymer
    polymer_replicate = f"data/output/merged/stats_analysis_output_replicate_{polymer_name}.xlsx"

    polymer_replicate_df = pd.read_excel(polymer_replicate, index_col=[0], header=[0])

    generate_fingerprint(polymer_df, polymer_replicate_df, polymer_name, binding_directory)



# plot DISCO Effect build up curves with insignificant and significant peaks overlaid
for polymer in polymer_library_all:

    binding_directory2 = f"{output_directory}/all_peaks"

    if not os.path.exists(binding_directory2):
        os.makedirs(binding_directory2)

    polymer_name = grab_polymer_name(full_filepath=polymer,
                                     common_filepath="data/output/merged/stats_analysis_output_mean_all_")

    # read polymer datasheet
    polymer_df = pd.read_excel(polymer, index_col=[0, 1, 2, 3], header=[0, 1])

    generate_buildup_curve(polymer_df, polymer_name, binding_directory2)


    

    
    


