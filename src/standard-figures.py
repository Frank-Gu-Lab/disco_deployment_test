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

# for only the peaks with a significant disco effect
polymer_library = set(glob.glob("data/output/merged/stats_analysis_output_mean_*")) - set(glob.glob("data/output/merged/stats_analysis_output_mean_all_*"))

# uncomment to see the significant and zero peaks
# polymer_library = glob.glob("data/output/merged/stats_analysis_output_mean_all_*")

# Define a custom output directory for formal figures
output_directory = "data/output/publications"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for polymer in polymer_library:
    
    # grab polymer name from file path
    _, polymer_name = polymer.split(
        "data/output/merged/stats_analysis_output_mean_")
    polymer_name = polymer_name[:-5]

    # read polymer data
    polymer_df = pd.read_excel(polymer, index_col=[0, 1, 2, 3], header=[0, 1])
    
    # 1: Disco Effect Build-up Curves - one plot for one polymer, all peaks - from stats df mean
    
    # TO DO: improve the aesthetics of the buildup curves, match DISCO paper
    generate_buildup_curve(polymer_df, polymer_name, output_directory)


# 2: Binding Fingerprint Snapshot - one plot for all peaks - replicate data


    
    


