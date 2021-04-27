#!/usr/bin/env python
# coding: utf-8

# Disco Data Processing Script
# Prior to running the script, please ensure you have inserted all the books you would like to be analyzed inside the input directory. The code will create custom output folders based on the name of the input books, so nothing overwrites. This code supports polymer data from the NMR analysis in "Book Format" (See PAA.xlsx for example) and "Batch Format" (See Batch 1, Batch 2 files for an example of what this looks like). 
# 
# For Batch Format inputs, please ensure unique observations intended to be analyzed together follow the same naming format. For example, if there are 4 total CMC results, 3 from one day to be analyzed together, and 1 from a separate occasion, the sheet tabs should be named according to one format: CMC (1), CMC (2), CMC (3) {These 3 will be analyzed together. The 4th CMC tab should be named something different, such as CMC_other, and will be treated separately.
#     
# Your Input Folder Should Look Like:    
# - disco-data-processing.py
# - data_wrangling_functions.py
# - input/"raw_book_with_a_short_title_you_like.xlsx" (i.e. "PAA.xlsx")
# 
# Then simply run on this .py script. 
# Part 1 : Reading and Cleaning Data  - prepare the data for statistical analysis
# Part 2 : Statistical Analysis - classify true positive binding proton observations, generate AFo plots
# (TO DO) Part 3 : Merge true positive and true Negative Observations into clean dataset for future machine learning

# importing required libraries:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import scipy.stats as stats
from scipy.optimize import curve_fit
import os 
import glob

# define handy shortcut for indexing a multi-index dataframe
idx = pd.IndexSlice

# import all data wrangling functions
from data_wrangling_functions import *

# ESTABLISH LOCAL DIRECTORY PATHS ---------------------

#assign the local path to the raw Excel books 
raw_book_path = os.path.abspath('input')
print('Searching in directory:', raw_book_path, '\n')

#list all raw books in file
list_of_raw_books = glob.glob("input/*.xlsx")
print('List of raw books to be analyzed are: ', list_of_raw_books, '\n')

# PERFORM DATA WRANGLING - CONVERT ALL EXCEL BOOKS IN INPUT FOLDER TO DATAFRAMES ---------------------

# initialize global lists to hold all tuples generated, one tuple per polymer input will be generated (current_book_title, clean_df)
clean_book_tuple_list = []
batch_tuple_list = []

# Convert all Excel books in the input folder into tuple key-value pairs that can be indexed
for book in list_of_raw_books:
    
    # indicates the book should be handled via batch processing data cleaning function
    if "Batch" in book:
        print("This should be cleaned via batch processing! Entering batch processing function.")
        
        #append tuples from the list output of the batch processing function, so that each unique polymer tuple is assigned to the clean_batch_tuple_list
        batch_tuple_list.append([polymer for polymer in convert_excel_batch_to_dataframe(book)]) 
        
    # indicates book is ok to be handled via the individual data cleaning function before appending to the clean data list    
    else: 
        print("Book contains individual polymer, entering individual processing function.")
        clean_book_tuple_list.append(convert_excel_to_dataframe(book))

# PERFORM DATA CLEANING ON ALL BOOKS PROCESSED VIA BATCH PROCESSING ----------------

#if there has been batch data processing, call the batch cleaning function
if len(batch_tuple_list) != 0: 
    clean_batch_list = clean_the_batch_tuple_list(batch_tuple_list)

# convert clean batch list to a clean batch tuple list format (polymer_name, df) for further processing
clean_batch_tuple_list = [(clean_batch_list[i]['polymer_name'].iloc[0], clean_batch_list[i]) for i in range(len(clean_batch_list))]

# LOOP THROUGH AND PROCESS EVERY CLEAN DATAFRAME IN THE POLYMER BOOK LIST GENERATED ABOVE, IF ANY ----------------------------------
# custom processing functions default to the "book" path, so no additional parameters passed here

if len(clean_book_tuple_list) != 0: 
    for i in range(len(clean_book_tuple_list)):

        # BEGINNING PART 1 -------- Reading in Data and Visualizing the Results ------------------------ 

        #define current dataframe to be analyzed and its title from the tuple output of the data cleaning code
        current_df = clean_book_tuple_list[i][1]
        current_df_title = clean_book_tuple_list[i][0]

        print("Beginning data analysis for {}...".format(current_df_title))

        # DEFINE GLOBAL CUSTOM OUTPUT DIRECTORIES FOR THIS DATAFRAME ------------------------------------------

        # Define a global custom output directory for the current df in the list
        output_directory = "output_from_{}".format(current_df_title)

        # make global directory if there isn't already one for overall output for current df
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Define a global output exploratory directory for the exploratory plots 
        output_directory_exploratory = "{}/exploratory_plots_from_{}".format(output_directory, current_df_title)

        # make directory if there isn't already one for exploratory output 
        if not os.path.exists(output_directory_exploratory):
            os.makedirs(output_directory_exploratory)

        # Define a global output directory for the curve fit plots for the current df title
        output_directory2 = "{}/curve_fit_plots_from_{}".format(output_directory, current_df_title)

        # make this directory if there isn't already one for the curve_fit_plots
        if not os.path.exists(output_directory2):
            os.makedirs(output_directory2)  

        # Define a global output directory for the final data tables after curve fitting and stats
        output_directory3 = "{}/data_tables_from_{}".format(output_directory, current_df_title)

        # make this directory if there isn't already one for the data tables
        if not os.path.exists(output_directory3):
            os.makedirs(output_directory3)    

        # CALCULATE ATTENUATION & CORR ATTENUATION -----------------------------------

        current_df_attenuation = add_attenuation_and_corr_attenuation_to_dataframe(current_df)

        # PERFORM EXPLORATORY DATA VISUALIZATION -----------------------------------

        print("Visualizing data for {} and saving to a custom exploratory plots output folder...".format(current_df_title))
        generate_concentration_plot(current_df_attenuation, output_directory_exploratory, current_df_title)
        generate_ppm_plot(current_df_attenuation, output_directory_exploratory, current_df_title)

        # This completes Part 1 - Data Preprocessing and Visualizing the Results!

        # BEGINNING PART 2 -------- Modelling the Data ---------------------------------------

        # STEP 1 OF 5 - Prepare and generate mean dataframe of current data for stats with degrees of freedom and sample size included -----
        current_df_mean = prep_mean_data_for_stats(current_df_attenuation)
        current_df_replicates = prep_replicate_data_for_stats(current_df_attenuation)

        # STEP 2 OF 5 - Perform t test for statistical significance -------------------------
        current_df_mean = get_t_test_results(current_df_mean, p=0.95)

        # STEP 3 OF 5 - Compute amplification factor -----------------------------------------
        current_df_mean, current_df_replicates = compute_amplification_factor(current_df_mean, current_df_replicates, af_denominator = 10)
        
        # export current dataframes to excel with no observations dropped, for future reference in ML -----
        output_file_name = "stats_analysis_output_replicate_all_{}.xlsx".format(current_df_title) # replicates
        current_df_replicates.to_excel(os.path.join(output_directory3, output_file_name)) 
        output_file_name = "stats_analysis_output_mean_all_{}.xlsx".format(current_df_title) # mean
        current_df_mean.to_excel(os.path.join(output_directory3, output_file_name))
        
        # STEP 4 OF 5 - Drop proton peaks from further analysis that fail our acceptance criteria -----------------------------------------
        current_df_mean, current_df_replicates = drop_bad_peaks(current_df_mean, current_df_replicates, current_df_title, output_directory)

        # STEP 5 OF 5 - Perform curve fitting, generate plots, and export results to file  -----------------------------------------
        current_df_mean, current_df_replicates = execute_curvefit(
            current_df_mean, current_df_replicates, output_directory2, output_directory3, current_df_title)
        print("All activities are now completed for: {}".format(current_df_title))

    print("Hooray! All polymers in the input files have been processed.")

# LOOP THROUGH AND PROCESS EVERY CLEAN DATAFRAME IN THE BATCH LIST GENERATED ABOVE, IF ANY ----------------------------------

if len(clean_batch_tuple_list) != 0: 
    for i in range(len(clean_batch_tuple_list)):

        # BEGINNING PART 1 -------- Reading in Data and Visualizing the Results ------------------------ 

        #define current dataframe to be analyzed and its title from the tuple output of the data cleaning code
        current_df = clean_batch_tuple_list[i][1]
        current_df_title = clean_batch_tuple_list[i][0]

        print("Beginning data analysis for {}...".format(current_df_title))

        # DEFINE GLOBAL CUSTOM OUTPUT DIRECTORIES FOR THIS DATAFRAME ------------------------------------------

        # Define a global custom output directory for the current df in the list
        output_directory = "output_from_{}".format(current_df_title)

        # make global directory if there isn't already one for overall output for current df
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Define a global output exploratory directory for the exploratory plots 
        output_directory_exploratory = "{}/exploratory_plots_from_{}".format(output_directory, current_df_title)

        # make directory if there isn't already one for exploratory output 
        if not os.path.exists(output_directory_exploratory):
            os.makedirs(output_directory_exploratory)

        # Define a global output directory for the curve fit plots for the current df title
        output_directory2 = "{}/curve_fit_plots_from_{}".format(output_directory, current_df_title)

        # make this directory if there isn't already one for the curve_fit_plots
        if not os.path.exists(output_directory2):
            os.makedirs(output_directory2)  

        # Define a global output directory for the final data tables after curve fitting and stats
        output_directory3 = "{}/data_tables_from_{}".format(output_directory, current_df_title)

        # make this directory if there isn't already one for the data tables
        if not os.path.exists(output_directory3):
            os.makedirs(output_directory3)    

        # CALCULATE ATTENUATION & CORR ATTENUATION -----------------------------------

        current_df_attenuation = add_attenuation_and_corr_attenuation_to_dataframe(current_df, 'batch')

        # PERFORM EXPLORATORY DATA VISUALIZATION -----------------------------------

        print("Visualizing data for {} and saving to a custom exploratory plots output folder...".format(current_df_title))
        generate_concentration_plot(current_df_attenuation, output_directory_exploratory, current_df_title)
        generate_ppm_plot(current_df_attenuation, output_directory_exploratory, current_df_title)

        # This completes Part 1 - Data Preprocessing and Visualizing the Results!

        # BEGINNING PART 2 -------- Modelling the Data ---------------------------------------

        # STEP 1 OF 5 - Prepare and generate mean dataframe of current data for stats with degrees of freedom and sample size included -----
        current_df_mean = prep_mean_data_for_stats(current_df_attenuation, 'batch')
        current_df_replicates = prep_replicate_data_for_stats(current_df_attenuation, 'batch')

        # STEP 2 OF 5 - Perform t test for statistical significance -------------------------
        current_df_mean = get_t_test_results(current_df_mean, p=0.95)

        # STEP 3 OF 5 - Compute amplification factor -----------------------------------------
        # note: if AF denominators are different for each polymer, should make a list of all values for all polymers, then pass list[i] to af_denominator here
        current_df_mean, current_df_replicates = compute_amplification_factor(current_df_mean, current_df_replicates, af_denominator = 10)
        
        # export current dataframes to excel with no observations dropped, for future reference in ML -----
        output_file_name = "stats_analysis_output_replicate_all_{}.xlsx".format(current_df_title) # replicates
        current_df_replicates.to_excel(os.path.join(output_directory3, output_file_name)) 
        output_file_name = "stats_analysis_output_mean_all_{}.xlsx".format(current_df_title) # mean
        current_df_mean.to_excel(os.path.join(output_directory3, output_file_name))
        
        # STEP 4 OF 5 - Drop proton peaks from further analysis that fail our acceptance criteria -----------------------------------------
        current_df_mean, current_df_replicates = drop_bad_peaks(current_df_mean, current_df_replicates, current_df_title, output_directory, batch_or_book='batch')

        # STEP 5 OF 5 - Perform curve fitting, generate plots, and export results to file  -----------------------------------------
        current_df_mean, current_df_replicates = execute_curvefit(
            current_df_mean, current_df_replicates, output_directory2, output_directory3, current_df_title, batch_or_book='batch')
        print("All activities are now completed for: {}".format(current_df_title))

    print("Hooray! All polymers in the input files have been processed.")
