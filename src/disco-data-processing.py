# !/usr/bin/env python
# coding: utf-8

# Disco Data Processing Script
# Prior to running the script, please ensure you have inserted all the books you would like to be analyzed inside the input directory. The code will create custom output folders based on the name of the input books, so nothing overwrites. This code supports polymer data from the NMR analysis in "Book Format" (See PAA.xlsx for example) and "Batch Format" (See Batch 1, Batch 2 files for an example of what this looks like). 

# For Batch Format inputs, please ensure unique observations intended to be analyzed together follow the same naming format. For example, if there are 4 total CMC results, 3 from one day to be analyzed together, and 1 from a separate occasion, the sheet tabs should be named according to one format: CMC (1), CMC (2), CMC (3) {These 3 will be analyzed together. The 4th CMC tab should be named something different, such as CMC_other, and will be treated separately.

# Your Directory Should Look Like:    
# - src/disco-data-processing.py
# - src/data_wrangling_functions.py
# - src/data_merging.py
# - data/input/"raw_book_with_a_short_title_you_like.xlsx" (i.e. "PAA.xlsx")

# Then simply run this .py script. 
# Part 1 : Reading and Cleaning Data  - prepare the data for statistical analysis
# Part 2 : Statistical Analysis - classify true positive binding proton observations, generate AFo plots
# Part 3 : Merge true positive and true negative observations into clean dataset for future machine learning

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

# import all data wrangling functions for Pt. 1 & 2, and data merge function for Pt. 3
from data_wrangling_functions import *
from data_merging import merge
from data_visualize import *

# ESTABLISH LOCAL DIRECTORY PATHS ---------------------

#assign the local path to the raw Excel books 
raw_book_path = os.path.abspath('input')
print('Searching in directory:', raw_book_path, '\n')

#list all raw books in file
list_of_raw_books = glob.glob("../data/input/*.xlsx")
print('List of raw books to be analyzed are: ', list_of_raw_books, '\n')

# makes a global ouput directory if there isn't already one
global_output_directory = "../data/output"

if not os.path.exists(global_output_directory):
    os.makedirs(global_output_directory)

# PERFORM DATA WRANGLING - CONVERT ALL EXCEL BOOKS IN INPUT FOLDER TO DATAFRAMES ---------------------

# initialize global lists to hold all tuples generated, one tuple per polymer input will be generated (current_book_title, clean_df)
clean_book_tuple_list = []
batch_tuple_list = []

clean_batch_tuple_list = []
clean_batch_list = []

# Convert all Excel books in the input folder into tuple key-value pairs that can be indexed
for book in list_of_raw_books:
    
    # indicates the book should be handled via batch processing data cleaning function
    if "Batch" in book:
        print("This should be cleaned via batch processing! Entering batch processing function.")
        
        #append tuples from the list output of the batch processing function, so that each unique polymer tuple is assigned to the clean_batch_tuple_list
        batch_tuple_list.append([polymer for polymer in batch_to_dataframe(book)]) 
        
    # indicates book is ok to be handled via the individual data cleaning function before appending to the clean data list    
    else: 
        print("Book contains individual polymer, entering individual processing function.")
        clean_book_tuple_list.append(book_to_dataframe(book, global_output_directory))

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
        
        # remove any spaces from title
        current_df_title = current_df_title.replace(' ', '')

        print("Beginning data analysis for {}...".format(current_df_title))

        # DEFINE CUSTOM OUTPUT DIRECTORIES FOR THIS DATAFRAME ------------------------------------------

        output_directory_exploratory, output_directory_curve, output_directory_tables, output_directory = generate_directories(current_df_title, global_output_directory)
                
        # CALCULATE ATTENUATION & CORR ATTENUATION -----------------------------------

        df_true, df_false = add_attenuation(current_df)
        current_df_attenuation = add_corr_attenuation(df_true, df_false)
        
        # PERFORM EXPLORATORY DATA VISUALIZATION -----------------------------------

        print("Visualizing data for {} and saving to a custom exploratory plots output folder...".format(current_df_title))
        generate_concentration_plot(current_df_attenuation, output_directory_exploratory, current_df_title)
        generate_ppm_plot(current_df_attenuation, output_directory_exploratory, current_df_title)

        # This completes Part 1 - Data Preprocessing and Visualizing the Results!

        # BEGINNING PART 2 -------- Modelling the Data ---------------------------------------
        
        current_df_mean, current_df_replicates = modelling_data(current_df_attenuation, current_df_title, output_directory, output_directory_curve, output_directory_tables)
        
        print("All activities are now completed for: {}".format(current_df_title))

    print("Hooray! All polymers in the input files have been processed.")

# LOOP THROUGH AND PROCESS EVERY CLEAN DATAFRAME IN THE BATCH LIST GENERATED ABOVE, IF ANY ----------------------------------

if len(clean_batch_tuple_list) != 0: 
    for i in range(len(clean_batch_tuple_list)):

        # BEGINNING PART 1 -------- Reading in Data and Visualizing the Results ------------------------ 

        #define current dataframe to be analyzed and its title from the tuple output of the data cleaning code
        current_df = clean_batch_tuple_list[i][1]
        current_df_title = clean_batch_tuple_list[i][0]

        # remove any spaces from title
        current_df_title = current_df_title.replace(' ', '')

        print("Beginning data analysis for {}...".format(current_df_title))

        # DEFINE CUSTOM OUTPUT DIRECTORIES FOR THIS DATAFRAME ------------------------------------------

        output_directory_exploratory, output_directory_curve, output_directory_tables, output_directory = generate_directories(current_df_title, global_output_directory) 
        
        # CALCULATE ATTENUATION & CORR ATTENUATION -----------------------------------
        
        df_true, df_false = add_attenuation(current_df, 'batch')
        current_df_attenuation = add_corr_attenuation(df_true, df_false, 'batch')

        # PERFORM EXPLORATORY DATA VISUALIZATION -----------------------------------

        print("Visualizing data for {} and saving to a custom exploratory plots output folder...".format(current_df_title))
        generate_concentration_plot(current_df_attenuation, output_directory_exploratory, current_df_title)
        generate_ppm_plot(current_df_attenuation, output_directory_exploratory, current_df_title)

        # This completes Part 1 - Data Preprocessing and Visualizing the Results!

        # BEGINNING PART 2 -------- Modelling the Data ---------------------------------------

        current_df_mean, current_df_replicates = modelling_data(current_df_attenuation, current_df_title, output_directory, output_directory_curve, output_directory_tables, 'batch')
        
        print("All activities are now completed for: {}".format(current_df_title))

    print("Hooray! All polymers in the input files have been processed.")

# PART 3 - MERGE TRUE POSITIVE AND TRUE NEGATIVE BINDING OBSERVATIONS ---------------------

# makes a global ouput directory for merged data if not existing
merge_output_directory = "{}/merged".format(global_output_directory)

if not os.path.exists(merge_output_directory):
    os.makedirs(merge_output_directory)

# define data source path and data destination path to pass to data merging function
source_path = '{}/*/data_tables_from_*'.format(global_output_directory)
destination_path = '{}'.format(merge_output_directory)

# call data merging function and write complete dataset to file
merged_dataset = merge(source_path, destination_path)
output_file_name = "merged_binding_dataset.xlsx"
merged_dataset.to_excel(os.path.join(merge_output_directory, output_file_name))
print("Data processing completed! Merged_binding_dataset.xlsx file available in the output directory under merged.")