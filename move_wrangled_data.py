#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import glob
import pandas as pd

def move(source_path, destination_path):

    ''' 
    Moves true positive and true negative Excel file outputs from Pt 1 and Pt 2 of disco-data-processing.py to a central folder
    where the merging of positive and negative observations into one dataset will occur.
    '''

    # grab list of directories for all files of interest as defined by source_path previously
    preprocessed_data_directories = glob.glob(source_path)

    for starting_directory in preprocessed_data_directories: # loop through file paths containing desired outputs

        files_to_move = glob.glob("{}/*".format(starting_directory))
        
        for file in files_to_move:

            # check if file only has one row (i.e. no information) and ignore the file from the import if yes 
            check_df = pd.read_excel(file)
            
            if check_df.shape[0] > 1:
                # copy the excel files in the source filepath into the destination
                shutil.copy(file, destination_path)

    
    return print("Files for merging have been moved to the destination directory.")
