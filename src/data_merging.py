# Part 3 Preprocessing Function - merges positive and negative data 

# import packages
import pandas as pd
import numpy as np
import os
import glob
import re
import shutil

def move(source_path, destination_path):

    ''' Moves true positive and true negative Excel file outputs from Pt 1 and Pt 2 of disco-data-processing.py to a central folder
    where the merging of positive and negative observations into one dataset will occur.
    
    Parameters
    ----------
    source_path : str
        String containing path of source directory, including the Unix wildcard * to indicate to the function to retrieve all files therein.
    
    destination_path : str
        String containing path of destination directory.
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

def clean(df_list, polymer_list, pos : bool):
    """This function cleans the dataframes in the inputted list by dropping extra columns, appending the missing polymer_name column, 
    and drops the appropriate columns to the index level.
    
    Parameters
    ----------
    df_list : list
        List of Pandas DataFrames to be cleaned.
    
    polymer_list : list
        List of polymer names contained in the passed DataFrames.
    
    pos : bool
        True for positive binding observations, False for negative binding observations.
    """
    # 1) clean list
    for i in range(len(df_list)):

        if 'polymer_name' not in df_list[i].columns:
            df_list[i].insert(loc = 0, column = 'polymer_name', value = polymer_list[i]) # add polymer name column
        
        # drop extra level
        df_list[i] = df_list[i].droplevel(1, axis = 1)
        
        if pos:
            # if this column exists as a column (it should be in the index), also drop
            if "ppm" in df_list[i].columns:
                df_list[i] = df_list[i].drop("ppm", axis = 1)
                
            # drop extra columns
            drop_data = df_list[i].loc[:,
                                    ['corr_%_attenuation','dofs',
                                    'amp_factor', 'yikj_bar','SSE_bar']]
        else:
            # drop other columns not needed and extra level
            drop_data = df_list[i].loc[:, ['corr_%_attenuation', 'dofs', 'amp_factor']]
        
        df_list[i] = df_list[i].drop(drop_data.columns, axis = 1)
        
        # changing column names to [None] for consistency
        df_list[i].columns.names = [None]

def reformat(df_list, pos : bool):
    """This function takes in a list of Pandas DataFrames, concatenates them, and returns the final reformatted DataFrame. 
    
    Parameters
    ----------
    df_list : list
        List of Pandas DataFrames to be concatenated and reformatted.
    
    pos : bool
        True for positive binding observations, False for negative binding observations.
        
    Returns
    -------
    df : Pandas.DataFrame
        Final reformatted DataFrame.
    """
    # concatenate
    df = pd.concat(df_list)

    if pos:
        
        # 1) drop sat time from index
        df.index = df.index.droplevel(1)

        # 1) reset and reformat index
        df = df.reset_index()
        df = df.rename(columns = {'level_2': "ppm"})
        df = df.rename_axis(columns = "index")

        # 1) round ppm values to 4 sig figs
        df['ppm'] = np.round(df.ppm.values, 4)
        df.loc[df['significance'] == False, ('AFo_bar')] = 0.0 # removes false associations, as these values weren't considered in curve fitting
        
    else:
    
        # 2) drop extra ppm column that (sometimes) gets created from combining multi indices
        
        df = df.loc[:, :"significance"]

        # 2) drop sat time from index
        df.index = df.index.droplevel(1)

        # 2) reset and reformat index
        
        df = df.reset_index()
        df = df.rename_axis(columns = "index")

        # 2) remove observations with sample size of 1 as that is not enough to know significance
        small_samples_mask = df[df['sample_size'] == 1]
        df = df.drop(index = small_samples_mask.index)

        # 2) if t test results are NaN, also remove, as don't know true data characteristics
        t_nan_mask = df[np.isnan(df['t_results'].values) == True]
        df = df.drop(index = t_nan_mask.index)

        # 1 & 2) assign AFo to zero in both dataframes if significance is FALSE
        df.loc[df['significance'] == False, ('AFo_bar')] = 0.0 # assigns true negatives

    return df

def join(df1, df2):
    """This function takes in two cleaned DataFrames and merges them in a way that makes sense.
    
    Parameters
    ----------
    df1, df2 : Pandas.DataFrame
        Cleaned dataframes containing positive and negative binding observations, respectively.
    
    Returns
    -------
    df : Pandas.DataFrame
        Final merged dataframe containing the merged dataset and positive and negative observations.
    """
    # 3) now need to JOIN data from the two tables in a way that makes sense

    key = ['concentration','proton_peak_index','ppm','polymer_name','sample_size','t_results','significance'] # combo of values that define a unique observation in each table

    # 3) first - need to join ALL the data from the two tables
    # 3) Outer Join A and B on the Key for unique observations, and include which table they were merged from for filtering

    df = pd.merge(left = df1, right = df2, how = 'outer', on = key, indicator = True, suffixes = ("_a","_b"))

    # 3) allocate the correct AFo_bar values from each table 
    df.loc[np.isnan(df['AFo_bar_a']) == False, ('AFo_bar')] = df['AFo_bar_a'] # assigns AFo_bar_a values to observations where they exist
    df.loc[np.isnan(df['AFo_bar_a']) != False, ('AFo_bar')] = df['AFo_bar_b'] # assigns AFo_bar_b where no AFo_bar_a value does not exist (i.e. polymer was all non)

    # 3) remaining NaNs in table_c have significance = True, 
    # BUT these were not used in Curve Fitting AFo calculations, due to strict drop criterion in preprocessing. 
    # Therefore should still drop them here, as we don't know their true AFo as they weren't considered in curve fitting

    drop_subset = df.loc[np.isnan(df['AFo_bar'])==True]
    df = df.drop(drop_subset.index)

    # 3) now can drop the extra columns AFo_bar_a, AFo_bar_b, _merge
    df = df.drop(columns = ["AFo_bar_a", "AFo_bar_b", "_merge"])

    return df

def merge(source_path, destination_path):

    ''' This function generates a merged machine learning ready dataset.

    It returns a Pandas dataframe containing the ground truth merged dataset of positive and negative observations.
    
    Parameters
    ----------
    source_path : str
        String containing path of source directory, including the Unix wildcard * to indicate to the function to retrieve all files therein.
    
    destination_path : str
        String containing path of destination directory.
        
    Returns
    -------
    Pandas.DataFrame
        DataFrame containing the resulting merged datafiles from destination_path.
    '''

    # Move relevant preprocessed Excel files from Pt. 1 and Pt. 2 to a central destination folder for data merging
    move(source_path, destination_path)

    all_files = glob.glob("{}/*.xlsx".format(destination_path))

    # 1 & 2) indicate which data format, mean based or replicate based, (from the output of the preprocessing) will be used for machine learning
    indicator1 = 'replicate'
    indicator2 = 'all'
    indicator3 = 'mean'

    # 1) need to merge all the input polymer files with a significant AFo into one tidy longform dataframe 
    selected_files = [file for file in all_files if indicator3 in file and indicator2 not in file]
    polymer_names = [re.search('mean_(.+?).xlsx', file).group(1).strip() for file in selected_files]
    selected_dataframes = [pd.read_excel(file, header = [0, 1], index_col = [0,1,2,3]) for file in selected_files]

    ################# REFACTOR --> CLEAN #################

    # 1) clean list
    
    clean(selected_dataframes, polymer_names, True)
        
    ################# REFACTOR --> REFORMAT #################
    
    selected_dataframes_pos = reformat(selected_dataframes, True)

    # to balance the dataset, add BACK in the negative examples in preprocessing dropped due to statistical insignificance with an AFo = 0

    # 2) need to merge all the input polymer observations (significant and not) into a dataframe (mean all)
    selected_files_neg = [file for file in all_files if indicator3 in file and indicator2 in file]
    polymer_names_neg = [re.search('all_(.+?).xlsx', file).group(1).strip() for file in selected_files_neg]
    selected_dataframes_neg_list = [pd.read_excel(file, header = [0, 1], index_col = [0,1,2,3]) for file in selected_files_neg]

    ################# REFACTOR --> CLEAN #################

    # 2) clean list
    
    clean(selected_dataframes_neg_list, polymer_names_neg, False)
        
    ################# REFACTOR --> REFORMAT #################
        
    selected_dataframes_neg = reformat(selected_dataframes_neg_list, False)
    
    ################# REFACTOR --> JOIN #################

    return join(selected_dataframes_pos, selected_dataframes_neg)