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
    
    print("Files for merging have been moved to the destination directory.")

    return

def clean(df_list, polymer_list, pos_or_neg = 'pos'):
    """This function cleans the dataframes in the inputted list by dropping extra columns, appending the missing polymer_name column, 
    and drops the appropriate columns to the index level.
    
    Parameters
    ----------
    df_list : list
        List of Pandas DataFrames to be cleaned.
    
    polymer_list : list
        List of polymer names contained in the passed DataFrames.
    
    pos_or_neg : str, {'pos', 'neg'}
        String to indicate path for positive binding observations (pos) or for negative binding observations (neg).

    Notes
    -----
    Relies on mutability nature of list.
    """
    # 1) clean list
    for i in range(len(df_list)):
        if 'polymer_name' not in df_list[i].columns:
            df_list[i].insert(loc = 0, column = 'polymer_name', value = polymer_list[i]) # add polymer name column
        
        if 'ppm' in df_list[i].columns:
            # drop (ppm, mean) to index
            df_list[i].set_index(('ppm', 'mean'), append=True, inplace=True)
            
            # renaming new index name to ppm
            df_list[i].index.names=['concentration', 'sat_time', 'proton_peak_index', 'ppm']
            
            # drop (ppm, std)
            df_list[i] = df_list[i].drop(('ppm', 'std'), axis = 1) 

        # drop extra level
        df_list[i] = df_list[i].droplevel(1, axis = 1)

        if pos_or_neg == 'pos':
            # drop extra columns
            drop_data = df_list[i].loc[:, ['corr_%_attenuation','dofs', 'amp_factor', 'yikj_bar','SSE_bar']]
            
        else:
            # drop other columns not needed and extra level
            drop_data = df_list[i].loc[:, ['corr_%_attenuation', 'dofs', 'amp_factor']]
        
        df_list[i] = df_list[i].drop(drop_data.columns, axis = 1)

def clean_replicates(df_list, polymer_list):
    '''Cleans replicate dataframes to ensure consistent schema.

    Parameters:
    -----------
    df_list: list of Pandas.DataFrames
        individual polymer data 

    polymer_list: list
        polymer names that correspond to dataframes
    
    Returns: None
    -------

    Notes:
    ------
    Operations performed in place
    '''

    for i, df in enumerate(df_list):
        
        # ensure polymer name in replicate table schema
        if 'polymer_name' not in df.columns:
            df.insert(loc=0, column='polymer_name', value=polymer_list[i])
        
        # remove ppm range from schema                
        if 'ppm_range' in df.columns:
            df = df.drop(columns=['ppm_range'])

    return 

def reformat(df_list, pos_or_neg = 'pos'):
    """This function takes in a list of Pandas DataFrames, concatenates them, and returns the final reformatted DataFrame. 
    
    Parameters
    ----------
    df_list : list
        List of Pandas DataFrames to be concatenated and reformatted.
    
    pos_or_neg : str, {'pos', 'neg'}
        String to indicate path for positive binding observations (pos) or for negative binding observations (neg).
        
    Returns
    -------
    df : Pandas.DataFrame
        Final reformatted DataFrame.
    """
    # concatenate
    df = pd.concat(df_list)

    if pos_or_neg == 'pos':
        
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

def reformat_replicates(df_list):
    ''' Concatenates true positive replicate binding data into one data table.
    Performs other reformatting operations as required.'''
    
    df = pd.concat(df_list)

    # drop duplicates from saturation time
    df = df.drop(columns=['sat_time', 'amp_factor', 'yikj', 'corr_%_attenuation', 'index','ppm_range'])
    df = df.drop_duplicates()

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

    key = ['concentration','proton_peak_index','ppm','polymer_name','sample_size','t_results','significance'] # combo of values that define a unique observation in each table

    # first - need to join ALL the data from the two tables
    # Outer Join A and B on the Key for unique observations, and include which table they were merged from for filtering

    df = pd.merge(left = df1, right = df2, how = 'outer', on = key, indicator = True, suffixes = ("_a","_b"))

    # allocate the correct AFo_bar values from each table 
    df.loc[np.isnan(df['AFo_bar_a']) == False, ('AFo_bar')] = df['AFo_bar_a'] # assigns AFo_bar_a values to observations where they exist
    df.loc[np.isnan(df['AFo_bar_a']) != False, ('AFo_bar')] = df['AFo_bar_b'] # assigns AFo_bar_b where no AFo_bar_a value does not exist (i.e. polymer was all non)

    # remaining NaNs in table_c have significance = True, 
    # BUT these were not used in Curve Fitting AFo calculations, due to strict drop criterion in preprocessing. 
    # Therefore should still drop them here, as we don't know their true AFo as they weren't considered in curve fitting

    drop_subset = df.loc[np.isnan(df['AFo_bar'])==True]
    df = df.drop(drop_subset.index)

    # now can drop the extra columns AFo_bar_a, AFo_bar_b, _merge
    df = df.drop(columns = ["AFo_bar_a", "AFo_bar_b", "_merge"])
    
    # extra column usually added for book input only
    if 'level_2' in df.columns:
        df = df.drop('level_2', axis=1)

    return df

def join_replicates(df1, df2):
    '''One to many join the replicate specific AFo replicate data into global dataset.

    Parameters:
    ----------
    df1: pandas.DataFrame
        the "one" dataframe

    df2: pandas.DataFrame
        the "many" dataframe

    Returns:
    -------
    merged_df: pandas.DataFrame
        the df after the join
    '''
    # set df1 schema
    df1 = df1[['concentration', 'proton_peak_index', 'ppm',
               'polymer_name', 'sample_size','AFo_bar']]

    # match schema of df1 + new cols
    df2 = df2[['concentration', 'proton_peak_index', 'ppm',
               'polymer_name', 'replicate','AFo','SSE']]

    # define primary key for the join
    key = ['concentration', 'proton_peak_index', 'polymer_name']

    merged_df = pd.merge(df1, df2, on = key, how = 'left')

    # clean redundant ppm noise - proton peak index ensures correct mappings during join
    merged_df = merged_df.drop(columns='ppm_y')
    merged_df = merged_df.drop_duplicates()

    return merged_df.rename(columns={'ppm_x': 'ppm'})

def summarize(df):
    ''' Takes a proton-specific dataframe and summarizes it in terms of summary statistics per experiment.
    Parameters:
    -----------
    df: Pandas.Dataframe
        contains proton-specific dataframe (output of join_replicates)

    Returns:
    -------
    summary_df: Pandas.Dataframe
        summary statistics of the experiment
    '''

    polymers = df['polymer_name'].unique()

    snapshot_list = []

    for polymer in polymers:

        polymer_df = df.loc[df['polymer_name']==polymer].copy()
        concentrations = polymer_df['concentration'].unique()
        peaks = polymer_df['proton_peak_index'].unique()

        for c in concentrations:
            for p in peaks:
                print("polymer", polymer)
                print("conc:", c)
                print("peak", p)
                snapshot_df = polymer_df.loc[(polymer_df['concentration'] == c) & (polymer_df['proton_peak_index'] == p)].copy()

                # if any of the AF0_bar values are NON zero, means the snapshot is a binding snapshot
                binding_flag = snapshot_df['AFo_bar'].any()

                if binding_flag == True: # reduce data to only the statistical binding summary for binding observations

                    drop_index = snapshot_df.loc[snapshot_df['AFo_bar']==0].index
                    snapshot_df = snapshot_df.drop(index = drop_index)
                    snapshot_df = snapshot_df.drop_duplicates(subset = ['concentration', 'proton_peak_index', 'polymer_name', 'sample_size', 'AFo_bar', 'replicate', 'AFo', 'SSE'])
                    
                # append to snapshot list
                snapshot_list.append(snapshot_df)

            
    summary_df = pd.concat(snapshot_list)

    return summary_df

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
    mean_df: Pandas.DataFrame
        useful for binary classification, shows individual observations and statistical test results with labels suitable for binary encoding
    
    replicates_df: Pandas.DataFrame
        mean_df including average results of each underlying replicate
    
    summary_df: Pandas.DataFrame
        data reduced to the statistical summary of each experiment
    '''

    # Move relevant preprocessed Excel files from Pt. 1 and Pt. 2 to a central destination folder for data merging
    move(source_path, destination_path)

    all_files = glob.glob("{}/*.xlsx".format(destination_path))

    # flags for data table selection
    indicator1 = 'replicate'
    indicator2 = 'all'
    indicator3 = 'mean'

    # Merge all the "mean" input polymer files with a significant AFo bar into one tidy longform dataframe 
    selected_files_pos = [file for file in all_files if indicator3 in file and indicator2 not in file]
    polymer_names_pos = [re.search('mean_(.+?).xlsx', file).group(1).strip() for file in selected_files_pos]
    #selected_dataframes = [pd.read_excel(file, header = [0, 1], index_col = [0,1,2,3]) for file in selected_files]
    selected_dataframes = selected_files_pos.copy() 

    # Grab replicate files for future join, same number of replicate files as mean files
    selected_files_pos_rep = [file for file in all_files if indicator1 in file and indicator2 not in file]
    polymer_names_pos_rep = [re.search('replicate_(.+?).xlsx', file).group(1).strip() for file in selected_files_pos_rep]
    selected_dataframes_rep = selected_files_pos_rep.copy()

    # generate lists of true positive dataframes
    for i in range(len(selected_files_pos)):

        # mean dataframes across polymers
        try: # ppm in index
            # Preserve multi-index when reading in Excel file
            df = pd.read_excel(selected_files_pos[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
            df_other = pd.read_excel(selected_files_pos[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
            df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
            selected_dataframes[i] = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

        except KeyError: # ppm in column
            # Preserve multi-index when reading in Excel file
            df = pd.read_excel(selected_files_pos[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
            df_other = pd.read_excel(selected_files_pos[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
            df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
            selected_dataframes[i] = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
    
        # replicate dataframes across polymers
        selected_dataframes_rep[i] = pd.read_excel(selected_files_pos_rep[i], header = [0], index_col = [0])
        

    clean(selected_dataframes, polymer_names_pos, 'pos')

    clean_replicates(selected_dataframes_rep, polymer_names_pos_rep)

    selected_dataframes_pos = reformat(selected_dataframes, 'pos')

    selected_dataframes_pos_rep = reformat_replicates(selected_dataframes_rep)

    # to balance the dataset, add back in the negative examples in preprocessing dropped due to statistical insignificance (AFo bar = 0)
    selected_files_neg = [file for file in all_files if indicator3 in file and indicator2 in file]
    polymer_names_neg = [re.search('all_(.+?).xlsx', file).group(1).strip() for file in selected_files_neg]
    #selected_dataframes_neg_list = [pd.read_excel(file, header = [0, 1], index_col = [0,1,2,3]) for file in selected_files_neg]
    selected_dataframes = selected_files_neg.copy()

    for i in range(len(selected_files_neg)):
    
        try: # ppm in index
            # Preserve multi-index when reading in Excel file
            df = pd.read_excel(selected_files_neg[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
            df_other = pd.read_excel(selected_files_neg[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
            df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
            selected_dataframes[i] = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

        except KeyError: # ppm in column
            # Preserve multi-index when reading in Excel file
            df = pd.read_excel(selected_files_neg[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
            df_other = pd.read_excel(selected_files_neg[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
            df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
            selected_dataframes[i] = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

    clean(selected_dataframes, polymer_names_neg, 'neg')
        
    selected_dataframes_neg = reformat(selected_dataframes, 'neg')

    # high level proton-specific data from true positive and true negatives in terms of AFo bar, can use for binary classification
    mean_df = join(selected_dataframes_pos, selected_dataframes_neg)

    # add in replicate specific AFo and SSE
    replicates_df = join_replicates(mean_df, selected_dataframes_pos_rep)
    
    # snapshot of the clean summary of each experiment (drops proton-specific information)
    summary_df = summarize(replicates_df)

    return mean_df, replicates_df, summary_df
