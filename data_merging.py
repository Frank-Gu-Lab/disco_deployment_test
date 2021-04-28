# Part 3 Preprocessing Function - merges positive and negative data 

# import packages
import pandas as pd
import numpy as np
import os
import glob
import re

# import functions
from move_wrangled_data import move

def merge(source_path, destination_path, merge_output_directory):

    ''' 
    This function generates a merged machine learning ready dataset.

    It calls the move function to move relevant preprocessing Excel file outputs from Pt. 1 and Pt. 2 to a central repository
    for data merging.

    It returns a Pandas dataframe containing the ground truth merged dataset of positive and negative observations.

    TO DO: Refactor this spaghetti code into something more maintainable + write tests. 

    Split this into multiple functions - each function should do only one thing. (Matching Numbers added in comments to help w refactor)
    - 1) Merge and clean applicable input folder data to generate a true positive dataset (+ class as determined in preprocessing)
    - 2) Merge and clean applicable input folder data to generate a true negative dataset (- class as determined in preprocessing)
    - 3) Merge Positive and Negative Example Datasets into one ground truth clean dataset 

    '''

    # Move relevant preprocessed Excel files from Pt. 1 and Pt. 2 to a central destination folder for data merging
    move(source_path, destination_path)

    # 1 & 2) create list of all Excel file candidates for merging
    # input_path = os.path.abspath(merge_output_directory)
    # all_files = glob.glob("{merge_output_directory}/*.xlsx")
    all_files = glob.glob("{}/*.xlsx".format(destination_path))

    # 1 & 2) indicate which data format, mean based or replicate based, (from the output of the preprocessing) will be used for machine learning
    indicator1 = 'replicate'
    indicator2 = 'all'
    indicator3 = 'mean'

    # 1) need to merge all the input polymer files with a significant AFo into one tidy longform dataframe 
    selected_files = [file for file in all_files if indicator3 in file and indicator2 not in file]
    polymer_names = [re.search('mean_(.+?).xlsx', file).group(1).strip() for file in selected_files]
    selected_dataframes = [pd.read_excel(file, header = [0, 1], index_col = [0,1,2,3]) for file in selected_files]

    # 1) clean list
    for i, df in enumerate(selected_dataframes):
        
        if 'polymer_name' not in df.columns:
            selected_dataframes[i].insert(loc = 0, column = 'polymer_name', value = polymer_names[i]) # add polymer name column
        
        # drop extra level
        selected_dataframes[i] = selected_dataframes[i].droplevel(1, axis = 1)
        
        # drop extra columns
        drop_data_mean = selected_dataframes[i].loc[:,
                                ['corr_%_attenuation','dofs',
                                'amp_factor', 'yikj_bar','SSE_bar']]
        
        selected_dataframes[i] = selected_dataframes[i].drop(drop_data_mean.columns, axis = 1)
        
        # if this column exists as a column (it should be in the index), also drop
        if "ppm" in df.columns:
            selected_dataframes[i] = selected_dataframes[i].drop("ppm", axis = 1)
        
            
    # 1) concatenate complete df of the positive data   
    data = pd.concat(selected_dataframes)

    # 1) drop sat time from index
    data.index = data.index.droplevel(1)

    # 1) reset and reformat index
    data = data.reset_index()
    data = data.rename(columns = {'level_2': "ppm"})
    data = data.rename_axis(columns = "index")

    # 1) round ppm values to 4 sig figs
    data['ppm'] = np.round(data.ppm.values, 4)

    # to balance the dataset, add BACK in the negative examples in preprocessing dropped due to statistical insignificance with an AFo = 0

    # 2) need to merge all the input polymer observations (significant and not) into a dataframe (mean all)
    selected_files_neg = [file for file in all_files if indicator3 in file and indicator2 in file]
    polymer_names_neg = [re.search('all_(.+?).xlsx', file).group(1).strip() for file in selected_files_neg]
    selected_dataframes_neg_list = [pd.read_excel(file, header = [0, 1], index_col = [0,1,2,3]) for file in selected_files_neg]

    # 2) clean list
    for i, df in enumerate(selected_dataframes_neg_list):
        df.insert(loc = 0, column = 'polymer_name', value = polymer_names_neg[i]) # add polymer name column
        
        if df.index.names[3] == None: # drop ppm std column not needed
            drop_data_2 = df.iloc[:, 1]
            selected_dataframes_neg_list[i] = df.drop(drop_data_2.name, axis = 1)
            
            # drop other columns not needed and extra level
            drop_data_3 = df.loc[:, 
                [('corr_%_attenuation',                'mean'),
                ('corr_%_attenuation',                 'std'),
                (              'dofs',  'Unnamed: 7_level_1'),
                (        'amp_factor', 'Unnamed: 11_level_1')]]
            selected_dataframes_neg_list[i] = df.drop(drop_data_3.columns, axis = 1).droplevel(1, axis = 1)
            

        else:
            # drop other columns not needed, and drop extra level in column index
            drop_data_3 = df.iloc[:, [1,2,3,7]]
            selected_dataframes_neg_list[i] = df.drop(drop_data_3.columns, axis = 1).droplevel(1, axis = 1)
        
    # 2) concat into one df
    selected_dataframes_neg = pd.concat(selected_dataframes_neg_list)

    # 2) drop extra ppm column that gets created from combining multi indices
    selected_dataframes_neg = selected_dataframes_neg.drop(columns = 'ppm')

    # 2) drop sat time from index
    selected_dataframes_neg.index = selected_dataframes_neg.index.droplevel(1)

    # 2) reset and reformat index
    selected_dataframes_neg = selected_dataframes_neg.reset_index()
    selected_dataframes_neg = selected_dataframes_neg.rename_axis(columns = "index")

    # 2) remove observations with sample size of 1 as that is not enough to know significance
    small_samples_mask = selected_dataframes_neg[selected_dataframes_neg['sample_size'] == 1]
    selected_dataframes_neg = selected_dataframes_neg.drop(index = small_samples_mask.index)

    # 2) if t test results are NaN, also remove, as don't know true data characteristics
    t_nan_mask = selected_dataframes_neg[np.isnan(selected_dataframes_neg['t_results'].values) == True]
    selected_dataframes_neg = selected_dataframes_neg.drop(index = t_nan_mask.index)

    # 1 & 2) assign AFo to zero in both dataframes if significance is FALSE
    selected_dataframes_neg.loc[selected_dataframes_neg['significance'] == False, ('AFo_bar')] = 0.0 # assigns true negatives
    data.loc[data['significance'] == False, ('AFo_bar')] = 0.0 # removes false associations, as these values weren't considered in curve fitting

    # 3) now need to JOIN data from the two tables in a way that makes sense
    table_a = data                          # contains all data post preprocessing with true positives
    table_b = selected_dataframes_neg       # contains true negatives, and some empty points that were sig but dropped before curve fitting due to strict criterion

    key = ['concentration','proton_peak_index','ppm','polymer_name','sample_size','t_results','significance'] # combo of values that define a unique observation in each table

    # 3) first - need to join ALL the data from the two tables
    # 3) Outer Join A and B on the Key for unique observations, and include which table they were merged from for filtering

    table_c = pd.merge(left = table_a, right = table_b, how = 'outer', on = key, indicator = True, suffixes = ("_a","_b"))

    # 3) allocate the correct AFo_bar values from each table 
    table_c.loc[np.isnan(table_c['AFo_bar_a']) == False, ('AFo_bar')] = table_c['AFo_bar_a'] # assigns AFo_bar_a values to observations where they exist
    table_c.loc[np.isnan(table_c['AFo_bar_a']) != False, ('AFo_bar')] = table_c['AFo_bar_b'] # assigns AFo_bar_b where no AFo_bar_a value does not exist (i.e. polymer was all non)

    # 3) remaining NaNs in table_c have significance = True, 
    # BUT these were not used in Curve Fitting AFo calculations, due to strict drop criterion in preprocessing. 
    # Therefore should still drop them here, as we don't know their true AFo as they weren't considered in curve fitting

    drop_subset = table_c.loc[np.isnan(table_c['AFo_bar'])==True]
    table_c = table_c.drop(drop_subset.index)

    # 3) now can drop the extra columns AFo_bar_a, AFo_bar_b, _merge
    table_c = table_c.drop(columns = ["AFo_bar_a", "AFo_bar_b", "_merge"])

    return table_c


