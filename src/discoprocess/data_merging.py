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
        #print(df)
        #print(type(df))

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
    Performs other reformatting operations as required.

    Note - obsolete, remove in future'''

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
    Note - obsolete - remove in future

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
    Note: obsolete - remove in future
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
        used for binary classification in May '21, shows individual observations and statistical test results with labels suitable for binary encoding
        Note: observed data is likely confounded by sat time, not handled properly, use ETL function instead
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

    #print("\n\n")
    #print(selected_dataframes)
    #print("\n\n")

    # Grab replicate files for future join, same number of replicate files as mean files
    selected_files_pos_rep = [file for file in all_files if indicator1 in file and indicator2 not in file]
    polymer_names_pos_rep = [re.search('replicate_(.+?).xlsx', file).group(1).strip() for file in selected_files_pos_rep]
    selected_dataframes_rep = []

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
        #print(pd.read_excel(selected_files_pos_rep[i], header = [0], index_col = [0]))
        selected_dataframes_rep.append(pd.read_excel(selected_files_pos_rep[i], header = [0], index_col = [0]))

    #print(selected_dataframes_rep)
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

    # high level proton-specific data from true positive and true negatives in terms of AFo bar
    mean_df = join(selected_dataframes_pos, selected_dataframes_neg)

    # replicates_df = join_replicates(mean_df, selected_dataframes_pos_rep)

    # summary_df = summarize(replicates_df)

    return mean_df

def filepath_to_dfs(df_file_paths, polymer_names):
    '''Reads df file path list, cleans and converts to list of dataframes,
    set schemas to be consistent according to desired columns.

    Parameters:
    -----------
    df_file_paths: list
        list of filepaths to data

    polymer_names: string
        list of strings that correspond to the polymer names

    Returns:
    --------
    df_list: list
        list of dataframes containing polymer info
    '''

    df_list = []

    for ix, file in enumerate(df_file_paths):

        if "mean" in file and "all" not in file:
            try:  # ppm in index
                # clear default "unnamed" column names in multi index
                df = pd.read_excel(file, header=[0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
                df_other = pd.read_excel(file, header=[0, 1], index_col=[
                                        0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
                df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
                clean_df = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=(
                    "concentration", "sat_time", "proton_peak_index", "ppm"))


            except KeyError: # ppm in column

                # clear default "unnamed" column names in multi index
                df = pd.read_excel(file, header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
                df_other = pd.read_excel(file, header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
                df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
                clean_df = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

                # add ppm back to index
                df_ppm = clean_df['ppm']['mean']  # grab mean ppm
                clean_df = clean_df.drop(columns='ppm', level=0)  # drop extra vals
                clean_df['ppm'] = df_ppm  # leave only mean ppm
                clean_df.set_index('ppm', append=True, inplace=True)

        elif "replicate" in file:
            clean_df = pd.read_excel(file)

        if "polymer_name" not in clean_df.columns:
            clean_df['polymer_name'] = polymer_names[ix]

        df_list.append(clean_df)

    return df_list

def etl_per_sat_time(source_path, destination_path):
    ''' This function extracts, transforms, and loads data into a merged machine learning ready dataset of DISCO experiments.
    Done per sat time including curve fit params for downstream quality checking, and disco effect values for all binding and
    non-binding protons.

    Parameters
    ----------
    source_path : str
        String containing path of source directory, including the Unix wildcard * to indicate to the function to retrieve all files therein.

    destination_path : str
        String containing path of destination directory.

    Returns
    -------
    summary_df: Pandas.DataFrame
        data reduced to the statistical summary of each experiment per sat time
    '''

    # move relevant preprocessed Excel files to central folder for data ETL
    move(source_path, destination_path)

    all_files = glob.glob("{}/*.xlsx".format(destination_path))
    # grab list of filepaths for:
    # - all replicates (bind and not bind),
    # - binding only replicates (with individual AFo & SSE)
    # - mean binding only data tables (AFo_bar and SSE_bar)

    rep_all = [file for file in all_files if 'replicate_all' in file]
    rep_bind = [file for file in all_files if 'replicate' in file and 'all' not in file]
    mean_bind = [file for file in all_files if 'mean' in file and 'all' not in file]


    # grab polymer names
    polymer_names_rep = [re.search('replicate_all_(.+?).xlsx', file).group(1).strip() for file in rep_all]
    polymer_names_rep_bind = [re.search('replicate_(.+?).xlsx', file).group(1).strip() for file in rep_bind]
    polymer_names_mean = [re.search('mean_(.+?).xlsx', file).group(1).strip() for file in mean_bind]

    # define keys and values for final table ETL
    primary_key_rep = ["polymer_name", "concentration", "proton_peak_index", "replicate", "sat_time"]
    primary_key_mean = ["polymer_name", "concentration", "proton_peak_index", "sat_time"]

    rep_all = filepath_to_dfs(rep_all, polymer_names_rep)
    rep_bind = filepath_to_dfs(rep_bind, polymer_names_rep_bind)
    mean_bind = filepath_to_dfs(mean_bind, polymer_names_mean)

    # concatenate polymer specific data of each type into cross-polymer tables
    rep_all_df = pd.concat(rep_all)
    rep_bind_df = pd.concat(rep_bind)
    mean_bind_df = pd.concat(mean_bind).reset_index() # make tidy

    # round away ppm noise
    rep_all_df['ppm'] = rep_all_df['ppm'].round(2)
    rep_bind_df['ppm'] = rep_bind_df['ppm'].round(2)

    # subset replicates to unique values of interest
    rep_all_df = rep_all_df[["polymer_name", "concentration", "proton_peak_index", "replicate", "sat_time", "ppm", "amp_factor", "corr_%_attenuation"]].drop_duplicates() # subset=["polymer_name", "concentration", "proton_peak_index", "replicate", "sat_time", "amp_factor"]
    rep_bind_df = rep_bind_df[["polymer_name", "concentration", "proton_peak_index", "replicate", "sat_time", "ppm", "corr_%_attenuation", "AFo", "SSE", "yikj", "alpha", "beta"]].drop_duplicates()
    mean_bind_df = mean_bind_df[["polymer_name", "concentration", "proton_peak_index", "sample_size", "sat_time", "AFo_bar", "SSE_bar", "yikj_bar", "alpha_bar", "beta_bar"]].drop_duplicates().droplevel(1, axis=1)

    # join replicate-specific AFo and SSE, clean noise from ppm
    midpoint_df = pd.merge(rep_all_df, rep_bind_df, how = 'left', on = primary_key_rep).drop(columns=['ppm_y', 'corr_%_attenuation_y']).rename(columns = {'ppm_x':'ppm', "corr_%_attenuation_x":"corr_%_attenuation"})


    # fill non-binding replicate peaks with zeros
    midpoint_df[['AFo', 'SSE', 'yikj', 'alpha', 'beta']] = midpoint_df[['AFo', 'SSE', 'yikj','alpha', 'beta']].fillna(0)

    # ensure consistent data types in primary key
    midpoint_df.polymer_name = midpoint_df.polymer_name.astype(str).str.replace(' ', '')
    mean_bind_df.polymer_name = mean_bind_df.polymer_name.astype(str).str.replace(' ', '')
    midpoint_df.concentration = midpoint_df.concentration.astype(float)
    mean_bind_df.concentration = mean_bind_df.concentration.astype(float)
    midpoint_df.proton_peak_index = midpoint_df.proton_peak_index.astype(int)
    mean_bind_df.proton_peak_index = mean_bind_df.proton_peak_index.astype(int)

    # sort values
    midpoint_df = midpoint_df.sort_values(by = primary_key_mean)
    mean_bind_df = mean_bind_df.sort_values(by = primary_key_mean)

    # join AFobar and SSEbar
    summary_df = pd.merge(midpoint_df, mean_bind_df, how='left', on = primary_key_mean)

    # for non binding, fill with zeros
    summary_df[['SSE_bar','AFo_bar', 'yikj_bar','alpha_bar','beta_bar']] = summary_df[['SSE_bar','AFo_bar', 'yikj_bar','alpha_bar','beta_bar']].fillna(0)

    # fill in sample size if have data, if not calculate based on max num replicates
    sample_size_mapper = summary_df[['polymer_name','concentration', 'sample_size']].drop_duplicates()
    replicate_mapper = summary_df[['polymer_name', 'concentration', 'replicate']].drop_duplicates()

    for row in sample_size_mapper.itertuples():
        polymer = row[1]
        conc = row[2]
        sample_size = row[3]

        if pd.isna(sample_size):
            # grab max num replicates for that polymer and conc
            max_nrep = replicate_mapper.loc[(replicate_mapper['polymer_name'] == polymer) & (replicate_mapper['concentration'] == conc), ('replicate')].max()
            summary_df.loc[(summary_df['polymer_name'] == polymer) & (summary_df['concentration'] == conc), ('sample_size')] = max_nrep

        else:
            summary_df.loc[(summary_df['polymer_name'] == polymer) & (summary_df['concentration'] == conc), ('sample_size')] = sample_size

    return summary_df

def etl_per_proton(summary_df):
    '''Subsets the full ETL to a smaller, per proton table.
    '''
    proton_df = summary_df.drop(columns = ["AFo_bar", "SSE_bar", "yikj_bar", "alpha_bar", "beta_bar"]).copy()
    primary_key1 = ["polymer_name", "concentration", "proton_peak_index", "sat_time", "ppm"]
    mean_proton_df = proton_df.groupby(by = primary_key1).mean().reset_index()

    # pivot to move sat time into feature cols
    primary_key2 = ["polymer_name", "concentration", "proton_peak_index", "ppm"]
    sat_time_cols = mean_proton_df.pivot(index = primary_key2, columns = ["sat_time"], values = ["corr_%_attenuation"]).reset_index()
    sat_time_cols.columns = sat_time_cols.columns.map('{0[0]}{0[1]}'.format) # make one level from multi index

    # grab all other mean data
    all_other_cols = mean_proton_df.drop(columns = ["sat_time", "corr_%_attenuation","replicate", "yikj"]).drop_duplicates()

    # merge into final df
    mean_binding_df = pd.merge(all_other_cols, sat_time_cols, on = primary_key2).drop_duplicates()

    return mean_binding_df

def etl_per_replicate(source_path, destination_path):
    ''' This function extracts, transforms, and loads data into a merged dataset of DISCO experiments.
    Extracted on a per technical replicate basis, AFo and SSE only provided (not buildup curve quality params).

    Parameters
    ----------
    source_path : str
        String containing path of source directory, including the Unix wildcard * to indicate to the function to retrieve all files therein.

    destination_path : str
        String containing path of destination directory.

    Returns
    -------
    summary_df: Pandas.DataFrame
        data reduced to the statistical summary of each experiment
    '''

    # move relevant preprocessed Excel files to central folder for data ETL
    move(source_path, destination_path)

    all_files = glob.glob("{}/*.xlsx".format(destination_path))

    # grab list of filepaths for:
    # - all replicates (bind and not bind),
    # - binding only replicates (with individual AFo & SSE)
    # - mean binding only data tables (AFo_bar and SSE_bar)

    rep_all = [file for file in all_files if 'replicate_all' in file]
    rep_bind = [
        file for file in all_files if 'replicate' in file and 'all' not in file]
    mean_bind = [
        file for file in all_files if 'mean' in file and 'all' not in file]

    # grab polymer names
    polymer_names_rep = [re.search(
        'replicate_all_(.+?).xlsx', file).group(1).strip() for file in rep_all]
    polymer_names_rep_bind = [
        re.search('replicate_(.+?).xlsx', file).group(1).strip() for file in rep_bind]
    polymer_names_mean = [
        re.search('mean_(.+?).xlsx', file).group(1).strip() for file in mean_bind]

    # define keys and values for final table ETL
    primary_key_rep = ["polymer_name", "concentration",
                       "proton_peak_index", "replicate"]
    primary_key_mean = ["polymer_name", "concentration", "proton_peak_index"]

    rep_all = filepath_to_dfs(rep_all, polymer_names_rep)
    rep_bind = filepath_to_dfs(rep_bind, polymer_names_rep_bind)
    mean_bind = filepath_to_dfs(mean_bind, polymer_names_mean)

    # concatenate polymer specific data of each type into cross-polymer tables
    rep_all_df = pd.concat(rep_all)
    rep_bind_df = pd.concat(rep_bind)
    mean_bind_df = pd.concat(mean_bind).reset_index()  # make tidy

    # round away ppm noise
    rep_all_df['ppm'] = rep_all_df['ppm'].round(2)
    rep_bind_df['ppm'] = rep_bind_df['ppm'].round(2)

    # subset replicates to unique values of interest
    rep_all_df = rep_all_df[["polymer_name", "concentration", "proton_peak_index", "replicate", "ppm", "amp_factor"]].drop_duplicates(
        subset=["polymer_name", "concentration", "proton_peak_index", "replicate", "amp_factor"])
    rep_bind_df = rep_bind_df[["polymer_name", "concentration", "proton_peak_index", "replicate", "ppm", "AFo", "SSE"]].drop_duplicates(
        subset=["polymer_name", "concentration", "proton_peak_index", "replicate", "AFo", "SSE"])
    mean_bind_df = mean_bind_df[["polymer_name", "concentration", "proton_peak_index", "sample_size",
                                 "SSE_bar", "AFo_bar"]].drop_duplicates().droplevel(1, axis=1)

    # join replicate-specific AFo and SSE, clean noise from ppm
    midpoint_df = pd.merge(rep_all_df, rep_bind_df, how='left', on=primary_key_rep).drop(
        columns=['ppm_y']).rename(columns={'ppm_x': 'ppm'})

    # fill non-binding replicate peaks with zeros
    midpoint_df[['AFo', 'SSE']] = midpoint_df[['AFo', 'SSE']].fillna(0)

    # ensure consistent data types in primary key
    midpoint_df.polymer_name = midpoint_df.polymer_name.astype(
        str).str.replace(' ', '')
    mean_bind_df.polymer_name = mean_bind_df.polymer_name.astype(
        str).str.replace(' ', '')
    midpoint_df.concentration = midpoint_df.concentration.astype(float)
    mean_bind_df.concentration = mean_bind_df.concentration.astype(float)
    midpoint_df.proton_peak_index = midpoint_df.proton_peak_index.astype(int)
    mean_bind_df.proton_peak_index = mean_bind_df.proton_peak_index.astype(int)

    # sort values
    midpoint_df = midpoint_df.sort_values(by=primary_key_mean)
    mean_bind_df = mean_bind_df.sort_values(by=primary_key_mean)

    # join AFobar and SSEbar
    summary_df = pd.merge(midpoint_df, mean_bind_df,
                          how='left', on=primary_key_mean)

    # for non binding, fill with zeros
    summary_df[['SSE_bar', 'AFo_bar']] = summary_df[['SSE_bar', 'AFo_bar']].fillna(0)

    # fill in sample size if have data, if not calculate based on max num replicates
    sample_size_mapper = summary_df[[
        'polymer_name', 'concentration', 'sample_size']].drop_duplicates()
    replicate_mapper = summary_df[[
        'polymer_name', 'concentration', 'replicate']].drop_duplicates()

    for row in sample_size_mapper.itertuples():
        polymer = row[1]
        conc = row[2]
        sample_size = row[3]

        if pd.isna(sample_size):
            # grab max num replicates for that polymer and conc
            max_nrep = replicate_mapper.loc[(replicate_mapper['polymer_name'] == polymer) & (
                replicate_mapper['concentration'] == conc), ('replicate')].max()
            summary_df.loc[(summary_df['polymer_name'] == polymer) & (
                summary_df['concentration'] == conc), ('sample_size')] = max_nrep

        else:
            summary_df.loc[(summary_df['polymer_name'] == polymer) & (
                summary_df['concentration'] == conc), ('sample_size')] = sample_size

    return summary_df
