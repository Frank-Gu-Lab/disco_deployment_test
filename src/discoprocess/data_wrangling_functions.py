# define functions used in data cleaning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
import scipy.stats as stats
from scipy.optimize import curve_fit
import os
import glob

# import helpers
try:
    from data_wrangling_helpers import *
    from data_plot import *
except:
    from .data_wrangling_helpers import *
    from .data_plot import *

# define handy shortcut for indexing a multi-index dataframe
idx = pd.IndexSlice

# functions are ordered below in the order they are called in the disco-data-processing script

def batch_to_dataframe(b):
    '''This function converts and cleans excel books of type "Batch" (containing many polymers in one book) into dataframes for further analysis.

    Parameters
    ----------
    b : str
        The file path to the excel book of interest.

    Returns
    -------
    list_of_clean_dfs : list
        List of tuples, where each tuple contains ('polymer_name', CleanPolymerDataFrame)
        Tuples are in a "key-value pair format", where the key (at index 0 of the tuple) is:
        current_book_title, a string containing the title of the current excel input book
        And the value (at index 1 of the tuple) is:
        clean_df, the cleaned pandas dataframe corresponding to that book title!
    '''
    # load excel book into Pandas
    current_book_title = os.path.basename(str(b))

    print("The current book being analyzed is: ", current_book_title)

    # return list of unique polymer names in the book, their replicates, and the sheets containing raw data to loop through
    unique_polymers, unique_polymer_replicates, name_sheets = initialize_excel_batch_replicates(b)

    # generate a new replicate index list that holds the nth replicate associated with each raw data sheet in book
    replicate_index = []

    for i in range(len(unique_polymer_replicates)):
        current_replicate_range = range(1,int(unique_polymer_replicates[i]+1),1)
        for j in current_replicate_range:
            replicate_index.append(j)

    # BEGIN WRANGLING DATA FROM THE EXCEL FILE, AND TRANSLATING INTO ORGANIZED DATAFRAME ----------------

    list_of_clean_dfs = wrangle_batch(b, name_sheets, replicate_index)

    # After all is said and done, return a list of the clean dfs containing polymer tuples of format (polymer_name, polymer_df)
    print("Returning a list of tuples containing all polymer information from Batch: ", b)

    return list_of_clean_dfs

def clean_the_batch_tuple_list(list_of_clean_dfs):
    '''This function performs simple data cleaning operations on batch processed data such that further analysis can be performed equivalently
    on inputs of batch or individual data formats.

    Parameters
    ----------
    list_of_clean_dfs : list
        List of dataframes to be cleaned.

    Returns
    -------
    final_clean_polymer_df_list : list
        List of clean polymer dataframes.
    '''

    print("Beginning batch data cleaning.")

    # initializing list of clean polymer dataframes to return
    final_clean_polymer_df_list = []

    # batch is a list of n batches provided as excel inputs
    for batch in list_of_clean_dfs:

        #batch[i][1] is the ith polymer dataframe within the batch
        for i in range(len(batch)):

            current_df = batch[i][1]

            #reset the index of the current df for easier cleaning/avoiding the setting with copy warning
            current_df = current_df.reset_index()

            # drop NaN rows from the current df and the old index column
            current_df = current_df.dropna(axis = 0, how = 'any', inplace = False)
            current_df = current_df.drop(columns = 'index')

            # remove brackets from polymer name if there are any
            clean_polymer_name = current_df['polymer_name'].apply(lambda x: x.split('(')[0])
            current_df.loc[:,'polymer_name'] = clean_polymer_name

            # change BSM and Control to 'sample' and 'control'
            sample_mask = current_df['sample_or_control'].str.contains('BSM')
            control_mask = current_df['sample_or_control'].str.contains('ontrol')
            current_df.loc[sample_mask, 'sample_or_control'] = 'sample'
            current_df.loc[control_mask, 'sample_or_control'] = 'control'

            # change irrad bool On/Off flags to True and False if not already a bool list
            if type(current_df['irrad_bool'].iloc[0]) != bool:
                irrad_true_mask = current_df['irrad_bool'].str.contains('On')
                irrad_false_mask = current_df['irrad_bool'].str.contains('Off')
                current_df.loc[irrad_true_mask, 'irrad_bool'] = True
                current_df.loc[irrad_false_mask, 'irrad_bool'] = False

            # convert ppm range to mean value in a new column
            ppm_content = current_df['ppm_range'].apply(lambda x: (float(x.split(' .. ')[0]) + float(x.split(' .. ')[1]))*(1/2) )
            current_df.insert(4, 'ppm', ppm_content)

            # convert proton_peak_index values from floats into ints via numpy int method
            ppi_content = current_df['proton_peak_index'].values.astype(int)
            current_df.loc[:,'proton_peak_index'] = ppi_content

            # append cleaned df to clean polymer data list
            final_clean_polymer_df_list.append(current_df)

    print("Batch data cleaning completed.\n")
    return final_clean_polymer_df_list


def add_attenuation(current_book):
    ''' This function calculates the attenuation if the dataframe passes all checks
    (based on the order of items) by means of simple arithmetic operations.

    Parameters
    ----------
    current_book : Pandas.DataFrame
        The dataframe output from the convert_excel_to_dataframe() function. (Or batch processing equivalent)

    batch_or_book : str, {'book', 'batch'}
        Default is book, but it will run the batch path if 'batch' is passed to function as the second arg.

    Returns
    -------
    intensity_irrad_true : Pandas.DataFrame
        An updated dataframe that includes the attenuation columns, contains the irradiated intensity datapoints.

    intensity_irrad_false : Pandas.DataFrame
        A dataframe containing the non-irradiated intensity datapoints.

    Raises
    ------
    ValueError
        If the subset dataframes ((corrected) true and false irrad dataframes) are not equal.
    '''

    # to get % attenuation of peak integral, define true and false irrad dataframes below, can perform simple subtraction if passes check


    intensity_irrad_true = current_book.loc[(current_book['irrad_bool'] == True), ['polymer_name', 'proton_peak_index', 'ppm_range', 'ppm', 'sample_or_control', 'replicate', 'concentration', 'sat_time', 'absolute']]
    intensity_irrad_false = current_book.loc[(current_book['irrad_bool'] == False), ['polymer_name', 'proton_peak_index',  'ppm_range', 'ppm', 'sample_or_control', 'replicate', 'concentration', 'sat_time', 'absolute']]

    # check if the fixed experimental values in irrad true and irrad false are equal, in the same order, and the same size, so that one to one calculations can be performed to calculate attenuation.
    fixed_values_equality_check = attenuation_calc_equality_checker(intensity_irrad_true, intensity_irrad_false)

    # if the test passes, calculate % attenuation and append to dataframe
    if fixed_values_equality_check == True:

        p_attenuation_intensity = intensity_irrad_false.absolute.values - intensity_irrad_true.absolute.values

        # Update irradiated dataframe to include the % attenuation of the irradiated samples and controls
        intensity_irrad_true['attenuation'] = p_attenuation_intensity
        print("Test 1 passed, attenuation has been calculated and appended to dataframe.")

    else:
        raise ValueError("Error, intensity_irrad true and false dataframes are not equal, cannot compute signal attenutation in a one-to-one manner.")

    return intensity_irrad_true, intensity_irrad_false

def add_corr_attenuation(intensity_irrad_true, intensity_irrad_false):
    """This function calculates the corr_%_attenuation if the dataframe passes all checks
    (based on the order of items) by means of simple arithmetic operations.

    Parameters
    ----------
    intensity_irrad_true : Pandas.DataFrame
        An updated dataframe that includes the attenuation columns, contains the irradiated intensity datapoints.

    intensity_irrad_false : Pandas.DataFrame
        A dataframe containing the non-irradiated intensity datapoints


    Returns
    -------
    corr_p_attenuation_df : Pandas.DataFrame
        An updated dataframe that includes the attenuation and corr_%_attenuation columns.

    Raises
    ------
    ValueError
        If the subset dataframes ((corrected) true and false irrad dataframes) are not equal.
    """
    # Now check we are good to calculate the corrected % attenuation, and then calculate: ------------------------

    #subset the dataframe where irrad is true for samples only
    p_atten_intensity_sample = intensity_irrad_true.loc[(intensity_irrad_true['sample_or_control'] == 'sample')]
    #subset the dataframe where irrad is true for controls only
    p_atten_intensity_control = intensity_irrad_true.loc[(intensity_irrad_true['sample_or_control'] == 'control')]
    #subset dataframe where irrad is false for samples only
    intensity_irrad_false_sample = intensity_irrad_false.loc[(intensity_irrad_false['sample_or_control'] == 'sample')]

    #check if the fixed experimental values in irrad true and irrad false are equal, in the same order, and the same size, so that one to one calculations can be performed to calculate attenuation.
    corr_atten_equality_check = corrected_attenuation_calc_equality_checker(p_atten_intensity_sample, p_atten_intensity_control, intensity_irrad_false_sample)

    if corr_atten_equality_check == True:

        #Grab Attenuation data for subset with samples only where irrad is true
        p_atten_intensity_sample_data = intensity_irrad_true.loc[(intensity_irrad_true['sample_or_control'] == 'sample')]['attenuation'].values
        #Grab Attenuation data for subset with controls only where irrad is true
        p_atten_intensity_control_data = intensity_irrad_true.loc[(intensity_irrad_true['sample_or_control'] == 'control')]['attenuation'].values
        #Grab Absolute peak integral data for subset with samples only where irrad is false to normalize attenuation
        intensity_irrad_false_sample_data = intensity_irrad_false.loc[(intensity_irrad_false['sample_or_control'] == 'sample')]['absolute'].values

        #initialize different cols
            #initialize dataframe to store corrected p_attenuation with the shared fixed parameters and sample identifier parameters
        corr_p_attenuation_df = pd.DataFrame(p_atten_intensity_sample[['polymer_name', 'concentration', 'sat_time', 'proton_peak_index', 'ppm_range', 'ppm', 'sample_or_control', 'replicate', 'absolute', 'attenuation']])

        #Calculate Corrected % Attentuation, as applies to
        corr_p_attenuation_df['corr_%_attenuation'] = ((1/intensity_irrad_false_sample_data)*(p_atten_intensity_sample_data - p_atten_intensity_control_data))
        print("Test 2 passed, corr_%_attenuation has been calculated and appended to dataframe.")

        return corr_p_attenuation_df

    else:
        raise ValueError("Error, input dataframes are not equal, cannot compute corrected signal attenutation in a one-to-one manner.")

def prep_mean(corr_p_attenuation_df):
    '''This function prepares the dataframe for statistical analysis after the attenuation and corr_%_attenuation
    columns have been added.

    Statisical analysis is performed on a "mean" basis, across many experimental replicates.
    This code prepares the per-observation data accordingly, and outputs the mean_df_for_stats.

    It drops the columns and rows not required for stats, calculates the mean and std of parameters we do
    care about, and also appends the degrees of freedom and sample size.

    Parameters
    ----------
    corr_p_attenuation_df : Pandas.DataFrame
        Dataframe after attenuation and corrected % attenuation have been calculated and added as columns
        (output from add_attenuation_and_corr_attenuation_to_dataframe).

    -------
    mean_corr_attenuation_ppm : Pandas.DataFrame
        Modified dataframe, where columns not required for statistical modelling are dropped and columns for the parameters of
        interest are appended.
    '''

    # if data came from a batch, ppm value is static and 1 less DoF, so perform adjusted operations

    # now drop the column fields that are not required for stats modelling and further analysis
    data_for_stats = corr_p_attenuation_df.drop(columns = ['sample_or_control', 'absolute', 'ppm_range'])

    # determine mean corr % attenuation per peak index, time, and concentration across replicates using groupby sum (reformat) and groupby mean (calculate mean)
    regrouped_df = data_for_stats.groupby(by = ['concentration','sat_time', 'proton_peak_index', 'replicate', 'ppm'])[['corr_%_attenuation']].sum()

    # generate a table that includes the mean and std for ppm and corr_%_atten across the replicates, reset index
    mean_corr_attenuation_ppm = regrouped_df.groupby(by = ['concentration', 'sat_time', 'proton_peak_index', 'ppm']).agg({'corr_%_attenuation': ['mean', 'std']})

    # prepare input for dofs function
    input_for_dofs = regrouped_df.index.get_level_values(2)

    # Calculate degrees of freedom and sample size for each datapoint using function above
    peak_index_array = np.array(input_for_dofs)

    dofs = get_dofs(peak_index_array, regrouped_df)
    # print(mean_corr_attenuation_ppm)
    # append a new column with the calculated degrees of freedom to the table for each proton peak index
    mean_corr_attenuation_ppm['dofs'] = dofs
    mean_corr_attenuation_ppm['sample_size'] = np.asarray(dofs) + 1

    return mean_corr_attenuation_ppm

def prep_replicate(corr_p_attenuation_df):
    '''This function prepares the dataframe for statistical analysis after the attenuation and corr_%_attenuation
    columns have been added.

    This code prepares the per-observation data accordingly, and outputs the replicate_df_for_stats.

    It drops the columns and rows not required for stats, and adds the proton peak index.

    Parameters
    ----------
    corr_p_attenuation_df : Pandas.DataFrame
        Dataframe after attenuation and corrected % attenuation have been calculated and added as columns
        (output from add_attenuation_and_corr_attenuation_to_dataframe).

    batch_or_book : str, {'book', 'batch'}
        Defaults to book processing path, but if 'batch' is passed to function will pursue batch path.

    Returns
    -------
    replicate_df_for_stats : Pandas.DataFrame
        Modified dataframe, where columns not required for statistical modelling are dropped and the column for proton peak index
        is appended.
    '''

    # drop any rows that are entirely null from the dataframe
    corr_p_attenuation_df = corr_p_attenuation_df.dropna(how = "any")

    replicate_df_for_stats = corr_p_attenuation_df.drop(columns = ['sample_or_control', 'absolute', 'attenuation'])

    return replicate_df_for_stats

def t_test(mean_corr_attenuation_ppm, p=0.05):
    ''' Tests to see if mean corr attenuation (DISCO effect) is statistically significantly different from zero
        at a given peak and saturation time point. One sample, two-sided t-test.

        Parameters
        ----------
        mean_corr_attenuation_ppm : Pandas.DataFrame
            Modified dataframe that has been prepared for statistical modelling (output from prep_mean_data_for_stats).

        p : float, default=0.05
            Threshold for statistical significance

        Returns
        -------
        mean_corr_attenuation_ppm : Pandas.DataFrame
            The input dataframe, now with the t-test values and results appended.

        Notes
        -----
        Book and batch paths are the same.

        Theory:
        ------
        Two sided Null/Alt Hyp
        --> H0: disco effect = 0 | H1: disco effect !=0

        Rejection Criteria for Fixed-Level Test
        --> |t0| > tcrit(alpha/2, n-1)
        --> abs(mean) - x0 / std/(sqrt(n)) > tcrit | here x0 = 0
        --> if abs(mean) - tcrit * (std/(sqrt(n))) > 0, reject null
    '''
    # Initialize Relevant t test Parameters
    dof = mean_corr_attenuation_ppm['dofs']
    mean_t = mean_corr_attenuation_ppm['corr_%_attenuation']['mean'].abs()
    std_t = mean_corr_attenuation_ppm['corr_%_attenuation']['std']
    sample_size_t = mean_corr_attenuation_ppm['sample_size']

    # retrieve value critical value, two sided test
    crit_t_value = t.ppf(q = 1-(p/2), df = dof)

    t_results =  mean_t - crit_t_value * (std_t/(np.sqrt(sample_size_t)))

    mean_corr_attenuation_ppm['t_results'] = t_results
    mean_corr_attenuation_ppm['significance'] = mean_corr_attenuation_ppm['t_results'] > 0

    #Return the dataframe with the new t test columns appended
    return mean_corr_attenuation_ppm

def compute_af(current_mean_stats_df, current_replicate_stats_df, af_denominator):
    '''This function computes an amplification factor for the mean stats df and the replicate stats df.
    Each polymer may have a different denominator to consider for the amp factor calculation, so it
    is passed into this function as a variable.

    Parameters
    ----------
    current_mean_stats_df : Pandas.DataFrame
        Modified dataframe with t-test values and results appended (output from get_t_test_results).

    current_replicate_stats_df : Pandas.DataFrame
        Modified dataframe from prep_replicate_data_for_stats.

    af_denominator : int
        The denominator for amp factor calculation.

    Returns
    -------
    current_mean_stats_df : Pandas.DataFrame
        The mean_stats_df dataframe with the amp factor column appended.

    current_replicates_stats_df : Pandas.DataFrame
        The replicates_stats_df dataframe with the amp factor column appended.

    Notes
    -----
    For each concentration, compute the amplification factor AFconc = Cj/10.

    Defaults to book path but will take batch path if 'batch' is passed.

    Might need to do some further work with custom af_denominators
    '''

    amp_factor_denominator = af_denominator
    amp_factor = np.array(current_mean_stats_df.index.get_level_values(0))/amp_factor_denominator
    current_mean_stats_df['amp_factor'] = amp_factor

    # Calculate the amplification factor for the data_for_stats ungrouped table as well (i.e. per replicate data table)
    current_replicate_stats_df['amp_factor'] = np.array(current_replicate_stats_df[['concentration']])/amp_factor_denominator

    # return the mean data table with the amp_factor added
    return current_mean_stats_df, current_replicate_stats_df

def drop_bad_peaks(current_df_mean, current_df_replicates, current_df_title, output_directory):
    '''This function identifies whether proton peaks pass or fail an acceptance criterion to allow
    them to be further analyzed. If the peaks fail, they are dropped from further analysis.

    Further consideration: If more than two proton peak datapoints are flagged as not significant in the mean dataframe
    WITHIN a given concentration, the associated proton peak is removed from further analysis.

    Parameters
    ----------
    current_df_mean : Pandas.DataFrame
        Modifed dataframe from compute_amplification_factor.

    current_df_replicates : Pandas.DataFrame
        Modified dataframe from compute_amplification_factor.

    current_df_title : str
        Title of the dataframe of interest.

    output_directory : str
        File path to the output directory where the text file containing the dropped points is saved.

    batch_or_book : str, {'book', 'batch'}
        Default is book, but it will run the batch path if 'batch' is passed to function.

    Returns
    -------
    current_df_mean_passed : Pandas.DataFrame
        The current_df_mean dataframe with the insignificant proton peaks removed.

    current_df_replicates_passed : Pandas.DataFrame
        The current_df_replicates dataframe with the insignificant proton peaks removed.

    Notes
    -----
    Function also saves a text file of which points have been dropped.
    '''

    #initialize a df that will keep data from the current mean df that meet the criterion above
    significant_corr_attenuation = current_df_mean

    #The below code checks data for the criterion and then removes rows if they do not pass for the mean df --------

    # Assign unique protons to a list to use for subsetting the df via multi-index and the multi-index slicing method
    unique_protons = current_df_replicates.proton_peak_index.unique()
    unique_concentrations = current_df_replicates.concentration.unique().tolist()

    #initialize list to contain the points to remove
    pts_to_remove = []

    for p in unique_protons:
        for c in unique_concentrations:

                #subset the df via index slice based on the current peak and concentration, ppm is part of index here
            current_subset_df = significant_corr_attenuation.loc[idx[c, :, p, :]]


            #subset further for where significance is false
            subset_insignificant = current_subset_df.loc[(current_subset_df['significance'] == False)]

            #if there's more than 2 datapoints where significance is False within the subset, drop p's proton peaks for c's concentration from the significant_corr_attenuation df

            if len(subset_insignificant) > 2:

                try:
                    # recreate the corresponding parent multi index based on the identified points to drop to feed to parent dataframe
                    index_to_drop_sat_time = np.array(current_subset_df.index.get_level_values(1))
                    index_to_drop_ppm = np.array(current_subset_df.index.get_level_values(3))
                    index_to_drop_conc = np.full(len(index_to_drop_sat_time), c)
                    index_to_drop_proton_peak = np.full(len(index_to_drop_sat_time), p)
                except IndexError:
                    # recreate the corresponding parent multi index based on the identified points to drop to feed to parent dataframe
                    index_to_drop_sat_time = np.array(current_subset_df.index.get_level_values(0))
                    index_to_drop_ppm = np.array(current_subset_df.index.get_level_values(1))
                    index_to_drop_conc = np.full(len(index_to_drop_sat_time), c)
                    index_to_drop_proton_peak = np.full(len(index_to_drop_sat_time), p)


                array_indexes_to_drop = [index_to_drop_conc, index_to_drop_sat_time, index_to_drop_proton_peak, index_to_drop_ppm]


                multi_index_to_drop_input = list(zip(*array_indexes_to_drop))
                multi_index_to_drop = pd.MultiIndex.from_tuples(multi_index_to_drop_input, names=['concentration', 'sat_time', 'proton_peak_index', 'ppm'])

                #append multi index to pts to remove
                pts_to_remove.append(multi_index_to_drop)
                print("points being dropped are:", multi_index_to_drop)

                # pass the multi index to drop to drop points from the parent dataframe
                #Problematic
                significant_corr_attenuation = significant_corr_attenuation.drop(multi_index_to_drop, inplace = False)

    print('Removed insignificant points have been printed to the output folder for {}.'.format(current_df_title))

    #create and print dropped points to a summary file
    dropped_points_file = open("{}/dropped_insignificant_points.txt".format(output_directory), "w")
    dropped_points_file.write("The datapoints dropped from consideration due to not meeting the criteria for significance are: \n{}".format(pts_to_remove))
    dropped_points_file.close()

    # define a mean df of the data that passed to return
    current_df_mean_passed = significant_corr_attenuation

    # The below code removes the same bad points from the replicate df ------------

    # Reset index and drop the old index column, just getting the dataframe ready for this, not sure why this works
    current_df_replicates = current_df_replicates.reset_index()
    current_df_replicates = current_df_replicates.drop('index', axis = 1)

    #drop the points
    for num_pts in pts_to_remove:
        for exp_parameters in num_pts:

            drop_subset = (current_df_replicates.loc[(current_df_replicates['concentration'] == exp_parameters[0]) & (current_df_replicates['sat_time'] == exp_parameters[1]) & (current_df_replicates['proton_peak_index'] == exp_parameters[2])])
            current_df_replicates = current_df_replicates.drop(drop_subset.index)

    #define a replicate df of the data that passed to return (reset index might not actually be needed? but I know this way works...)
    current_df_replicates_passed = current_df_replicates.reset_index()

    return current_df_mean_passed, current_df_replicates_passed

def execute_curvefit(stats_df_mean, stats_df_replicates, output_directory2, output_directory3, current_df_title):
    ''' We are now ready to calculate the nonlinear curve fit models (or "hat" models),
    for both individual replicate data (via stats_df_replicates), and on a mean (or "bar") basis (via stats_df_mean).

    This function carries out the curve fitting process for the current dataframe.
    It calculates the nonlinear curve fit, then the SSE and AFo on a mean and replicate basis using only significant points.

    Parameters
    ----------
    stats_df_mean, stats_df_replicates : Pandas.DataFrame
        Fully pre-processed dataframes.

    output_directory2 : str
        File path to the output directory where the figures are saved.

    output_directory3 : str
        File path to the output directory where the final Excel files are saved.

    current_df_title : str
        Title of current dataframe.

    batch_or_book : str, {'book', 'batch'}
        Default is book, but it will run the batch path if 'batch' is passed to function.

    Returns
    -------
    stats_df_mean : Pandas.DataFrame
        The input stats_df_mean dataframe with results from curve fitting appended to table.

    stats_df_replicates : Pandas.DataFrame
        The input stats_df_replicates dataframe with results from curve fitting appended to table.

    Notes
    -----
    Figures are displayed and saved as a file in custom directory. Dataframes are also saved to Excel files with final results.

    Mean, average, and bar are used equivalently in this part of the code.

    ikj sub-scripts are used in this script to keep track of the fixed experimental variables pertinent to this analysis.
    For clarity:
    i = NMR saturation time (sat_time) column
    k = Sample proton peak index (proton_peak_index) column
    j = Sample concentration (concentration) column

    yikj = response model (Amp Factor x Corr % Attenuation) on a per replicate basis (not mean), fits with stats_df_replicates
    yikj_bar = response model (Amp Factor x Corr % Attenuation) on an average basis, fits with stats_df_mean
    yikj_hat = the fitted nonlinear model according to the levenburg marquadt minimization of least squares algorithm
    '''

    # first assign yijk to the replicate dataframe
    stats_df_replicates['yikj'] = stats_df_replicates[['corr_%_attenuation']].values*(stats_df_replicates[['amp_factor']].values)

    # then assign yijk_bar to the mean dataframe
    stats_df_mean['yikj_bar'] = (stats_df_mean['corr_%_attenuation']['mean'])*(stats_df_mean['amp_factor'])

    # Assign unique protons to a list to use for subsetting the df via the shortcut multi-index index slicing method
    unique_protons = stats_df_replicates.proton_peak_index.unique().tolist()
    unique_concentrations = stats_df_replicates.concentration.unique().tolist()
    unique_replicates = stats_df_replicates.replicate.unique().tolist()
    ppm_index = stats_df_replicates['ppm']

    # Now preparing to curve fit, export the curve fit plots to a file, and tabulate the final results ------------------------------

    print('Exporting all mean and individual curve fit figures to an output directory... this may take a moment.')

    for c in unique_concentrations:
        for p in unique_protons:

            # COMPLETE MEAN CURVE FITTING OPERATIONS PER PROTON & PER CONCENTRATION

            # subset the df into the data for one graph, via index slice based on the current peak and concentration
            #one_graph_data_mean = stats_df_mean.loc[(slice(c), slice(None), slice(p)), :] # --> slice(c) grabs all the values up to and including c

            if p not in stats_df_mean.loc[c].index.get_level_values(1).unique(): # proton peak not significant for this concentration
                continue

            one_graph_data_mean = stats_df_mean.loc[(c, slice(None), p), :]

            #Make a boolean significance mask based on the one graph subset, for calculating parameters based on only significant pts
            boolean_sig_mask = one_graph_data_mean.significance == True

            #assign ALL datapoints and ALL data times to test_data variables for this ONE GRAPH
            all_yikj_bar = one_graph_data_mean['yikj_bar']

            all_sat_time = np.asarray(all_yikj_bar.index.get_level_values(1))

            #apply boolean significance mask to create the data to be used for actual curve fitting/parameter generation
            significant_yikj_bar = all_yikj_bar[boolean_sig_mask]
            significant_sat_time = all_sat_time[boolean_sig_mask]

            # grab the current mean ppm for this graph to use in naming and plotting
            ppm_bar = np.round(one_graph_data_mean.index.get_level_values(3)[0].astype(float),4)

            # this will skip the graphing and analysis for cases where an insignificant proton peak has been removed from consideration PREVIOUSLY due to cutoff
            if all_yikj_bar.size == 0:
                continue

            # initial guess for alpha and beta (applies equally to replicate operations below)
            initial_guess = np.asarray([1, 1])

            # Generate best alpha & beta parameters for data based on only significant pts, for the current proton and concentration, optimizing for minimization of square error via least squares levenburg marquadt algorithm
            best_param_vals_bar, covar_bar = curve_fit(y_hat_fit, significant_sat_time, significant_yikj_bar, p0 = initial_guess, method = 'lm', maxfev=5000)

            # calculate ultimate sum of square errors after minimization for each time point
            sse_bar = np.square(y_hat_fit(significant_sat_time, *best_param_vals_bar) - significant_yikj_bar)

            #append sum of square error calculated for this graph to the PARENT mean dataframe at this c and p
            #stats_df_mean.loc[(slice(c), slice(None), slice(p)), ('SSE_bar')] = sse_bar.sum()

            stats_df_mean.loc[(c, slice(None), p), ('SSE_bar')] = sse_bar.sum()

            # append best parameters to variables, and then generate the instantaneous amplification factor
            a_kj_bar = best_param_vals_bar[0]
            b_kj_bar = best_param_vals_bar[1]

            amp_factor_instantaneous_bar = a_kj_bar * b_kj_bar

            #append instantaneous amplification factor calculated to the PARENT mean dataframe, for all datapoints in this graph
            #stats_df_mean.loc[(slice(c), slice(None), slice(p)), ('AFo_bar')] = [amp_factor_instantaneous_bar]*(len(all_yikj_bar))
            stats_df_mean.loc[(c, slice(None), p), ('alpha_bar')] = a_kj_bar
            stats_df_mean.loc[(c, slice(None), p), ('beta_bar')] = b_kj_bar
            stats_df_mean.loc[(c, slice(None), p), ('AFo_bar')] = [amp_factor_instantaneous_bar]*(len(all_yikj_bar))

            # define file name for curve fits by mean
            output_file_name_figsmean = "{}/mean_conc{}_ppm{}.png".format(output_directory2, c, ppm_bar)

            generate_curvefit_plot(all_sat_time, one_graph_data_mean, best_param_vals_bar, ppm_bar, output_file_name_figsmean, c, mean_or_rep = 'mean')

            for r in unique_replicates:

                #COMPLETE REPLICATE SPECIFIC CURVE FIT OPERATIONS - subset the df via index slice based on the current peak, concentration, and replicate
                one_graph_data = stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r)]

                # define the experimental data to compare square error with (amp_factor * atten_corr_int), for y_ikj
                y_ikj = one_graph_data['yikj']

                #this will skip the graphing and analysis for cases where a proton peak has been removed from consideration
                if y_ikj.size == 0:
                    continue

                # define sat_time to be used for the x_data
                sat_time = one_graph_data[['sat_time']].values.ravel()

                #Fit Curve for curren proton, concentration and replicate, optimizing for minimization of square error via least squares levenburg marquadt algorithm
                best_param_vals, covar = curve_fit(y_hat_fit, sat_time, y_ikj, p0 = initial_guess, method = 'lm', maxfev=5000)

                #calculate ultimate sum of square errors after minimization for each time point, and append to list
                sse = np.square(y_hat_fit(sat_time, *best_param_vals) - y_ikj)

                #appends to PARENT stats replicate dataframe
                stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r), ('SSE')] = sse.sum()

                # solve for the instantaneous amplification factor
                a_kj = best_param_vals[0]
                b_kj = best_param_vals[1]
                amp_factor_instantaneous = a_kj * b_kj

                #appends instantaneous amplification factor and params calculated to the PARENT stats replicate dataframe, for all datapoints in this graph
                stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r), ('alpha')] = a_kj
                stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r), ('beta')] = b_kj
                stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r), ('AFo')] = [amp_factor_instantaneous]*(len(y_ikj))

                #determine mean current ppm across the sat_times for this replicate so that we can add it to the file name


                mean_current_ppm = one_graph_data.loc[(one_graph_data['concentration'] == c) & (one_graph_data['proton_peak_index'] == p) & (one_graph_data['replicate'] == r)]['ppm'].values[0].astype(float).round(4)

                # file name for curve fits by replicate
                output_file_name_figsrep = "{}/replicate{}_conc{}_ppm{}.png".format(output_directory2, r, c, mean_current_ppm)

                generate_curvefit_plot(sat_time, y_ikj, best_param_vals, mean_current_ppm, output_file_name_figsrep, c, r, mean_or_rep = 'rep')

    #export tabulated results to file and return updated dataframes
    output_file_name = "stats_analysis_output_replicate_{}.xlsx".format(current_df_title)

    #export replicates final results table to a summary file in Excel
    stats_df_replicates.to_excel(os.path.join(output_directory3, output_file_name))

    #export mean final results table to a summary file in Excel
    #if there are replicates, and mean data was created, export the final mean data to excel as well
    if stats_df_mean.shape[0] != 0:

        output_file_name = "stats_analysis_output_mean_{}.xlsx".format(current_df_title)
        stats_df_mean.to_excel(os.path.join(output_directory3, output_file_name))

    print('Export of all figures to file complete!')
    return stats_df_mean, stats_df_replicates
