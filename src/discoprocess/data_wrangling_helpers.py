import pandas as pd
import numpy as np
import os

# helper for initialize_excel_batch_replicates
def DropComplete(x):
    """Checks if [Cc]omplete is a substring of x.

    Parameters
    ----------
    x : str

    Returns
    -------
    bool
        False if [Cc]omplete is a substring, otherwise True.

    Examples
    --------
    >>> DropComplete("Complete")
    False

    >>> DropComplete("complete")
    False

    >>> DropComplete("Other")
    True
    """

    if "omplete" in x:
        return False
    else:
        return True

# helper for batch_to_dataframe
def initialize_excel_batch_replicates(b):
    '''This function determines the unique polymers contained in an excel book, the number of replicates of those polymers,
    and also returns an iterable of the sheet names without Complete in them.

    Parameters
    ----------
    b : str
        Path file to the Excel Batch file of interest.

    Returns
    -------
    unique_polymers : list
        List of unique polymer names contained in b.

    unique_polymer_replicates : list
        List of number of replicates for each unique polymer.

    all_sheets_complete_removed : list
        List of sheet names in b without Complete in them.
    '''
    all_sheets_iterator = []
    all_data_sheets = []
    name_sheets = []
    unique_polymers = []
    unique_polymer_replicates = []
    all_sheets_complete_removed = []
    intermediate_string = []
    match_checker = []

    # load excel book into Pandas
    current_book_title = os.path.basename(str(b))

    # determine the number and names of sheets in the book
    name_sheets = pd.ExcelFile(b).sheet_names

    # remove sheets with complete from the list
    all_sheets_iterable = [sheet for sheet in name_sheets]

    all_sheets_complete_removed = list(filter(DropComplete, all_sheets_iterable))

    # sort all sheets with complete removed alphebetically for generalizable replicate ordering and processing
    all_sheets_complete_removed = sorted(all_sheets_complete_removed)

    # generate a list of unique polymers in the book
    for sheet in range(len(all_sheets_complete_removed)):
        # drop string after brackets
        intermediate_string = all_sheets_complete_removed[sheet].split('(', 1)[0]

        #if there's a trailing space after dropping the bracket, remove it as well
        if intermediate_string[-1] == ' ':
            intermediate_string = intermediate_string[0:-1]

        unique_polymers.append(intermediate_string)

    # drop duplicates to generate a unique polymers list
    unique_polymers = list(dict.fromkeys(unique_polymers))

    # initialize zero array that corresponds to each unique_polymer
    unique_polymer_replicates = np.zeros(len(unique_polymers))

    # calculate the number of replicates of the polymers in the book by matching unique polymers to sheet names
    for i in range(len(unique_polymers)):
        for j in range(len(all_sheets_complete_removed)):

            #calculate the current match checker the same way unique polymers were calculated
            match_checker = all_sheets_complete_removed[j].split('(', 1)[0]
            if match_checker[-1] == ' ':
                match_checker = match_checker[0:-1]

            #if unique polymer name matches the checker, increment replicate counter for that polymer
            if (unique_polymers[i] == match_checker):
                unique_polymer_replicates[i] +=1

    return unique_polymers, unique_polymer_replicates, all_sheets_complete_removed


def clean_string(string):
    '''Standard string cleaning operations, make lower case and remove spaces/unwanted chars.'''

    string = string.lower()
    string = string.replace(" ", "")
    string = string.replace("(", "")
    string = string.replace(")", "")

    return string

def grab_conc(polymer_name):
    '''Extracts concentration from the polymer name string.

    Parameters:
    ----------
    polymer_name: string
        string that includes concentration as part of data entry

    Returns:
    --------
    conc: int
        int containing the substring of conentration (uM) of the experiment

    Notes:
    -----
    In polymer name variable for data entry, need to ensure the concentration followed by uM is always the terminal substring.
    '''

    polymer_name = polymer_name.split(' ', -1)[0] # remove the (#) replicate number from polymer name
    polymer_name = clean_string(polymer_name)
    substrings = polymer_name.split('_', -1) # grab the conc substring
    conc = int(substrings[-1][:-2]) # without the uM chars

    return conc

# helper for batch_to_dataframe
def wrangle_batch(b, name_sheets, replicate_index):
    """This functions takes in a list of Excel sheets and translates it into a usable Pandas.DataFrame.

    Parameters
    ----------
    b : str
        Path file to the Excel Batch file of interest.

    name_sheets : list
        List of sheet names in b without Complete in them.

    replicate_index : list
        List of integers representing the replicate index of the Excel sheets in name_sheets.

    Returns
    -------
    list_of_clean_dfs : list
        List of tuples, where each tuple contains ('polymer_name', CleanPolymerDataFrame)
        Tuples are in a "key-value pair format", where the key (at index 0 of the tuple) is:
        current_book_title, a string containing the title of the current excel input book
        And the value (at index 1 of the tuple) is:
        clean_df, the cleaned pandas dataframe corresponding to that book title!
    """
    # initialize an empty list and dataframe to contain the mini experimental dataframes collected from one polymer, which will be ultimately appended to the global list_of_clean_dfs as a tuple with the polymer name
    current_polymer_df_list = []
    list_of_clean_dfs = []
    current_polymer_df = pd.DataFrame([],[])

    # loop once through every sheet in the book containing raw data, and execute actions along the way
    for i in range(len(name_sheets)):

        #assign current sheet name to current sheet
        current_sheet = name_sheets[i]

        # read in the current book's current sheet into a Pandas dataframe
        current_sheet_df = pd.read_excel(b, sheet_name = current_sheet)

        #initialize nth_replicate to the correct replicate index
        nth_replicate = replicate_index[i]

        # if it's the first replicate of a polymer AND it's not the first datasheet being assessed, append old polymer info to df and reset
        if (nth_replicate == 1 and (i != 0)):

            # concatenate the current_polymer_df_list into the current polymer_df
            current_polymer_df = pd.concat(current_polymer_df_list)

            # append the current polymer df to the global list_of_clean_dfs as a tuple with the polymer name
            list_of_clean_dfs.append((current_polymer_name, current_polymer_df))

            # reset the current_polymer_df_list to empty so that the next polymer can be appended to it
            current_polymer_df_list = []

        print("Reading in Data From Sheet: ", current_sheet)

        # update current polymer name, if it's the first replicate
        if nth_replicate == 1:
            current_polymer_name = current_sheet

        # Now that we know it's not a Complete sheet, and have ensured values have been reset as required, enter current sheet

        # use np.where to get sheet sub-table coordinates, and infer the table bounds from its surroundings
        sub_table_indicator = 'Range'

        # assigns coordinates of all upper left 'Range' cells to an index (row) array and a column numerical index
        table_indices, table_columns = np.where(current_sheet_df == sub_table_indicator)

        # determine the number of experimental rows in each NMR results sub-table
        # minus 1 to exclude Range cell row itself
        num_experimental_indices = np.unique(table_indices)[2] - np.unique(table_indices)[1] - 1

        # minus 2 to account for an empty row and the indices of the next column, as Range is the defining word
        num_experimental_cols = np.unique(table_columns)[1] - np.unique(table_columns)[0] - 2

        # initialize/reset current_exp_df, current is for CURRENT sub-tables being looped over
        current_exp_df = []

        # make a list of coordinate pair tuples for this sheet using list comprehension
        sheet_coords_list = [(table_indices[i], table_columns[i]) for i in range(len(table_indices))]

        #print("\n\n\n", sheet_coords_list, "\n\n\n")

        # iterate through coordinates
        for coords in sheet_coords_list:

            #makes coords mutable as a numpy array unlike tuples
            coords_copy = np.asarray([coords[0], coords[1]])

            # assign current values to the fixed experimental parameters for this experimental sub-table relative to Range...

            # sat time is one cell up and one cell to the left of Range, chain indexed at second col in the row generated by loc
            current_sat_time = current_sheet_df.iloc[(coords_copy[0]-1), (coords_copy[1]-1)]
            # irrad_bool is one cell above Range in same column, chain indexed at third col in the row generated by loc
            current_irrad_bool = current_sheet_df.iloc[(coords_copy[0]-1), (coords_copy[1])]

            # Grab concentration from polymer name string
            current_concentration = grab_conc(current_polymer_name)
            # sample or control is one cell above Range in one column to the right, chain indexed at fourth col in the row generated by loc
            current_sample_or_control = current_sheet_df.iloc[(coords_copy[0]-1), (coords_copy[1]+1)]
            current_replicate = nth_replicate

            # initialize/reset experimental lists to null for this experimental set
            list_current_subtable_experimental = []

            # now need to find and assign the sub-table's actual experimental data to the new lists

            # Initializing the indexing "boundaries" of the sub-table rectangle containing values of interest in a generalizable way, based on white cell left of Range
            experimental_index_starter = coords_copy[0]+1
            experimental_index_range = range(num_experimental_indices)
            experimental_column_range = range(num_experimental_cols)
            combined_experimental_index_range = experimental_index_starter + experimental_index_range
            experimental_column_starter = coords_copy[1]
            combined_experimental_column_range = experimental_column_starter + experimental_column_range

            # for ease of reading, designating index start and end variables
            index_slice_start = experimental_index_starter
            index_slice_end = experimental_index_starter + num_experimental_indices -1
            column_slice_start = experimental_column_starter -1
            column_slice_end = experimental_column_starter + num_experimental_cols

            #make new mini dataframe of the current experimetnal subtable values
            current_subtable_experimental = pd.DataFrame(current_sheet_df.iloc[index_slice_start:index_slice_end,column_slice_start:column_slice_end])

            # define length of the experimental parameter lists (number of experimental rows per subtable)
            exp_list_length = index_slice_end - index_slice_start

            # create "ranged" lists of the constant experimental values to make them the same length as the unique variable experimental values, so we can add information "per observation" to the dataframe
            ranged_sample_or_control = exp_list_length * [current_sample_or_control]
            ranged_replicate = exp_list_length * [current_replicate]
            ranged_polymer_name = exp_list_length * [current_polymer_name]
            ranged_concentration = exp_list_length * [current_concentration]
            ranged_sat_time = exp_list_length * [current_sat_time]
            ranged_irrad_bool = exp_list_length * [current_irrad_bool]


            # assign all current sub-table experimental values for this experimental sub-table to a dataframe via a dictionary
            current_dict = {        "polymer_name":ranged_polymer_name,
                                    "concentration":ranged_concentration,
                                    "proton_peak_index":current_subtable_experimental.iloc[:,0],
                                    "ppm_range":current_subtable_experimental.iloc[:,1],
                                    "absolute":current_subtable_experimental.iloc[:,3],
                                    "sample_or_control":ranged_sample_or_control,
                                    "replicate":ranged_replicate,
                                    "sat_time":ranged_sat_time,
                                    "irrad_bool":ranged_irrad_bool,
                                    }


            # this is now a dataframe of a polymer sub-table
            current_exp_df = pd.DataFrame(current_dict)

            # append the dataframe of the polymer sub-table to the global polymer-level list of dataframes
            current_polymer_df_list.append(current_exp_df)

        # after we have looped through all the coordinates of one sheet, before going to the next sheet, assign the last_sheet polymer variable to this sheet's name
        last_polymer_sheet = current_sheet

        # if it's the last sheet, append final polymer info to dfs
        if current_sheet == name_sheets[-1]:

            # concatenate the current_polymer_df_list into the current polymer_df
            current_polymer_df = pd.concat(current_polymer_df_list)

            # append the current polymer df to the global list_of_clean_dfs as a tuple with the polymer name
            list_of_clean_dfs.append((current_polymer_name, current_polymer_df))

            # reset the current_polymer_df_list to empty so that the next polymer can be appended to it
            current_polymer_df_list = []

    return list_of_clean_dfs

# helper for book_to_dataframe
def count_sheets(name_sheets):
    """This function counts the number of sample and control datasheets from which data will be extracted for further processing.

    Parameters
    ----------
    name_sheets : list
        List of sheet names.

    Returns
    -------
    num_samples : int
        Number of sheets containing sample data.

    num_controls : int
        Number of sheets containing control data.

    sample_control_initializer : list
        List of Excel sheets containing all sample and control data.

    sample_replicate_initializer : list
        List of integers containing the replicate number of each sample sheet in sample_control_initializer.

    control_replicate_initializer : list
        List of integers containing the replicate number of each control sheet in sample_control_initializer.
    """
    # initialize number of samples and controls to zero, then initialize the "list initializers" which will hold book-level data to eventually add to the book-level dataframe.
    num_samples = 0
    num_controls = 0

    sample_control_initializer = []
    sample_replicate_initializer = []
    control_replicate_initializer = []

    # loop through sheet names to determine number of samples and controls in this book
    for s in range(len(name_sheets)):

        # if the current sheet is labeled Sample:
        # increment sample counter, add a list item called 'sample' to be initialized into the 'sample_or_control' list, add a list item of the replicate number to be initialized into the 'replicate' list.
        if ('Sample' or 'sample') in name_sheets[s]:
            num_samples += 1
            sample_control_initializer.append('sample')
            sample_replicate_initializer.append(num_samples)

        # if the current sheet is labeled Control:
        # increment control counter, add a list item called 'control' to be initialized into the 'sample_or_control' list, add a list item of the replicate number to be initialized into the 'replicate' list.
        elif ('Control' or 'control') in name_sheets[s]:
            num_controls += 1
            sample_control_initializer.append('control')
            control_replicate_initializer.append(num_controls)

    return num_samples, num_controls, sample_control_initializer, sample_replicate_initializer, control_replicate_initializer

# helper for add_attenuation_and_corr_attenuation_to_dataframe
def attenuation_calc_equality_checker(df1, df2):
    '''This functions checks to see if two subset dataframes for the attenuation calculation are equal and in the same order
    in terms of their fixed experimental parameters: 'sample_or_control', 'replicate', 'title_string', 'concentration', and
    'sat_time'.

    Parameters
    ----------
    df1, df2 : Pandas.DataFrame
        DataFrames involved in the attenuation calculation, one where irrad bool is true and one where irrad bool is false.

    Returns
    -------
    bool
        Returns True if the subset dataframes are equal, otherwise False.

    Raises
    ------
    ValueError
        If the passed dataframes do not have the same shape.
    '''

    if (df1.shape == df2.shape):


        #check sample_or_control is the same
        subset_compare_1 = df1.sample_or_control.values
        subset_compare_2 = df2.sample_or_control.values
        exactly_equal_1 = np.array_equal(subset_compare_1, subset_compare_2)

        #check replicate is the same
        subset_compare_1 = df1.replicate.values
        subset_compare_2 = df2.replicate.values
        exactly_equal_2 = np.array_equal(subset_compare_1, subset_compare_2)
        #             exactly_equal_2 = subset_compare_2.equals(subset_compare_1)

        #check proton peak index is the same
        subset_compare_1 = df1.proton_peak_index.values
        subset_compare_2 = df2.proton_peak_index.values
        exactly_equal_3 = np.array_equal(subset_compare_1, subset_compare_2)

        #check sat time is the same
        subset_compare_1 = df1.sat_time.values
        subset_compare_2 = df2.sat_time.values
        exactly_equal_4 = np.array_equal(subset_compare_1, subset_compare_2)

        # if passes all 4 checks return true, if not false
        if exactly_equal_1 == exactly_equal_2 == exactly_equal_3 == exactly_equal_4 == True:
            return True

        else:
            return False

    else:
        raise ValueError("Error, irrad_false and irrad_true dataframes are not the same shape to begin with.")

# helper for add_attenuation_and_corr_attenuation_to_dataframe
def corrected_attenuation_calc_equality_checker(df1, df2, df3):
    '''This functions checks to see if the three subset dataframes for calculating the corrected % attenuation
    are equal and in the same order in terms of their shared fixed experimental parameters: 'replicate', 'concentration', and
    'sat_time'

    Parameters
    ----------
    df1, df2, df3 : Pandas.DataFrame
        DataFrames involved in the corrected attenuation calculation.

    Returns
    -------
    bool
        Returns True if the subset dataframes are equal, otherwise False.

    Raises
    ------
    ValueError
        If the passed dataframes do not have the same shape.
    '''
    #check if number of rows same in each df, number of columns not same as samples dfs contain attenuation data
    if (df1.shape[0] == df2.shape[0] == df3.shape[0]):

        #check replicate is the same
        subset_compare_1 = df1.replicate.values
        subset_compare_2 = df2.replicate.values
        subset_compare_3 = df3.replicate.values

        exactly_equal_1 = np.logical_and((subset_compare_1==subset_compare_2).all(), (subset_compare_2==subset_compare_3).all())

        #check sat time is the same
        subset_compare_1 = df1.sat_time.values
        subset_compare_2 = df2.sat_time.values
        subset_compare_3 = df3.sat_time.values

        exactly_equal_2 = np.logical_and((subset_compare_1==subset_compare_2).all(), (subset_compare_2==subset_compare_3).all())

        #check concentration is the same
        subset_compare_1 = df1.concentration.values
        subset_compare_2 = df2.concentration.values
        subset_compare_3 = df3.concentration.values

        exactly_equal_3 = np.logical_and((subset_compare_1==subset_compare_2).all(), (subset_compare_2==subset_compare_3).all())

        #if passes all 3 checks return true, if not false
        if exactly_equal_1 == exactly_equal_2 == exactly_equal_3 == True:
            return True

        else:
            return False

    else:
        raise ValueError("Error, corrected % attenuation input dataframes are not the same shape to begin with.")

# helper for prep_mean_data_for_stats
def get_dofs(peak_indices_array, df = pd.DataFrame([1, 2, 3])):
    ''' This function calculates the number of degrees of freedom (i.e. number of experimental replicates minus one) for statistical calculations
    using the "indices array" of a given experimenal set as input.

    Input should be in the format: (in format: 11111 22222 3333 ... (specifically, it should be the proton_peak_index column from the "regrouped_df" above)
    where the count of each repeated digit minus one represents the degrees of freedom for that peak (i.e. the number of replicates -1).
    With zero based indexing, the function below generates the DOFs for the input array of proton_peak_index directly, in a format
    that can be directly appended to the stats table.

    Parameters
    ----------
    peak_indices_array : NumPy.array
        NumPy array representing the proton_peak_index column from the "regrouped_df" from prep_mean_data_for_stats.

    df: Pandas.DataFrame
        contains the mean data, regrouped per concentration, sat time, proton peak ix, replicate, ppm

    Returns
    -------
    dof_list : list
        List containing the DOFs for peak_indices_array.
    '''

    dof_list = []
    dof_count = 0
    global_count = 0

    # if there is more than one unique peak
    if len(np.unique(peak_indices_array)) > 1 :

        # loop through range of the peak indices array
        for i in range(len(peak_indices_array)):
            global_count = global_count +1

            # if at the end of the global range, append final incremented dof count
            if global_count == len(peak_indices_array):
                dof_count = dof_count+1
                dof_list.append(dof_count)
                break

            # if current index is not equal to the value of the array at the next index, apply count of current index to DOF list, and reset DOF counter
            elif peak_indices_array[i] != peak_indices_array[i+1]:
                dof_list.append(dof_count)
                dof_count = 0

            # otherwise, increment DOF count and continue
            else:
                dof_count = dof_count + 1

        # subtract one from final value to account for an extra +1 from looping
        dof_list[-1] = dof_list[-1] - 1

    # otherwise, calc dofs another way
    else:
        dof_list = get_dofs_one_peak(df)

    # print(dof_list)
    return dof_list

def get_dofs_one_peak(df):
    '''Calculates the degrees of freedom for the edge case where there is only one peak.

    Paramters:
    ----------
    df: Pandas.DataFrame
        contains the mean data, regrouped per concentration, sat time, proton peak ix, replicate, ppm
    Returns:
    --------
    dof_list: list
        list to append to the dataframe with the degrees of freedom
    '''

    replicates = df.index.get_level_values(3)

    # use max num of replicates as a reference to identify degrees of freedom
    dof = np.max(replicates)-1

    # make size of the list correspond to be appended to dataframe
    dof_list = [dof] * len(np.unique(df.index.get_level_values(1)))

    return dof_list

#Function is never actually used
'''
def flatten_multicolumns(mean_df):
'''
'''Takes in a mean df and flattens the multi-tier column
    index into a directly indexable index.'''
'''
    # clean up multi index for both
    colnames = mean_df.columns.get_level_values(0).values
    mean_df = mean_df.droplevel(1, axis=1)
    colnames[4] = "corr_%_attenuation_mean"
    colnames[5] = "corr_%_attenuation_std"
    mean_df.columns = colnames

    return mean_df

'''
# helper for execute_curvefit
def y_hat_fit(t, a, b):
    '''This function returns y_ikj_hat as the fit model based on alpha and beta.

    Parameters
    ----------
    t : NumPy.array
        NumPy array representing all the variable saturation times for each unique proton.

    a : float
        Least-squares curve fitting parameter, represents STDmax. the asymptomic maximum of the build-up curve.

    b : float
        Least-squares curve fitting parameter, represents -k_sat, the negative of the rate constant related to the relaxation
        properties of a given proton.

    Returns
    -------
    NumPy.array
        NumPy array representing the output values of the fit model based on alpha and beta.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0.25, 0.5, 0.75])
    >>> a = 0.5
    >>> b = 0.5
    >>> y_hat_fit(t, a, b)
    array([0.05875155, 0.11059961, 0.15635536])

    >>> a = 0
    >>> y_hat_fit(t, a, b)
    array([0., 0., 0.])

    Notes
    -----

    Some useful scipy example references for curve fitting:
    1) https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    2) https://lmfit.github.io/lmfit-py/model.html
    3) https://astrofrog.github.io/py4sci/_static/15.%20Fitting%20models%20to%20data.html
    '''
    return a * (1 - np.exp(t * -b))

# running doctest
if __name__ == "__main__":
    import doctest
    doctest.testmod()
