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
        True if [Cc]omplete is a substring, otherwise False.
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

# helper for add_attenuation_and_corr_attenuation_to_dataframe
def attenuation_calc_equality_checker(df1, df2, batch_or_book = 'book'):
    '''This functions checks to see if two subset dataframes for the attenuation calculation are equal and in the same order 
    in terms of their fixed experimental parameters: 'sample_or_control', 'replicate', 'title_string', 'concentration', and
    'sat_time' 
    
    Parameters
    ----------
    df1, df2 : Pandas.DataFrame
        DataFrames involved in the attenuation calculation, one where irrad bool is true and one where irrad bool is false.
    
    batch_or_book : str, {'book', 'batch'}
        String indicating the DataFrame's format. The default runs the 'book' path, but will run the 'batch' path 
        if indicated in the third argument.
    
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
        
        if batch_or_book == 'book':
            subset_compare_1 = df1[['sample_or_control', 'replicate', 'title_string', 'concentration', 'sat_time']]
            subset_compare_2 = df2[['sample_or_control', 'replicate', 'title_string', 'concentration', 'sat_time']]
        
            exactly_equal = subset_compare_2.equals(subset_compare_1)
    
            return exactly_equal
    
        else:
            
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
            
            #if passes all 4 checks return true, if not false
            if exactly_equal_1 == exactly_equal_2 == exactly_equal_3 == exactly_equal_4 == True:
                return True
            
            else: return False
        
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
            
        else: return False
        
        
    else:
        raise ValueError("Error, corrected % attenuation input dataframes are not the same shape to begin with.")

# helper for prep_mean_data_for_stats
def get_dofs(peak_indices_array):
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
    
    Returns
    -------
    dof_list : list
        List containing the DOFs for peak_indices_array.
    '''

    dof_list = []
    dof_count = 0
    global_count = 0

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

    return dof_list

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
    
    Notes
    -----
    
    Some useful scipy example references for curve fitting:
    1) https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    2) https://lmfit.github.io/lmfit-py/model.html
    3) https://astrofrog.github.io/py4sci/_static/15.%20Fitting%20models%20to%20data.html 
    '''
    return a * (1 - np.exp(t * -b))
