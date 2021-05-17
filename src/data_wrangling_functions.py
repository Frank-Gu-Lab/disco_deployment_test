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

from . import data_wrangling_helpers
initialize_excel_batch_replicates = data_wrangling_helpers.initialize_excel_batch_replicates

# define handy shortcut for indexing a multi-index dataframe
idx = pd.IndexSlice

# functions are ordered below in the order they are called in the disco-data-processing script

def convert_excel_batch_to_dataframe(b):
    '''
    This function converts and cleans excel books of type "Batch" (containing many polymers in one book) into dataframes for further analysis.
    
    Parameters
    ----------
    b : str
        String representing the relative path to an Excel Batch Book.
    
    Returns
    -------
    list_clean_dfs : list
        List of tuples, where each tuple contains ('polymer_name', CleanPolymerDataFrame)
    '''
    # initialize an empty list and dataframe to contain the mini experimental dataframes collected from one polymer, which will be ultimately appended to the global list_of_clean_dfs as a tuple with the polymer name
    current_polymer_df_list = []
    list_of_clean_dfs = []
    current_polymer_df = pd.DataFrame([],[])
    replicate_index = []
    
    # initialize current_polymer_name and last_polymer_sheet to empty strings
    current_polymer_name = ' '
    last_polymer_sheet = ' '
    
    #load excel book into Pandas
    current_book_title = os.path.basename(str(b))
    
    print("The current book being analyzed is: ", current_book_title)
    
    # return list of unique polymer names in the book, their replicates, and the sheets containing raw data to loop through
    unique_polymers, unique_polymer_replicates, name_sheets = initialize_excel_batch_replicates(b)
    
    # generate a new replicate index list that holds the nth replicate associated with each raw data sheet in book
    for i in range(len(unique_polymer_replicates)):
        current_replicate_range = range(1,int(unique_polymer_replicates[i]+1),1)
        for j in current_replicate_range:
            replicate_index.append(j)
    
    # BEGIN WRANGLING DATA FROM THE EXCEL FILE, AND TRANSLATING INTO ORGANIZED DATAFRAME ----------------

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
            
        i, c = np.where(current_sheet_df == sub_table_indicator)

        # assigns coordinates of all upper left 'Range' cells to an index (row) array and a column numerical index
        table_indices = i
        table_columns = c

        # determine the number of experimental rows in each NMR results sub-table
        # minus 1 to exclude Range cell row itself
        num_experimental_indices = np.unique(table_indices)[2] - np.unique(table_indices)[1] - 1

        # minus 2 to account for an empty row and the indices of the next column, as Range is the defining word
        num_experimental_cols = np.unique(table_columns)[1] - np.unique(table_columns)[0] - 2

        # initialize/reset current_exp_df, current is for CURRENT sub-tables being looped over
        current_exp_df = []

        # make a list of coordinate pair tuples for this sheet using list comprehension
        sheet_coords_list = [(table_indices[i], table_columns[i]) for i in range(len(table_indices))]

        # iterate through coordinates
        for coords in sheet_coords_list:
            
            #makes coords mutable as a numpy array unlike tuples
            coords_copy = np.asarray([coords[0], coords[1]])
            
            # assign current values to the fixed experimental parameters for this experimental sub-table relative to Range...
    
            # sat time is one cell up and one cell to the left of Range, chain indexed at second col in the row generated by loc
            current_sat_time = current_sheet_df.iloc[(coords_copy[0]-1), (coords_copy[1]-1)]
            # irrad_bool is one cell above Range in same column, chain indexed at third col in the row generated by loc
            current_irrad_bool = current_sheet_df.iloc[(coords_copy[0]-1), (coords_copy[1])]
            # Hard coding arbitrary concentration as batches were from industry and only know the volume conc (9mg/ml we know)
            current_concentration = 9
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
    
    # After all is said and done, return a list of the clean dfs containing polymer tuples of format (polymer_name, polymer_df)
    print("Returning a list of tuples containing all polymer information from Batch: ", b)
    
    return list_of_clean_dfs

def convert_excel_to_dataframe(b, global_output_directory):
    '''
    This function converts raw Excel books containing outputs from DISCO-NMR experiments into cleaned and 
    organized Pandas DataFrames for further processing.
    
    Inputs are:
    b = The file path to the excel book of interest, obtained generalizably from the "list_of_raw_books" defined above
    
    Output is a tuple (clean_tuple) in a "key-value pair format,"" where the key (at index 0 of the tuple) is:
        current_book_title, a string containing the title of the current excel input book
   
    And the value (at index 1 of the tuple) is:
        clean_df, the cleaned pandas dataframe corresponding to that book title!

    '''
    # PREPARE AND INITIALIZE REQUIRED VARIABLES FOR DATA WRANGLING --------------
    
    # grab the current book title, and drop the file extension from current book title for easier file naming in the rest of the code
    current_book_title = os.path.basename(str(b))
    sep = '.'
    current_book_title = current_book_title.split(sep, 1)[0]
    
    print("The current book being analyzed is: ", current_book_title)

    # determine the number of sheets, samples, & controls in the workbook
    name_sheets = pd.ExcelFile(b).sheet_names
    num_sheets = (len(pd.ExcelFile(b).sheet_names))
    
    # initialize number of samples and controls to zero, then initialize the "list initializers" which will hold book-level data to eventually add to the book-level dataframe.
    num_samples = 0
    num_controls = 0
    sample_control_initializer = []
    sample_replicate_initializer = []
    control_replicate_initializer = []
    
    # initialize a list to contain the mini experimental dataframes collected from within the current book, which will be concatenated at the end to create one dataframe "organized_df" that represents this book
    df_list = []
    
    # BEGIN WRANGLING DATA FROM THE EXCEL FILE, AND TRANSLATING INTO ORGANIZED DATAFRAME ----------------

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
    
    print("There are", num_sheets, "sheets identified in the current book.")
    print("Number of samples sheets identified:", num_samples)
    print("Number of control sheets identified:", num_controls)
    
    if num_samples != num_controls:
        print('\nERROR: The number of sample sheets is not equal to the number of control sheets in', b, 'please confirm the data in the book is correct.')
    
    # combine sample and control initializers to create the total replicate index list
    total_replicate_index = sample_replicate_initializer + control_replicate_initializer
    print("Sample and control data initialization complete at the book-level. Beginning experiment-specific data acquisition.\n")
    
    # FOR TESTING, can set num_sheets = 1
    # num_sheets = 1
    
    # loop through each sheet in the workbook
    for n in range(num_sheets):

        # read in the current book's nth sheet into a Pandas dataframe
        current_sheet_raw = pd.read_excel(b, sheet_name=n)
        print("Reading Book Sheet:", n)
        
        # if the name of the sheet is not a sample or control, skip this sheet and continue to the next one 
        if (any(['ample' not in name_sheets[n]]) & any(['ontrol' not in name_sheets[n]])):
            print("Skipping sheet", n, "as not a sample or control.")
            continue

        # drop first always empty unnamed col (NOTE - consider building error handling into this to check the col is empty first)
        to_drop = ['Unnamed: 0']
        current_sheet_raw.drop(columns = to_drop, inplace = True)

        # loop through columns and "fill right" to replace all Unnamed columns with their corresponding title_string value
        for c in current_sheet_raw.columns:
            current_sheet_raw.columns = [current_sheet_raw.columns[i-1] if 'Unnamed' in current_sheet_raw.columns[i] else current_sheet_raw.columns[i] for i in range(len(current_sheet_raw.columns))]

        # identifies the coordinates of the left-most parameter in each experimental set, conc (um)
        i, c = np.where(current_sheet_raw == 'conc (um)')

        # assigns coordinates of all upper left 'conc (um) cells to an index (row) array and a column array
        conc_indices = current_sheet_raw.index.values[i]
        conc_columns = current_sheet_raw.columns.values[c]

        # determine the number of experimental rows in each NMR results sub-table
        # subtract one to exclude conc cell row itself
        num_experimental_indices = np.unique(conc_indices)[2] - np.unique(conc_indices)[1] - 1

        # determine the number of experimental columns in each NMR results sub-table
        (unique_exp_cols, count_experimental_cols) = np.unique(conc_columns, return_counts = True)
        num_experimental_cols = np.unique(count_experimental_cols)[0]

        # initialize/reset dfs, current is for CURRENT sub-tables being looped over, organized_df is to concatenate all the sub-dfs in this book
        current_exp_df = []
        organized_df = []
        
        # make a list of coordinate pair tuples for this sheet using list comprehension
        sheet_coords_list = [(conc_indices[i], conc_columns[i]) for i in range(len(conc_indices))]

        for coords in sheet_coords_list:
        
            # Determine the current 'sample_or_control' and 'replicate' values by taking the nth value (aka current sheet value) from the lists determined above
            current_sample_or_control = sample_control_initializer[n]
            current_replicate = total_replicate_index[n]

            # assign current values to the fixed experimental parameters for this experimental sub-table
            fixed_parameters_per_set = current_sheet_raw.loc[coords[0], coords[1]]
    
            # Hard coding the indices of the different parameters based on constant pattern in input file 
            current_title_string = fixed_parameters_per_set.index[0]
            current_concentration = fixed_parameters_per_set[1]
            current_sat_time = fixed_parameters_per_set[3]
            current_irrad_bool = fixed_parameters_per_set[5]

            # initialize/reset experimental lists to null for this experimental set
            list_current_ppm_experimental = []
            list_current_intensity_experimental = []
            list_current_width_experimental = []
            list_current_area_experimental = []
            list_current_type_experimental = []
            list_current_flags_experimental = []
            list_current_impuritycomp_experimental = []
            list_current_annotation_experimental = []
            list_current_range_experimental = []
            list_current_normalized_experimental = []
            list_current_absolute_experimental = []

            # now need to find and assign the sub-table's actual experimental data to the new lists 
                
            # Initializing the "boundaries" of the sub-table rectangle containing values of interest in a generalizable way, based on conc as the upper left value
            experimental_index_starter = coords[0]+1
            experimental_index_range = range(num_experimental_indices)
            experimental_column_range = range(num_experimental_cols)
            combined_experimental_index_range = experimental_index_starter + experimental_index_range

            # Obtain experimental column range using boolean mask as column labels are title_strings for each experimental set
            experimental_column_range_mask = current_sheet_raw.columns.get_loc(coords[1])
            combined_experimental_column_range = np.where(experimental_column_range_mask)
            
            # loop through and collect experimental index and experimental column range for the ith and jth experimental sub-table
            for ei in combined_experimental_index_range:
                for ec in combined_experimental_column_range:
                    
                    # use iloc to grab NMR experimental variables for this set into a series
                    variable_parameters_per_set = current_sheet_raw.iloc[ei, ec]

                    # append designated variable parameters into lists for those parameters in the current experimental sub-set
                    # this is hard coded based on known order of columns in the input file structure that should theoretically be consistent...
                    list_current_ppm_experimental.append(variable_parameters_per_set.iloc[1])
                    list_current_intensity_experimental.append(variable_parameters_per_set.iloc[2])
                    list_current_width_experimental.append(variable_parameters_per_set.iloc[3])
                    list_current_area_experimental.append(variable_parameters_per_set.iloc[4])
                    list_current_type_experimental.append(variable_parameters_per_set.iloc[5])
                    list_current_flags_experimental.append(variable_parameters_per_set.iloc[6])
                    list_current_impuritycomp_experimental.append(variable_parameters_per_set.iloc[7])
                    list_current_annotation_experimental.append(variable_parameters_per_set.iloc[8])
                    list_current_range_experimental.append(variable_parameters_per_set.iloc[11])
                    list_current_normalized_experimental.append(variable_parameters_per_set.iloc[12])
                    list_current_absolute_experimental.append(variable_parameters_per_set.iloc[13])
                    

            # after all the experimental lists are populated, define length of the experimental parameter lists (number of true experimental rows)
            exp_list_length = len(list_current_ppm_experimental)

            # create "ranged" lists of the constant experimental values to make them the same length as the unique variable experimental values, so we can add information "per observation" to the dataframe
            ranged_sample_or_control = exp_list_length * [current_sample_or_control]
            ranged_replicate = exp_list_length * [current_replicate]
            ranged_title_string = exp_list_length * [current_title_string]
            ranged_concentration = exp_list_length * [current_concentration] 
            ranged_sat_time = exp_list_length * [current_sat_time]
            ranged_irrad_bool = exp_list_length * [current_irrad_bool]
                
            # assign all current experimental values for this experimental set to a dataframe via a dictionary
            current_dict = {"sample_or_control":ranged_sample_or_control,
                                "replicate":ranged_replicate,
                                "title_string":ranged_title_string, 
                                "concentration":ranged_concentration,
                                "sat_time":ranged_sat_time, 
                                "irrad_bool":ranged_irrad_bool, 
                                "ppm":list_current_ppm_experimental, 
                                "intensity":list_current_intensity_experimental, 
                                "width":list_current_width_experimental, 
                                "area":list_current_area_experimental, 
                                "type":list_current_type_experimental, 
                                "flags":list_current_flags_experimental, 
                                "impurity_compound":list_current_impuritycomp_experimental, 
                                "annotation":list_current_annotation_experimental, 
                                "range":list_current_range_experimental, 
                                "normalized":list_current_normalized_experimental, 
                                "absolute":list_current_absolute_experimental}
            
            current_exp_df = pd.DataFrame(current_dict)
    
            # before moving on to next experimental set, append the dataframe from this experimental set to a book-level list of dataframes
            df_list.append(current_exp_df)

        print("Data frame for the experimental set at coordinates:", coords, "has been appended to the book-level list of dataframes.\n")
    
    # concatenate the mini dataframes appended to the df_list into one big organized dataframe for this book!
    organized_df = pd.concat(df_list)
    print("The input book {} has been wrangled into an organized dataframe! Now initializing dataframe cleaning steps.\n".format(current_book_title))
    
    # BEGIN CLEANING THE ORGANIZED DATAFRAME ----------------------------------------------------
    
    # Need to remove redundant rows (generated by the data processing loops above)

    # assign prosepctive redundant title rows at index zero to a variable
    organized_df_redundant = organized_df[organized_df.index == 0]

    # to make sure the standard expected redundant rows are actually redundant (expected redundant row is a duplicate of the column labels)
    # check that there is only one unique 'absolute' value (duplicate rows will only have one unique value, as they are titles)
    if organized_df_redundant.nunique().loc['absolute'] == 1:
        clean_df = organized_df[organized_df.index != 0]
    else: 
        print("Warning, assumed redundant rows for further processing are not actually redundant. Proceeding.\n")
        clean_df = organized_df
    
    # remove any duplicate rows
    clean_df = clean_df.drop_duplicates()

    # export the cleaned dataframe of the book to excel in a custom output folder
    output_directory = "{}/{}".format(global_output_directory, current_book_title)
    output_file_name = "{}_clean_raw_df.xlsx".format(current_book_title)

    # make directory if there isn't already one for output 
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    print('Dataframe cleaning complete!\n')
    print('Now exporting the cleaned dataframe in Excel format to a custom output directory for reference.\n')
    clean_df.to_excel(os.path.join(output_directory, output_file_name))

    print('Excel export complete! Navigate to output directory to see the clean Excel file.\n')
    
    # merge current_book_title and clean_df into a tuple, as "key value pairs" that can be generically indexed
    clean_tuple = (current_book_title, clean_df)
    print('Function has returned a tuple containing the title of the current book, and the cleaned dataframe for {}.\n'.format(current_book_title))
    
    return clean_tuple

def attenuation_calc_equality_checker(compare_df_1, compare_df_2, batch_or_book = 'book'):
    
    '''This functions checks to see if two subset dataframes for the attenuation calculation, one where irrad bool is true, one where irrad bool is false, for calculating attenuation
    are equal and in the same order in terms of their fixed experimental parameters. 'sample_or_control', 'replicate', 'title_string', 'concentration', 
    'sat_time' '''
    
    if (compare_df_1.shape == compare_df_2.shape):
        
        if batch_or_book == 'book':
            subset_compare_1 = compare_df_1[['sample_or_control', 'replicate', 'title_string', 'concentration', 'sat_time']]
            subset_compare_2 = compare_df_2[['sample_or_control', 'replicate', 'title_string', 'concentration', 'sat_time']]
        
            exactly_equal = subset_compare_2.equals(subset_compare_1)
    
            return exactly_equal
    
        else:
            
            #check sample_or_control is the same
            subset_compare_1 = compare_df_1.sample_or_control.values
            subset_compare_2 = compare_df_2.sample_or_control.values
            exactly_equal_1 = np.array_equal(subset_compare_1, subset_compare_2)
            
            
            #check replicate is the same
            subset_compare_1 = compare_df_1.replicate.values
            subset_compare_2 = compare_df_2.replicate.values
            exactly_equal_2 = np.array_equal(subset_compare_1, subset_compare_2)
            #             exactly_equal_2 = subset_compare_2.equals(subset_compare_1)
            
            #check proton peak index is the same
            subset_compare_1 = compare_df_1.proton_peak_index.values
            subset_compare_2 = compare_df_2.proton_peak_index.values
            exactly_equal_3 = np.array_equal(subset_compare_1, subset_compare_2)
            
            #check sat time is the same
            subset_compare_1 = compare_df_1.sat_time.values
            subset_compare_2 = compare_df_2.sat_time.values
            exactly_equal_4 = np.array_equal(subset_compare_1, subset_compare_2)
            
            #if passes all 4 checks return true, if not false
            if exactly_equal_1 == exactly_equal_2 == exactly_equal_3 == exactly_equal_4 == True:
                return True
            
            else: return False
        
    else:
        raise ValueError("Error, irrad_false and irrad_true dataframes are not the same shape to begin with.")

def corrected_attenuation_calc_equality_checker(compare_df_1, compare_df_2, compare_df_3):  
    
    '''This functions checks to see if the three subset dataframes for calculating the corrected % attenuation
    are equal and in the same order in terms of their shared fixed experimental parameters. 'replicate', 'concentration', 
    'sat_time' '''
    
    #check if number of rows same in each df, number of columns not same as samples dfs contain attenuation data
    if (compare_df_1.shape[0] == compare_df_2.shape[0] == compare_df_3.shape[0]):
        
        #check replicate is the same
        subset_compare_1 = compare_df_1.replicate.values
        subset_compare_2 = compare_df_2.replicate.values
        subset_compare_3 = compare_df_3.replicate.values
        
        exactly_equal_1 = np.logical_and((subset_compare_1==subset_compare_2).all(), (subset_compare_2==subset_compare_3).all())
        
        #check sat time is the same
        subset_compare_1 = compare_df_1.sat_time.values
        subset_compare_2 = compare_df_2.sat_time.values
        subset_compare_3 = compare_df_3.sat_time.values
        
        exactly_equal_2 = np.logical_and((subset_compare_1==subset_compare_2).all(), (subset_compare_2==subset_compare_3).all())
        
        #check concentration is the same
        subset_compare_1 = compare_df_1.concentration.values
        subset_compare_2 = compare_df_2.concentration.values
        subset_compare_3 = compare_df_3.concentration.values
        
        exactly_equal_3 = np.logical_and((subset_compare_1==subset_compare_2).all(), (subset_compare_2==subset_compare_3).all())
        

        #if passes all 3 checks return true, if not false
        if exactly_equal_1 == exactly_equal_2 == exactly_equal_3 == True:
            return True
            
        else: return False
        
        
    else:
        raise ValueError("Error, corrected % attenuation input dataframes are not the same shape to begin with.")

def add_attenuation_and_corr_attenuation_to_dataframe(current_book, batch_or_book = 'book'):
    '''
    This function calculates the attenuation, and corr_%_attenuation if the dataframe passes all checks
    (based on the order of items) by means of simple arithmetic operations.
    
    Input: current_book, the dataframe output from the convert_excel_to_dataframe() function. (Or batch processing equivalent)
    
    Outut: corr_p_attenuation_df, an updated dataframe that includes the attenuation and corr_%_attenuation columns.
    
    Default is book, but it will run the batch path if 'batch' is passed to function as the second arg.
    
    '''
    
    # to get get % attenuation of peak integral, define true and false irrad dataframes below, can perform simple subtraction if passes check
    
    if batch_or_book == 'book':
        intensity_irrad_true = current_book.loc[(current_book['irrad_bool'] == 1.0), ['sample_or_control', 'replicate', 'title_string', 'concentration', 'sat_time', 'ppm', 'intensity', 'range', 'normalized', 'absolute']]
        intensity_irrad_false = current_book.loc[(current_book['irrad_bool'] == 0.0), ['sample_or_control', 'replicate', 'title_string', 'concentration', 'sat_time', 'ppm', 'intensity', 'range', 'normalized', 'absolute']]
    
    else:
        intensity_irrad_true = current_book.loc[(current_book['irrad_bool'] == True), ['polymer_name', 'proton_peak_index', 'ppm_range', 'ppm', 'sample_or_control', 'replicate', 'concentration', 'sat_time', 'absolute']]
        intensity_irrad_false = current_book.loc[(current_book['irrad_bool'] == False), ['polymer_name', 'proton_peak_index',  'ppm_range', 'ppm', 'sample_or_control', 'replicate', 'concentration', 'sat_time', 'absolute']]
        
        
    # check if the fixed experimental values in irrad true and irrad false are equal, in the same order, and the same size, so that one to one calculations can be performed to calculate attenuation.
    fixed_values_equality_check = attenuation_calc_equality_checker(intensity_irrad_true, intensity_irrad_false, batch_or_book)

    # if the test passes, calculate attenuation and append to dataframe
    if fixed_values_equality_check == True:

        p_attenuation_intensity = intensity_irrad_false.absolute.values - intensity_irrad_true.absolute.values
    
        #Update irradiated dataframe to include the % attenuation of the irradiated samples and controls
        intensity_irrad_true['attenuation'] = p_attenuation_intensity
        print("Test 1 passed, attenuation has been calculated and appended to dataframe.")
        
    else:
        raise ValueError("Error, intensity_irrad true and false dataframes are not equal, cannot compute signal attenutation in a one-to-one manner.")

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
        
        #initialize different cols depending on whether batch or book
        if batch_or_book == 'book':
            #initialize dataframe to store corrected p_attenuation with the shared fixed parameters and sample identifier parameters
            corr_p_attenuation_df = pd.DataFrame(p_atten_intensity_sample[['sample_or_control', 'replicate', 'title_string', 'concentration', 'sat_time', 'ppm', 'intensity', 'range', 'normalized', 'absolute', 'attenuation']])
        else:
            #initialize dataframe to store corrected p_attenuation with the shared fixed parameters and sample identifier parameters
            corr_p_attenuation_df = pd.DataFrame(p_atten_intensity_sample[['polymer_name', 'concentration', 'sat_time', 'proton_peak_index', 'ppm_range', 'ppm', 'sample_or_control', 'replicate', 'absolute', 'attenuation']])
            
        #Calculate Corrected % Attentuation, as applies to
        corr_p_attenuation_df['corr_%_attenuation'] = ((1/intensity_irrad_false_sample_data)*(p_atten_intensity_sample_data - p_atten_intensity_control_data))
        print("Test 2 passed, corr_%_attenuation has been calculated and appended to dataframe.")
        
        return corr_p_attenuation_df

    else:
        raise ValueError("Error, input dataframes are not equal, cannot compute corrected signal attenutation in a one-to-one manner.")
        
def generate_concentration_plot(current_df_attenuation, output_directory_exploratory, current_df_title):
    
    '''
    This function generates a basic exploratory stripplot of polymer sample attenuation vs saturation time using
    concentration is a "hue" to differentiate points. 
    
    This function also saves the plot to a custom output folder.
    
    Input: dataframe after attenuation and corrected % attenuation have been calculated.
    Output: saves plot to file, and displays it.
    
    '''
    a4_dims = (11.7, 8.27)
    fig1, ax = plt.subplots(figsize = a4_dims)
    sns.stripplot(ax = ax, x = 'sat_time', y = 'corr_%_attenuation', data = current_df_attenuation, hue = 'concentration', palette = 'viridis')

    plt.title("Polymer Sample Attenuation vs Saturation Time")
    plt.ylabel("Corrected Signal Intensity Attenuation (%)")
    plt.xlabel("NMR Pulse Saturation Time (s)")
    
        
    # define file name for the concentration plot
    output_file_name_conc = "{}/exploratory_concentration_plot_from_{}.png".format(output_directory_exploratory, current_df_title)
    
    # export to file
    fig1.savefig(output_file_name_conc, dpi=300)

    return

def generate_ppm_plot(current_df_attenuation, output_directory_exploratory, current_df_title):
    
    '''
    This function generates a basic exploratory scatterplot of polymer sample attenuation vs saturation time using
    ppm as a "hue" to differentiate points. 
    
    This function also saves the plot to a custom output folder.
    
    Input: dataframe after attenuation and corrected % attenuation have been calculated.
    Output: saves plot to file, and displays it.
    '''
    
    a4_dims = (11.7, 8.27)
    fig2, ax2 = plt.subplots(figsize = a4_dims)
    sns.scatterplot(ax = ax2, x = 'sat_time', y = 'corr_%_attenuation', data = current_df_attenuation, hue ='ppm', palette = 'viridis', y_jitter = True, legend = 'brief')

    # a stripplot looks nicer than this, but its legend is unneccessarily long with each individual ppm, need to use rounded ppm to use the below line
    # sns.stripplot(ax = ax2, x = 'sat_time', y = 'corr_%_attenuation', data = corr_p_attenuation_df, hue ='ppm', palette = 'viridis', dodge = True)

    plt.title("Polymer Sample Attenuation vs Saturation Time")
    plt.ylabel("Corrected Signal Intensity Attenuation  (%) by ppm")
    plt.xlabel("NMR Pulse Saturation Time (s)")
    ax2.legend() 

    # define file name for the concentration plot
    output_file_name_ppm = "{}/exploratory_ppm_plot_from_{}.png".format(output_directory_exploratory, current_df_title)

    # export to file
    fig2.savefig(output_file_name_ppm, dpi=300)
        
    return

def prep_mean_data_for_stats(corr_p_attenuation_df, batch_or_book = 'book'):
    
    '''
    This function prepares the dataframe for statistical analysis after the attenuation and corr_%_attenuation
    columns have been added.
    
    Statisical analysis is performed on a "mean" basis, across many experimental replicates.
    This code prepares the per-observation data accordingly, and outputs the mean_df_for_stats.
    
    It drops the columns and rows not required for stats, calculates the mean and std of parameters we do 
    care about, and also appends the degrees of freedom and sample size.
    
    Input: current_df_attenuation
    Output: mean_current_df_for_stats 
    
    Default is book, but it will run the batch path if 'batch' is passed to function as the second arg.
    
    '''
    # follow this path if data is from a single polymer book
    if batch_or_book == 'book':
        
        # drop any rows that are entirely null from the dataframe 
        corr_p_attenuation_df = corr_p_attenuation_df.dropna(how = "any")

        # now drop the column fields that are not required for stats modelling and further analysis
        data_for_stats = corr_p_attenuation_df.drop(columns = ['title_string', 'sample_or_control', 'intensity', 'range', 'normalized', 'absolute', 'attenuation'])

        # Add a new column to data for the index of proton peaks in an experimental set of a polymer (i.e. proton peak index applies to index protons within one polymer book)
        proton_index = data_for_stats.index
        data_for_stats['proton_peak_index'] = proton_index
    
    
        # determine mean corr % attenuation and mean ppm per peak index, time, and concentration across replicates using groupby sum (reformat) and groupby mean (calculate mean)
        regrouped_df = data_for_stats.groupby(by = ['concentration', 'sat_time', 'proton_peak_index', 'replicate'])[['ppm','corr_%_attenuation']].sum()

        # generate a table that includes the mean and std for ppm and corr_%_atten across the replicates, reset index
        mean_corr_attenuation_ppm = regrouped_df.groupby(by = ['concentration', 'sat_time', 'proton_peak_index']).agg({'ppm': ['mean', 'std'], 'corr_%_attenuation': ['mean', 'std']})
        
        # prepare input for dofs function
        input_for_dofs = regrouped_df.index.get_level_values(2)
    
    # if data came from a batch, ppm value is static and 1 less DoF, so perform adjusted operations
    else: 

        # now drop the column fields that are not required for stats modelling and further analysis
        data_for_stats = corr_p_attenuation_df.drop(columns = ['sample_or_control', 'absolute', 'ppm_range'])

        # determine mean corr % attenuation per peak index, time, and concentration across replicates using groupby sum (reformat) and groupby mean (calculate mean)
        regrouped_df = data_for_stats.groupby(by = ['concentration','sat_time', 'proton_peak_index', 'replicate', 'ppm'])[['corr_%_attenuation']].sum()

        # generate a table that includes the mean and std for ppm and corr_%_atten across the replicates, reset index
        mean_corr_attenuation_ppm = regrouped_df.groupby(by = ['concentration', 'sat_time', 'proton_peak_index', 'ppm']).agg({'corr_%_attenuation': ['mean', 'std']})
        
        # prepare input for dofs function
        input_for_dofs = regrouped_df.index.get_level_values(2)
        
    
    def get_dofs(peak_indices_array):
    
        ''' This function calculates the number of degrees of freedom (i.e. number of experimental replicates minus one) for statistical calculations 
        using the "indices array" of a given experimenal set as input.
    
        Input should be in the format: (in format: 11111 22222 3333 ... (specifically, it should be the proton_peak_index column from the "regrouped_df" above)
        where the count of each repeated digit minus one represents the degrees of freedom for that peak (i.e. the number of replicates -1).
        With zero based indexing, the function below generates the DOFs for the input array of proton_peak_index directly, in a format
        that can be directly appended to the stats table.
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
    
    # Calculate degrees of freedom and sample size for each datapoint using function above
    peak_index_array = np.array(input_for_dofs)
    dofs = get_dofs(peak_index_array)

    # append a new column with the calculated degrees of freedom to the table for each proton peak index
    mean_corr_attenuation_ppm['dofs'] = dofs
    mean_corr_attenuation_ppm['sample_size'] = np.asarray(dofs) + 1
        
        
    return mean_corr_attenuation_ppm 

def prep_replicate_data_for_stats(corr_p_attenuation_df, batch_or_book = 'book'):
    
    '''
    This function prepares the dataframe for statistical analysis after the attenuation and corr_%_attenuation
    columns have been added.
    
    This code prepares the per-observation data accordingly, and outputs the replicate_df_for_stats.
    
    It drops the columns and rows not required for stats, and adds the proton peak index.
    
    Input: current_df_attenuation
    Output: replicate_current_df_for_stats 
    
    Defaults to book processing path, but if 'batch' is passed to function will pursue batch path.
    
    '''
    
    # drop any rows that are entirely null from the dataframe 
    corr_p_attenuation_df = corr_p_attenuation_df.dropna(how = "any")
    
    if batch_or_book == 'book':
        # now drop the column fields that are not required for stats modelling and further analysis
        replicate_df_for_stats = corr_p_attenuation_df.drop(columns = ['title_string', 'sample_or_control', 'intensity', 'range', 'normalized', 'absolute', 'attenuation'])
        
        # Add a new column to data for the index of proton peaks in an experimental set of a polymer (i.e. proton peak index applies to index protons within one polymer book)
        proton_index = replicate_df_for_stats.index
        replicate_df_for_stats['proton_peak_index'] = proton_index
    
    else:
        replicate_df_for_stats = corr_p_attenuation_df.drop(columns = ['sample_or_control', 'absolute', 'attenuation'])
    
    return replicate_df_for_stats

def get_t_test_results(mean_corr_attenuation_ppm, p=0.95):

    ''' 
        Procedure followed from: https://machinelearningmastery.com/critical-values-for-statistical-hypothesis-testing/ 

        One sample t test: tests whether the mean of a population is significantly different than a sample mean.
        A proper t-test analysis performs calculations to help infer what the expected population mean (that 
        contains the sample) given just a sample mean. (Inferential statistics).

        Null Hypothesis: population mean = sample mean (avg corr % attenuation for a proton peak of N replicates at fixed experimental parameters)
        Alternative Hypothesis: population mean >= sample mean (avg corr % attenuation for a proton peak of N replicates at fixed experimental parameters)

        Population parameters used are that of the student's t distribution.
        Alpha = 0.1 for statistical significance. 

        Still theoretically need to validate sample data meets assumptions for one sample t test: normal dist, homegeneity of variance, & no outliers. **
        
        Input: current_df_mean after having been prepared for stats (output from prep_mean_data_for_stats)
        Output: The same df as input, only now with the t test values and results appended
        
        Book and batch paths are the same.
        
        Target p value defaults to 0.95 unless other argument passed.
        
    '''

    # Initialize Relevant t test Parameters
    dof = mean_corr_attenuation_ppm['dofs']
    mean_t = mean_corr_attenuation_ppm['corr_%_attenuation']['mean'].abs()
    std_t = mean_corr_attenuation_ppm['corr_%_attenuation']['std']
    sample_size_t = mean_corr_attenuation_ppm['sample_size']

    # retrieve value <= probability from t distribution, based on DOF for sample 
    crit_t_value = t.ppf(p, dof)
    # print(crit_t_value)

    # confirm the p value with cdf, for sanity checking 
    # p = t.cdf(crit_t_value, dof)
    # print(p)

    #perform one sample t-test for significance, significant if t_test_results < 0 
    t_test_results =  mean_t - crit_t_value * (std_t/(np.sqrt(sample_size_t)))
    mean_corr_attenuation_ppm['t_results'] = t_test_results
    mean_corr_attenuation_ppm['significance'] = mean_corr_attenuation_ppm['t_results'] > 0

    #Return the dataframe with the new t test columns appended
    return mean_corr_attenuation_ppm

def compute_amplification_factor(current_mean_stats_df, current_replicate_stats_df, af_denominator):
    '''
    This function computes an amplification factor for the mean stats df and the replicate stats df.
    Each polymer may have a different denominator to consider for the amp factor calculation, so it
    is passed into this function as a variable.
    
    Input: mean_stats_df, replicates stats df, the denominator for amp factor calculation
    Output: mean_stats_df, replicates_stats_df with the amp factor column 
    
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

def drop_bad_peaks(current_df_mean, current_df_replicates, current_df_title, output_directory, batch_or_book='book'):

    '''
    This function identifies whether proton peaks pass or fail an acceptance criterion to allow
    them to be further analyzed. If the peaks fail, they are dropped from further analysis.
    
    Criterion for dropping peaks from Further consideration: If more than two proton peak datapoints are flagged as not significant in the mean dataframe 
    WITHIN a given concentration, the associated proton peak is removed from further analysis.
    
    Input: current_df_mean and current_df_replicats
    Outputs: the same dataframes, after dropping proton peaks that fail criterion, writes a text file of which points have been dropped

    '''

    #initialize a df that will keep data from the current mean df that meet the criterion above
    significant_corr_attenuation = current_df_mean

    #The below code checks data for the criterion and then removes rows if they do not pass for the mean df --------

    # Assign unique protons to a list to use for subsetting the df via multi-index and the multi-index slicing method
    unique_protons = current_df_replicates.proton_peak_index.unique()
    unique_concentrations = current_df_replicates.concentration.unique().tolist()

    #initialize list to contain the points to remove 
    pts_to_remove = []

    if batch_or_book == 'book':
        for p in unique_protons:
            for c in unique_concentrations:

                #subset the df via index slice based on the current peak and concentration
                current_subset_df = significant_corr_attenuation.loc[idx[c, :, p]]

                #subset further for where significance is false
                subset_insignificant = current_subset_df.loc[(current_subset_df['significance'] == False)]

                #if there's more than 2 datapoints where significance is False within the subset, drop p's proton peaks for c's concentration from the significant_corr_attenuation df
                if len(subset_insignificant) > 2:

                    pts_to_remove.append(current_subset_df.index)
                    significant_corr_attenuation = significant_corr_attenuation.drop(current_subset_df.index, inplace = False)  

        print('Removed insignificant points have been printed to the output folder for {}.'.format(current_df_title))

        #create and print dropped points to a summary file
        dropped_points_file = open("{}/dropped_insignificant_points.txt".format(output_directory), "w")
        dropped_points_file.write("The datapoints dropped from consideration due to not meeting the criteria for significance are: \n{}".format(pts_to_remove))
        dropped_points_file.close()

        #define a mean df of the data that passed to return
        current_df_mean_passed = significant_corr_attenuation

        #The below code removes the same bad points from the replicate df ------------

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


    else:
        
        for p in unique_protons:
            for c in unique_concentrations:
                    
                #subset the df via index slice based on the current peak and concentration, ppm is part of index here
                current_subset_df = significant_corr_attenuation.loc[idx[c, :, p, :]]
                
                #subset further for where significance is false
                subset_insignificant = current_subset_df.loc[(current_subset_df['significance'] == False)]

                #if there's more than 2 datapoints where significance is False within the subset, drop p's proton peaks for c's concentration from the significant_corr_attenuation df
                if len(subset_insignificant) > 2:
                    
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
                    significant_corr_attenuation = significant_corr_attenuation.drop(multi_index_to_drop, inplace = False)                     
        print('Removed insignificant points have been printed to the output folder for {}.'.format(current_df_title))

        #create and print dropped points to a summary file
        dropped_points_file = open("{}/dropped_insignificant_points.txt".format(output_directory), "w")
        dropped_points_file.write("The datapoints dropped from consideration due to not meeting the criteria for significance are: \n{}".format(pts_to_remove))
        dropped_points_file.close()

        #define a mean df of the data that passed to return
        current_df_mean_passed = significant_corr_attenuation

        #The below code removes the same bad points from the replicate df ------------

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

def y_hat_fit(t, a, b):
    '''
    This function returns y_ikj_hat as the fit model based on alpha and beta.
    
    Some useful scipy example references for curve fitting:
    1) https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    2) https://lmfit.github.io/lmfit-py/model.html
    3) https://astrofrog.github.io/py4sci/_static/15.%20Fitting%20models%20to%20data.html 

    '''
    return a * (1 - np.exp(t * -b))

def execute_curvefit(stats_df_mean, stats_df_replicates, output_directory2, output_directory3, current_df_title, batch_or_book = 'book'):
    '''
    
    We are now ready to calculate the nonlinear curve fit models (or "hat" models), 
    for both individual replicate data (via stats_df_replicates), and on a mean (or "bar") basis (via stats_df_mean). 
    
    This function carries out the curve fitting process for the current dataframe.
    It calculates the nonlinear curve fit, then the SSE and AFo on a mean and replicate basis using only significant points.
    
    Input: stats_df_mean, stats_df_replicates after all other pre processing activities have occurred/bad pts dropped
    Output: returns final updated dataframes, figures are displayed, figures are saved to file in custom directory, and dataframes saved to Excel file with final results
    
    Some housekeeping notes to avoid confusion: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
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
    
    
    # book path
    if batch_or_book == 'book':
    
        for c in unique_concentrations:
            for p in unique_protons:        

                #COMPLETE MEAN CURVE FITTING OPERATIONS PER PROTON & PER CONCENTRATION

                # subset the df into the data for one graph, via index slice based on the current peak and concentration
                one_graph_data_mean = stats_df_mean.loc[(slice(c), slice(None), slice(p)) , :]

                #Make a boolean significance mask based on the one graph subset, for calculating parameters based on only significant pts
                boolean_sig_mask = one_graph_data_mean.significance == True

                #assign ALL datapoints and ALL data times to test_data variables for this ONE GRAPH
                all_yikj_bar = one_graph_data_mean['yikj_bar']
                all_sat_time = np.asarray(all_yikj_bar.index.get_level_values(1))

                #apply boolean significance mask to create the data to be used for actual curve fitting/parameter generation
                significant_yikj_bar = all_yikj_bar[boolean_sig_mask]
                significant_sat_time = all_sat_time[boolean_sig_mask]

                # grab the current mean ppm for this graph to use in naming and plotting
                ppm_bar = one_graph_data_mean['ppm']['mean'].values.mean().astype(float).round(4)

                # this will skip the graphing and analysis for cases where an insignificant proton peak has been removed from consideration PREVIOUSLY due to cutoff
                if all_yikj_bar.size == 0: continue

                # initial guess for alpha and beta (applies equally to replicate operations below)
                initial_guess = np.asarray([1, 1])

                # Generate best alpha & beta parameters for data based on only significant pts, for the current proton and concentration, optimizing for minimization of square error via least squares levenburg marquadt algorithm
                best_param_vals_bar, covar_bar = curve_fit(y_hat_fit, significant_sat_time, significant_yikj_bar, p0 = initial_guess, method = 'lm', maxfev=5000)

                # calculate ultimate sum of square errors after minimization for each time point
                sse_bar = np.square(y_hat_fit(significant_sat_time, *best_param_vals_bar) - significant_yikj_bar)

                #append sum of square error calculated for this graph to the PARENT mean dataframe at this c and p
                stats_df_mean.loc[(slice(c), slice(None), slice(p)), ('SSE_bar')] = sse_bar.sum()
                
                # append best parameters to variables, and then generate the instantaneous amplification factor 
                a_kj_bar = best_param_vals_bar[0]
                b_kj_bar = best_param_vals_bar[1]

                amp_factor_instantaneous_bar = a_kj_bar * b_kj_bar
                
                #append instantaneous amplification factor calculated to the PARENT mean dataframe, for all datapoints in this graph
                # stats_df_mean.loc[idx[c, :, p], ('AFo_bar')] = [amp_factor_instantaneous_bar]*(len(all_yikj_bar))
                stats_df_mean.loc[(slice(c), slice(None), slice(p)), ('AFo_bar')] = [amp_factor_instantaneous_bar]*(len(all_yikj_bar))
                # define file name for curve fits by mean
                output_file_name_figsmean = "{}/mean_conc{}_ppm{}.png".format(output_directory2, c, ppm_bar)

                # PLOT MEAN DF CURVE FITS with the original data and save to file
                fig1, (ax1) = plt.subplots(1, figsize = (8, 4))
                ax1.plot(all_sat_time, y_hat_fit(all_sat_time, a_kj_bar, b_kj_bar), 'g-', label='model_w_significant_params')
                ax1.plot(all_sat_time, all_yikj_bar, 'g*', label='all_raw_data')
                ax1.set_title('Mean Curve Fit, Concentration = {} µmolar, ppm = {}'.format(c,ppm_bar))
                ax1.set_xlabel('NMR Saturation Time (s)')
                ax1.set_ylabel('I/Io')
                plt.rcParams.update({'figure.max_open_warning': 0})
                fig1.tight_layout()

                # export to file
                fig1.savefig(output_file_name_figsmean, dpi=300)

                for r in unique_replicates:

                    #COMPLETE REPLICATE SPECIFIC CURVE FIT OPERATIONS - subset the df via index slice based on the current peak, concentration, and replicate
                    one_graph_data = stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r)]

                    # define the experimental data to compare square error with (amp_factor * atten_corr_int), for y_ikj
                    y_ikj = one_graph_data['yikj']

                    #this will skip the graphing and analysis for cases where a proton peak has been removed from consideration 
                    if y_ikj.size == 0: continue

                    # define sat_time to be used for the x_data 
                    sat_time = one_graph_data[['sat_time']].values.ravel()

                    #Fit Curve for curren proton, concentration and replicate, optimizing for minimization of square error via least squares levenburg marquadt algorithm
                    best_param_vals, covar = curve_fit(y_hat_fit, sat_time, y_ikj, p0 = initial_guess, method = 'lm', maxfev=5000)

                    #calculate ultimate sum of square errors after minimization for each time point, and append to list
                    sse = np.square(y_hat_fit(sat_time, *best_param_vals) - y_ikj)

                    #appends sum of square error calculated to the PARENT stats replicate dataframe, summed for all datapoints in this graph
                    stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r), ('SSE')] = sse.sum()    

                    # solve for the instantaneous amplification factor
                    a_kj = best_param_vals[0]
                    b_kj = best_param_vals[1]
                    amp_factor_instantaneous = a_kj * b_kj

                    #appends instantaneous amplification factor calculated to the PARENT stats replicate dataframe, for all datapoints in this graph
                    stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r), ('AFo')] = [amp_factor_instantaneous]*(len(y_ikj))

                    #determine mean current ppm across the sat_times for this replicate so that we can add it to the file name
                    mean_current_ppm = one_graph_data.loc[(one_graph_data['concentration'] == c) & (one_graph_data['proton_peak_index'] == p) & (one_graph_data['replicate'] == r)]['ppm'].mean().astype(float).round(4)    

                    # file name for curve fits by replicate 
                    output_file_name_figsrep = "{}/replicate{}_conc{}_ppm{}.png".format(output_directory2, r, c, mean_current_ppm)
                    
                    # PLOT CURVE FITS with original data per Replicate and save to file
                    fig2, (ax2) = plt.subplots(1, figsize = (8, 4))
                    ax2.plot(sat_time, y_hat_fit(sat_time, *best_param_vals), 'b-', label='data')
                    ax2.plot(sat_time, y_ikj, 'b*', label='data')
                    ax2.set_title('Replicate = {} Curve Fit, Concentration = {} µmolar, ppm = {}'.format(r, c, mean_current_ppm))
                    ax2.set_xlabel('NMR Saturation Time (s)')
                    ax2.set_ylabel('I/Io')
                    plt.rcParams.update({'figure.max_open_warning': 0})
                    fig2.tight_layout()

                    #export to file
                    fig2.savefig(output_file_name_figsrep, dpi=300)

        print('Export of all figures to file complete!')

        #export tabulated results to file and return updated dataframes
        output_file_name = "stats_analysis_output_replicate_{}.xlsx".format(current_df_title) 

        #export replicates final results table to a summary file in Excel
        stats_df_replicates.to_excel(os.path.join(output_directory3, output_file_name))

        #export mean final results table to a summary file in Excel
        output_file_name = "stats_analysis_output_mean_{}.xlsx".format(current_df_title)
        stats_df_mean.to_excel(os.path.join(output_directory3, output_file_name))

        return stats_df_mean, stats_df_replicates
    
    #batch path
    else:
        for c in unique_concentrations:
            for p in unique_protons:        

                #COMPLETE MEAN CURVE FITTING OPERATIONS PER PROTON & PER CONCENTRATION

                # subset the df into the data for one graph, via index slice based on the current peak and concentration
                # one_graph_data_mean = stats_df_mean.loc[idx[c, :, p], :]
                one_graph_data_mean = stats_df_mean.loc[(slice(c), slice(None), slice(p)), :]

                #Make a boolean significance mask based on the one graph subset, for calculating parameters based on only significant pts
                boolean_sig_mask = one_graph_data_mean.significance == True

                #assign ALL datapoints and ALL data times to test_data variables for this ONE GRAPH
                all_yikj_bar = one_graph_data_mean['yikj_bar']
                all_sat_time = np.asarray(all_yikj_bar.index.get_level_values(0))

                #apply boolean significance mask to create the data to be used for actual curve fitting/parameter generation
                significant_yikj_bar = all_yikj_bar[boolean_sig_mask]
                significant_sat_time = all_sat_time[boolean_sig_mask]

                # grab the current ppm for this graph to use in naming and plotting
                ppm_bar = one_graph_data_mean.index.get_level_values(1)[0].astype(float).round(4)

                # this will skip the graphing and analysis for cases where an insignificant proton peak has been removed from consideration PREVIOUSLY due to cutoff
                if all_yikj_bar.size == 0: continue

                # initial guess for alpha and beta (applies equally to replicate operations below)
                initial_guess = np.asarray([1, 1])

                # Generate best alpha & beta parameters for data based on only significant pts, for the current proton and concentration, optimizing for minimization of square error via least squares levenburg marquadt algorithm
                best_param_vals_bar, covar_bar = curve_fit(y_hat_fit, significant_sat_time, significant_yikj_bar, p0 = initial_guess, method = 'lm', maxfev=5000)

                # calculate ultimate sum of square errors after minimization for each time point
                sse_bar = np.square(y_hat_fit(significant_sat_time, *best_param_vals_bar) - significant_yikj_bar)

                #append sum of square error calculated for this graph to the PARENT mean dataframe at this c and p
                stats_df_mean.loc[(slice(c), slice(None), slice(p)), ('SSE_bar')] = sse_bar.sum()

                # append best parameters to variables, and then generate the instantaneous amplification factor 
                a_kj_bar = best_param_vals_bar[0]
                b_kj_bar = best_param_vals_bar[1]

                amp_factor_instantaneous_bar = a_kj_bar * b_kj_bar

                #append instantaneous amplification factor calculated to the PARENT mean dataframe, for all datapoints in this graph
                stats_df_mean.loc[(slice(c), slice(None), slice(p)), ('AFo_bar')] = [amp_factor_instantaneous_bar]*(len(all_yikj_bar))
                
                # define file name for curve fits by mean
                output_file_name_figsmean = "{}/mean_conc{}_ppm{}.png".format(output_directory2, c, ppm_bar)

                # PLOT MEAN DF CURVE FITS with the original data and save to file
                fig1, (ax1) = plt.subplots(1, figsize = (8, 4))
                ax1.plot(all_sat_time, y_hat_fit(all_sat_time, a_kj_bar, b_kj_bar), 'g-', label='model_w_significant_params')
                ax1.plot(all_sat_time, all_yikj_bar, 'g*', label='all_raw_data')
                ax1.set_title('Mean Curve Fit, Concentration = {} µmolar, ppm = {}'.format(c,ppm_bar))
                ax1.set_xlabel('NMR Saturation Time (s)')
                ax1.set_ylabel('I/Io')
                plt.rcParams.update({'figure.max_open_warning': 0})
                fig1.tight_layout()

                # export to file
                fig1.savefig(output_file_name_figsmean, dpi=300)

                for r in unique_replicates:

                    #COMPLETE REPLICATE SPECIFIC CURVE FIT OPERATIONS - subset the df via index slice based on the current peak, concentration, and replicate
                    one_graph_data = stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r)]

                    # define the experimental data to compare square error with (amp_factor * atten_corr_int), for y_ikj
                    y_ikj = one_graph_data['yikj']

                    #this will skip the graphing and analysis for cases where a proton peak has been removed from consideration 
                    if y_ikj.size == 0: continue

                    # define sat_time to be used for the x_data 
                    sat_time = one_graph_data[['sat_time']].values.ravel()

                    #Fit Curve for curren proton, concentration and replicate, optimizing for minimization of square error via least squares levenburg marquadt algorithm
                    best_param_vals, covar = curve_fit(y_hat_fit, sat_time, y_ikj, p0 = initial_guess, method = 'lm', maxfev=5000)

                    #calculate ultimate sum of square errors after minimization for each time point, and append to list
                    sse = np.square(y_hat_fit(sat_time, *best_param_vals) - y_ikj)

                    #appends sum of square error calculated to the PARENT stats replicate dataframe, summed for all datapoints in this graph
                    stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r), ('SSE')] = sse.sum()    

                    # solve for the instantaneous amplification factor
                    a_kj = best_param_vals[0]
                    b_kj = best_param_vals[1]
                    amp_factor_instantaneous = a_kj * b_kj

                    #appends instantaneous amplification factor calculated to the PARENT stats replicate dataframe, for all datapoints in this graph
                    stats_df_replicates.loc[(stats_df_replicates['proton_peak_index'] == p) & (stats_df_replicates['concentration'] == c) & (stats_df_replicates['replicate'] == r), ('AFo')] = [amp_factor_instantaneous]*(len(y_ikj))

                    #determine mean current ppm across the sat_times for this replicate so that we can add it to the file name
                    mean_current_ppm = one_graph_data.loc[(one_graph_data['concentration'] == c) & (one_graph_data['proton_peak_index'] == p) & (one_graph_data['replicate'] == r)]['ppm'].values[0].astype(float).round(4)    
    
                    # file name for curve fits by replicate 
                    output_file_name_figsrep = "{}/replicate{}_conc{}_ppm{}.png".format(output_directory2, r, c, mean_current_ppm)
                    
                    # PLOT CURVE FITS with original data per Replicate and save to file
                    fig2, (ax2) = plt.subplots(1, figsize = (8, 4))
                    ax2.plot(sat_time, y_hat_fit(sat_time, *best_param_vals), 'b-', label='data')
                    ax2.plot(sat_time, y_ikj, 'b*', label='data')
                    ax2.set_title('Replicate = {} Curve Fit, Concentration = {} µmolar, ppm = {}'.format(r, c, mean_current_ppm))
                    ax2.set_xlabel('NMR Saturation Time (s)')
                    ax2.set_ylabel('I/Io')
                    plt.rcParams.update({'figure.max_open_warning': 0})
                    fig2.tight_layout()

                    #export to file
                    fig2.savefig(output_file_name_figsrep, dpi=300)
        

        #export tabulated results to file and return updated dataframes
        output_file_name = "stats_analysis_output_replicate_{}.xlsx".format(current_df_title) 

        #export replicates final results table to a summary file in Excel
        stats_df_replicates.to_excel(os.path.join(output_directory3, output_file_name))
        
        #if there are replicates, and mean data was created, export the final mean data to excel as well
        if stats_df_mean.shape[0] != 0:

            #export mean final results table to a summary file in Excel
            output_file_name = "stats_analysis_output_mean_{}.xlsx".format(current_df_title)
            stats_df_mean.to_excel(os.path.join(output_directory3, output_file_name))
        
        print('Export of all figures to file complete!')
        return stats_df_mean, stats_df_replicates      

def clean_the_batch_tuple_list(list_of_clean_dfs):
    '''
    This function performs simple data cleaning operations on 
    batch processed data such that further analysis can be performed equivalently 
    on inputs of batch or individual data formats.
    
    Outputs a list of clean polymer dataframes.
    
    '''
    
    print("Beginning batch data cleaning.")

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

    print("Batch data cleaning completed.") 
    return final_clean_polymer_df_list
