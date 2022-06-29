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

def name_checker(list_of_raw_books):
    '''
        This function checks to ensure that the polymer naming is consistent across input files and returns a descriptive error
        message for inconsistent naming

        Input:

            List of all raw books to be used for the data PROCESSING

        Output:

            Will return 1 if all data is input correctly, raise an error otherwise.
    '''

    for book in list_of_raw_books:

        unique_polymers, unique_polymer_replicates, name_sheets = initialize_excel_batch_replicates(book)

        for name in unique_polymers:

            list_of_name_parts = name.split("_")

            if list_of_name_parts[1][-1] != 'k':
                print(name, "Please check this polymer!")
                raise Exception("Please add a 'k' to the end of the molecular weight (for kilodaltons) for each polymer name, specifically for " + name)
                return 0
            if list_of_name_parts[2][-2:] != 'uM':
                print(name, "Please check this polymer!")
                raise Exception("Please add 'uM' to the end of the concentration in the polymer name, specifically for " + name)
                return 0
            if len(list_of_name_parts) != 3:
                print(name, "Please check this polymer!")
                raise Exception("Please format the name in the form: CMC_90k_20uM, check " + name)
                return 0

    return True

def resonance_and_column_checker(list_of_raw_books):
    '''
    This function checks the on and off resonances in the input tables, and also checks to make sure the BSM and CONTROL columns are in the correct places

    Input:
        list_of_raw_books - a list of excel books to be checked

    Output:
        Nothing if all clear, an error message if a check raises a flags

    '''

    for b in list_of_raw_books:

            current_book_title = os.path.basename(str(b))

            # return list of unique polymer names in the book, their replicates, and the sheets containing raw data to loop through
            unique_polymers, unique_polymer_replicates, name_sheets = initialize_excel_batch_replicates(b)

            # generate a new replicate index list that holds the nth replicate associated with each raw data sheet in book
            replicate_index = []

            for i in range(len(unique_polymer_replicates)):
                current_replicate_range = range(1,int(unique_polymer_replicates[i]+1),1)
                for j in current_replicate_range:
                    replicate_index.append(j)

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

                count = 0

                if len(sheet_coords_list) % 4 != 0 and len(sheet_coords_list) >= 4:
                    print(current_polymer_name, "Please check this sheet!")

                    raise Exception("In the excel book " + current_book_title +  " please ensure the last table is on the off resonance and the first is on the on resonance and the range keyword is only used for relevant data tables in sheet " + current_polymer_name)

                for i in range(0, len(sheet_coords_list), 2):

                    if count % 2 == 0:

                        if current_sheet_df.iloc[(sheet_coords_list[i][0]) - 1, (sheet_coords_list[i][1])] != "On" or current_sheet_df.iloc[(sheet_coords_list[i+1][0]) - 1, (sheet_coords_list[i+1][1])] != "On":
                            print(current_polymer_name, "Please check this sheet!")

                            raise Exception("In the excel book " + current_book_title +  " please ensure that all odd replicates are On resonance and Range keyword is only used in tables meant for data analysis in sheet " + current_polymer_name)

                    else:

                        if current_sheet_df.iloc[(sheet_coords_list[i][0]) - 1, (sheet_coords_list[i][1])] != "Off" or current_sheet_df.iloc[(sheet_coords_list[i+1][0]) - 1, (sheet_coords_list[i+1][1])] != "Off":
                            print(current_polymer_name, "Please check this sheet!")
                            raise Exception("In the excel book " + current_book_title +  " please ensure that all even replicates are Off resonance and Range keyword is only used in tables meant for data analysis in sheet " + current_polymer_name)

                    if current_sheet_df.iloc[(sheet_coords_list[i][0]) - 1, (sheet_coords_list[i][1]) + 1] != "Control" or current_sheet_df.iloc[(sheet_coords_list[i + 1][0]) - 1, (sheet_coords_list[i + 1][1]) + 1] != "BSM":
                        print(current_polymer_name, "Please check this sheet!")
                        raise Exception("In the excel book " + current_book_title +  " please ensure Control column on the left and BSM column on the right in sheet" + current_polymer_name)


                    count += 1
    return True

def range_checker(list_of_raw_books):
    '''
    Checks to make sure the ranges are the same for all tables in the excel files

    Input:
        list_of_raw_books - list containing excel books to be PROCESSED

    Output:
        Nothing if all clear, an error if the ranges dont match across all tables.

    '''

    for b in list_of_raw_books:

            current_book_title = os.path.basename(str(b))

            # return list of unique polymer names in the book, their replicates, and the sheets containing raw data to loop through
            unique_polymers, unique_polymer_replicates, name_sheets = initialize_excel_batch_replicates(b)

            # generate a new replicate index list that holds the nth replicate associated with each raw data sheet in book
            replicate_index = []

            for i in range(len(unique_polymer_replicates)):
                current_replicate_range = range(1,int(unique_polymer_replicates[i]+1),1)
                for j in current_replicate_range:
                    replicate_index.append(j)

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

                coordinate_0 = sheet_coords_list[0]

                i = 1
                track = 1

                ranges = []

                while ".." in current_sheet_df.iloc[(coordinate_0[0] + i), coordinate_0[1]]:

                    ranges.append(current_sheet_df.iloc[(coordinate_0[0] + i), coordinate_0[1]])

                    i += 1
                    track += 1



                for coord in sheet_coords_list:

                    i = 1

                    new_ranges = []

                    while i < track:

                        new_ranges.append(current_sheet_df.iloc[(coord[0] + i), coord[1]])

                        i += 1

                    if new_ranges != ranges:
                        print(current_polymer_name, "Please check this sheet!")
                        raise Exception("In the excel book " + current_book_title +  " please ensure the ranges are equivalent across all tables in sheet " + current_polymer_name)


    return True
