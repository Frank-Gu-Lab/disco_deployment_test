import pytest
import sys
import os
import glob
import shutil

# appending path to access sibling directory
sys.path.append(os.getcwd() + '\\..\\src')

from data_wrangling_functions import *

class TestDrop:
    """ This class contains all the unit tests relating to the helper function DropComplete."""
    
    def test_drop(self):
        """ Tests for three different inputs: two of which are expected to return False, and one where the expected result is True."""
        
        s = "Complete"
        
        assert not DropComplete(s)
        
        t = "complete"
        
        assert not DropComplete(t)

        u = "another word"
        
        assert DropComplete(u)

class TestInitialize:
    """This class contains all the unit tests relating to the function initialize_excel_batch_replicates."""
    
    def test_batch_initialize(self):
        """ Taking in a filepath to an Excel batch, this function checks for whether the expected polymer names, replicates indices, and sheets are returned."""
        
        path = "./test-files/test_helpers"
        b = path + "/input/batch_initialize_input.xlsx"
        
        actual_polymers, actual_replicates, actual_sheets = initialize_excel_batch_replicates(b)
        
        expected_polymers = open(path + "/expected/unique_polymer_output.txt").readlines()
        expected_polymers = [l.rstrip() for l in expected_polymers]
        
        expected_replicates = open(path + "/expected/unique_polymer_replicate_output.txt").readlines()
        expected_replicates = np.array([float(word) for word in expected_replicates])
        
        expected_sheets = open(path + "/expected/all_sheets_output.txt").readlines()
        expected_sheets = [l.rstrip() for l in expected_sheets]
        
        msg1 = "Polymer names were not extracted as expected."
        assert actual_polymers == expected_polymers, msg1
        
        msg2 = "Replicate indices were not extracted as expected."
        assert np.array_equal(actual_replicates, expected_replicates), msg2
        
        msg3 = "Excel sheets were not extracted as expected."
        assert actual_sheets == expected_sheets, msg3
    
class TestWrangle:
    """ This class contains all the unit tests relating to the wrangle functions."""
    
    def test_wrangle_batch(self):
        """ As part of batch initialization, checks for whether the expected polymers and its associated dataframes are returned in a list format.
        
        Notes
        -----
        Equality checking ignores datatype matching.
        """
        
        path = "./test-files/test_helpers"
        b = path + "/input/wrangle_batch_input.xlsx"
        
        name_sheets = open(path + "/input/wrangle_batch_names.txt").readlines()
        name_sheets = [l.rstrip() for l in name_sheets]

        replicate_index = open(path + "/input/wrangle_batch_index.txt").readlines()
        replicate_index = [int(l.rstrip()) for l in replicate_index]
        
        df_list = wrangle_batch(b, name_sheets, replicate_index)
        
        df_names = open(path + "/expected/wrangle_batch_output.txt").readlines()
        df_names = [l.rstrip() for l in df_names]
        
        dfs = sorted(glob.glob(path + "/expected/wrangle_batch_output/*"), key=lambda x : int(os.path.basename(x)[6:-5])) # sort by numerical order 
        dfs = [pd.read_excel(df, index_col=0) for df in dfs]
        
        actual = []
        
        for i in range(len(df_names)):
            actual.append((df_names[i], dfs[i]))
            
        msg1 = "Too many or too few dataframes appended to output list."
        assert len(df_list) == len(actual), msg1
        
        for i in range(len(df_list)):
            msg2 = "Actual title of dataframe: {}\nExpected title of dataframe: {}".format(df_list[i][0], actual[i][0])
            assert df_list[i][0] == actual[i][0], msg2
            pd.testing.assert_frame_equal(df_list[i][1], actual[i][1], check_dtype=False, check_exact=True)
        
    def test_wrangle_book(self):
        """ As part of book initialization, checks for whether the expected dataframes from an Excel book are returned in list format.
        
        Notes
        -----
        Equality checking ignores datatype matching.
        """
        
        path = "./test-files/test_helpers"
        b = path + "/input/wrangle_book_input.xlsx"
        
        name_sheets = open(path + "/input/wrangle_book_names.txt").readlines()
        name_sheets = [l.rstrip() for l in name_sheets]
        
        sample_control = open(path + "/input/wrangle_book_sample.txt").readlines()
        sample_control = [l.rstrip() for l in sample_control]
        
        replicate_index = open(path + "/input/wrangle_book_index.txt").readlines()
        replicate_index = [int(l.rstrip()) for l in replicate_index]
        
        df_list = wrangle_book(b, name_sheets, sample_control, replicate_index)
        
        dfs = sorted(glob.glob(path + "/expected/wrangle_book_output/*"), key=lambda x : int(os.path.basename(x)[6:-5])) # sort by numerical order
        dfs = [pd.read_excel(df, index_col=0) for df in dfs]
        
        msg1 = "Too many or too few dataframes appended to output list."
        assert len(df_list) == len(dfs), msg1
        
        for i in range(len(df_list)):
            pd.testing.assert_frame_equal(df_list[i], dfs[i], check_dtype=False, check_exact=True)

class TestCount:
    """ This class contains all the unit tests related to the helper function count_sheets."""
    
    def test_count(self):
        """ Checks for whether the expected number of samples and controls are returned, as well as the expected initializer labels (sameple, control) and their respective indices."""
        
        path = "./test-files/test_helpers"
        
        input_list = []
        
        # recreating input list
        with open(path + "/input/count_input.txt") as file:
            for line in file.readlines():
                input_list.append(line.rstrip())
                
        num_samples, num_controls, sample_control_initializer, sample_replicate_initializer, control_replicate_initializer = count_sheets(input_list)

        # CHECKING SAMPLE AND CONTROL COUNTS
        msg1 = "Actual number of samples counted: {}\nExpected number of samples counted: {}".format(num_samples, 3)
        assert num_samples == 3, msg1
        
        msg2 = "Actual number of controls counted: {}\nExpected number of controls counted: {}".format(num_controls, 3)
        assert num_controls == 3, msg2
        
        # CHECKING SAMPLE AND CONTROL INITIALIZER
        expected_sample_control = []
        
        with open(path + "/input/sample_control.txt") as file:
            for line in file.readlines():
                expected_sample_control.append(line.rstrip())
                
        msg3 = "Sample control initializer list does not contain expected sequence."
        assert sample_control_initializer == expected_sample_control, msg3

        # CHECKING SAMPLE REPLICATE INDICES
        expected_sample_replicate = []

        with open(path + "/input/sample_replicate.txt") as file:
            for line in file.readlines():
                expected_sample_replicate.append(int(line.rstrip()))
                
        msg4 = "Sample replicate indices are not ordered as expected."
        assert sample_replicate_initializer == expected_sample_replicate, msg4
        
        # CHECKING CONTROL REPLICATE INDICES
        expected_control_replicate = []
        
        with open(path + "/input/control_replicate.txt") as file:
            for line in file.readlines():
                expected_control_replicate.append(int(line.rstrip()))
                
        msg5 = "Control replicate indices are not ordered as expected."
        assert control_replicate_initializer == expected_control_replicate, msg5

class TestEqualityChecker:
    """ This class contains all the unit tests related to the attenuation equality checkers."""
    
    def test_checker_book(self):
        """ Checks whether two dataframes are equal according to the specifications of attenuation calculations; book path."""
        
        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/checker1_book.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/checker2_book.xlsx", index_col=0)    
        
        msg = "Equality checker did not correctly identify the dataframes as equal."
        assert attenuation_calc_equality_checker(df1, df2), msg
        
    def test_checker_batch(self):
        """ Checks whether two dataframes are equal according to the specifications of attenuation calculations; batch path."""      
        
        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/checker1_batch.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/checker2_batch.xlsx", index_col=0)    
        
        msg = "Equality checker did not correctly identify the dataframes as equal."
        assert attenuation_calc_equality_checker(df1, df2, 'batch'), msg
    
    def test_corrected_checker(self):
        """ Checks whether the conditions for the corrected attenuation calculations are met between three dataframes."""
        
        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/corr_checker1.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/corr_checker2.xlsx", index_col=0)
        df3 = pd.read_excel(path + "/corr_checker3.xlsx", index_col=0)
        
        msg = "Equality checker did not correct identify the dataframes as equal."
        assert corrected_attenuation_calc_equality_checker(df1, df2, df3), msg
    
class TestDofs:
    """ This class contains all the unit tests related to the function get_dofs."""
    
    def test_getdofs(self):
        """ Given a list of peaks, checks for whether the expected dofs are returned."""
        
        path = "./test-files/test_helpers"

        peak_list = open(path + "/input/dof_input.txt").readlines()    
        peak_list = [int(word.rstrip()) for word in peak_list]
        
        peak_input = np.array(peak_list)
        
        actual = get_dofs(peak_input)
        
        expected = open(path + "/expected/dof_output.txt").readlines()    
        expected = [int(word.rstrip()) for word in expected]
        
        msg = "Dofs were not identifed as expected."
        assert actual == expected, msg
