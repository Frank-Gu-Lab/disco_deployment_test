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
        
        path = "./test-files/test_helpers/"

        b = path + "/input/batch_initialize_input.xlsx"
        
        actual_polymers, actual_replicates, actual_sheets = initialize_excel_batch_replicates(b)
        
        expected_polymers = open(path + "/expected/unique_polymer_output.txt").readlines()
        expected_polymers = [l.rstrip() for l in expected_polymers]
        
        expected_replicates = open(path + "/expected/unique_polymer_replicate_output.txt").readlines()
        expected_replicates = np.array([float(word) for word in expected_replicates])
        
        expected_sheets = open(path + "/expected/all_sheets_output.txt").readlines()
        expected_sheets = [l.rstrip() for l in expected_sheets]
        
        assert actual_polymers == expected_polymers
        assert np.array_equal(actual_replicates, expected_replicates)
        assert actual_sheets == expected_sheets
    
class TestWrangle:
    """ This class contains all the unit tests relating to the wrangle functions."""
    
    def test_wrangle_batch(self):
        """ As part of batch initialization, checks for whether the expected polymers and its associated dataframes are returned in a list format."""
        
        path = "./test-files/test_helpers"
        
        b = path + "/input/wrangle_batch_input.xlsx"
        
        name_sheets = open(path + "/input/wrangle_batch_names.txt").readlines()
        name_sheets = [l.rstrip() for l in name_sheets]

        replicate_index = open(path + "/input/wrangle_batch_index.txt").readlines()
        replicate_index = [int(l.rstrip()) for l in replicate_index]
        
        df_list = wrangle_batch(b, name_sheets, replicate_index)
        
        df_names = open(path + "/expected/wrangle_batch_output.txt").readlines()
        df_names = [l.rstrip() for l in df_names]
        
        dfs = sorted(glob.glob(path + "/expected/wrangle_batch_output/*"), key=os.path.getmtime)
        dfs = [pd.read_excel(df, index_col=0) for df in dfs]
        
        actual = []
        
        for i in range(len(df_names)):
            actual.append((df_names[i], dfs[i]))
            
        assert len(df_list) == len(actual)
        
        for i in range(len(df_list)):
            assert df_list[i][0] == actual[i][0]
            pd.testing.assert_frame_equal(df_list[i][1], actual[i][1], check_dtype=False)
        
    def test_wrangle_book(self):
        """ As part of book initialization, checks for whether the expected dataframes from an Excel book are returned in list format."""
        
        path = "./test-files/test_helpers"
        
        b = path + "/input/wrangle_book_input.xlsx"
        
        name_sheets = open(path + "/input/wrangle_book_names.txt").readlines()
        name_sheets = [l.rstrip() for l in name_sheets]
        
        sample_control = open(path + "/input/wrangle_book_sample.txt").readlines()
        sample_control = [l.rstrip() for l in sample_control]
        
        replicate_index = open(path + "/input/wrangle_book_index.txt").readlines()
        replicate_index = [int(l.rstrip()) for l in replicate_index]
        
        df_list = wrangle_book(b, name_sheets, sample_control, replicate_index)
        
        dfs = sorted(glob.glob(path + "/expected/wrangle_book_output/*"), key=os.path.getmtime)
        dfs = [pd.read_excel(df, index_col=0) for df in dfs]
        
        assert len(df_list) == len(dfs)
        for i in range(len(df_list)):
            pd.testing.assert_frame_equal(df_list[i], dfs[i], check_dtype=False)

class TestCount:
    """ This class contains all the unit tests related to the helper function count_sheets."""
    
    def test_count(self):
        """ Checks for whether the expected number of samples and controls are returned, as well as the expected initializer labels (sameple, control) and their respective indices."""
        
        path = "./test-files/test_helpers"
        
        input_list = []
        
        with open(path + "/input/count_input.txt") as file:
            
            for line in file.readlines():
                input_list.append(line.rstrip())
                
        num_samples, num_controls, sample_control_initializer, sample_replicate_initializer, control_replicate_initializer = count_sheets(input_list)

        assert num_samples == 3
        assert num_controls == 3
        
        expected_sample_control = []
        
        with open(path + "/input/sample_control.txt") as file:
            for line in file.readlines():
                expected_sample_control.append(line.rstrip())
                
        assert sample_control_initializer == expected_sample_control

        expected_sample_replicate = []

        with open(path + "/input/sample_replicate.txt") as file:
            for line in file.readlines():
                expected_sample_replicate.append(int(line.rstrip()))
                
        assert sample_replicate_initializer == expected_sample_replicate
        
        expected_control_replicate = []
        
        with open(path + "/input/control_replicate.txt") as file:
            for line in file.readlines():
                expected_control_replicate.append(int(line.rstrip()))
                
        assert control_replicate_initializer == expected_control_replicate

class TestEqualityChecker:
    """ This class contains all the unit tests related to the attenuation equality checkers."""
    
    def test_checker_book(self):
        """ Checks whether two dataframes are equal according to the specifications of attenuation calculations; book path."""
        
        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/checker1_book.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/checker2_book.xlsx", index_col=0)    
        
        assert attenuation_calc_equality_checker(df1, df2)
        
    def test_checker_batch(self):
        """ Checks whether two dataframes are equal according to the specifications of attenuation calculations; batch path."""      
        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/checker1_batch.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/checker2_batch.xlsx", index_col=0)    
        
        assert attenuation_calc_equality_checker(df1, df2, 'batch')
    
    def test_corrected_checker(self):
        """ Checks whether the conditions for the corrected attenuation calculations are met between three dataframes."""
        
        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/corr_checker1.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/corr_checker2.xlsx", index_col=0)
        df3 = pd.read_excel(path + "/corr_checker3.xlsx", index_col=0)
        
        assert corrected_attenuation_calc_equality_checker(df1, df2, df3)
    
class TestDofs:
    """ This class contains all the unit tests related to the function get_dofs."""
    
    def test_getdofs(self):
        """ Checks for whether the expected dofs are returned, given a list of peaks."""
        
        path = "./test-files/test_helpers"

        peak_list = open(path + "/input/dof_input.txt").readlines()    
        peak_list = [int(word.rstrip()) for word in peak_list]
        
        peak_input = np.array(peak_list)
        
        actual = get_dofs(peak_input)
        
        expected = open(path + "/expected/dof_output.txt").readlines()    
        expected = [int(word.rstrip()) for word in expected]
        
        assert actual == expected
