import pytest
import sys
import os
import glob
import shutil

# appending path to access sibling directory
#sys.path.append(os.getcwd() + '/../src')

from discoprocess.data_wrangling_helpers import *

# global testng directories
path = "./test-files/test_helpers"
input_path = path + "/input"
expected_path = path + "/expected"

class TestDrop:
    """ This class contains all the unit tests relating to the helper function DropComplete."""
    
    def test_DropComplete(self):
        """ Tests for three different inputs: two of which are expected to return False, and one where the expected result is True."""
        
        s = "Complete"
        
        assert not DropComplete(s)
        
        t = "complete"
        
        assert not DropComplete(t)

        u = "another word"
        
        assert DropComplete(u)

class TestInitialize:
    """This class contains all the unit tests relating to the function initialize_excel_batch_replicates."""
    
    def test_initialize_excel_batch_replicates(self):
        """ Taking in a filepath to an Excel batch, this function checks for whether the expected polymer names, replicates indices, and sheets are returned."""
        
        b = input_path + "/batch_initialize_input.xlsx"
        
        actual_polymers, actual_replicates, actual_sheets = initialize_excel_batch_replicates(b)
        
        expected_polymers = open(expected_path + "/unique_polymer_output.txt").readlines()
        expected_polymers = [l.rstrip() for l in expected_polymers]
        
        expected_replicates = open(expected_path + "/unique_polymer_replicate_output.txt").readlines()
        expected_replicates = np.array([float(word) for word in expected_replicates])
        
        expected_sheets = open(expected_path + "/all_sheets_output.txt").readlines()
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
        
        b = input_path + "/wrangle_batch_input.xlsx"
        
        name_sheets = ['CMC (2)', 'CMC (3)', 'CMC (4)', 'CMC_ours', 'HEMAcMPC (1)', 'HEMAcMPC (2)', 
                       'HEMAcMPC (3)', 'HEMAcMPC (4)']

        replicate_index = [1, 2, 3, 1, 1, 2, 3, 4]
        
        actual = wrangle_batch(b, name_sheets, replicate_index)
        
        df_names = ['CMC (2)', 'CMC_ours', 'HEMAcMPC (1)'] # correspond to sheet_0.xlsx, sheet_1.xlsx, sheet_2.xlsx in wrangle_batch_output, respectively
        
        dfs = sorted(glob.glob(expected_path + "/wrangle_batch_output/*"), key=lambda x : int(os.path.basename(x)[6:-5])) # sort by numerical order 
        dfs = [pd.read_excel(df, index_col=0) for df in dfs]
        
        expected = []
        
        for i in range(len(df_names)):
            expected.append((df_names[i], dfs[i]))
            
        if len(actual) < len(expected):
            msg1 = "Not enough dataframes appended to output list."
        else:
            msg1 = "Too many dataframes appended to output list."

        assert len(actual) == len(expected), msg1
        
        for i in range(len(actual)):
            msg2 = "Actual title of dataframe: {}\nExpected title of dataframe: {}".format(actual[i][0], expected[i][0])
            assert actual[i][0] == expected[i][0], msg2
            pd.testing.assert_frame_equal(actual[i][1], expected[i][1], check_dtype=False, check_exact=True)
    
    def test_wrangle_book(self):
        """ As part of book initialization, checks for whether the expected dataframes from an Excel book are returned in list format.
        
        Notes
        -----
        Equality checking ignores datatype matching.
        """

        b = input_path + "/wrangle_book_input.xlsx"

        name_sheets = ['Sample', 'Control']
                
        sample_control = ['sample', 'control']

        replicate_index = [1, 1]

        actual = wrangle_book(b, name_sheets, sample_control, replicate_index)
        
        dfs = sorted(glob.glob(expected_path + "/wrangle_book_output/*"), key=lambda x : int(os.path.basename(x)[6:-5])) # sort by numerical order
        dfs = [pd.read_excel(df, index_col=0) for df in dfs]
        
        if len(actual) > len(dfs):
            msg1 = "Too many dataframes appended to output list."
        else:
            msg1 = "Too few dataframes appended to output list."
            
        assert len(actual) == len(dfs), msg1
        
        for i in range(len(actual)):
            pd.testing.assert_frame_equal(actual[i], dfs[i], check_dtype=False, check_exact=True)
    
class TestCount:
    """ This class contains all the unit tests related to the helper function count_sheets."""
    
    def test_count_sheets(self):
        """ Checks for whether the expected number of samples and controls are returned, as well as the expected initializer labels (sameple, control) and their respective indices."""
                
        input_list = ['Sample', 'Sample (2)', 'Sample (3)', 'Control', 'Control (2)', 'Control (3)']
                
        num_samples, num_controls, sample_control_initializer, sample_replicate_initializer, control_replicate_initializer = count_sheets(input_list)

        # CHECKING SAMPLE AND CONTROL COUNTS
        msg1 = "Actual number of samples counted: {}\nExpected number of samples counted: {}".format(num_samples, 3)
        assert num_samples == 3, msg1
        
        msg2 = "Actual number of controls counted: {}\nExpected number of controls counted: {}".format(num_controls, 3)
        assert num_controls == 3, msg2
        
        # CHECKING SAMPLE AND CONTROL INITIALIZER
        expected_sample_control = ['sample', 'sample', 'sample', 'control', 'control', 'control']
                
        msg3 = "Sample control initializer list does not contain expected sequence."
        assert sample_control_initializer == expected_sample_control, msg3

        # CHECKING SAMPLE REPLICATE INDICES
        expected_sample_replicate = [1, 2, 3]
                
        msg4 = "Sample replicate indices are not ordered as expected."
        assert sample_replicate_initializer == expected_sample_replicate, msg4
        
        # CHECKING CONTROL REPLICATE INDICES
        expected_control_replicate = [1, 2, 3]
                
        msg5 = "Control replicate indices are not ordered as expected."
        assert control_replicate_initializer == expected_control_replicate, msg5

class TestCleanBook:
    
    def test_clean_book_list(self):

        dfs = sorted(glob.glob(expected_path + "/wrangle_book_output/*"), key=lambda x : int(os.path.basename(x)[6:-5])) # sort by numerical order
        dfs = [pd.read_excel(df, index_col=0) for df in dfs]
        
        actual = clean_book_list(dfs, "KHA")
        
        expected = pd.read_excel(expected_path + "/book_to_dataframe_output.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

class TestEqualityChecker:
    """ This class contains all the unit tests related to the attenuation equality checkers."""
    
    @pytest.mark.parametrize('path', ['book', 'batch'])
    def test_attenuation_calc_equality_checker(self, path):
        """ Checks whether two dataframes are equal according to the specifications of attenuation calculations; book path."""   
        
        df1 = pd.read_excel(input_path + "/checker1_" + path + ".xlsx", index_col=0)
        df2 = pd.read_excel(input_path + "/checker2_" + path + ".xlsx", index_col=0)    
        
        msg = "Equality checker did not correctly identify the dataframes as equal."
        assert attenuation_calc_equality_checker(df1, df2, path), msg

    def test_corrected_attenuation_calc_equality_checker(self):
        """ Checks whether the conditions for the corrected attenuation calculations are met between three dataframes."""
        
        df1 = pd.read_excel(input_path + "/corr_checker1.xlsx", index_col=0)
        df2 = pd.read_excel(input_path + "/corr_checker2.xlsx", index_col=0)
        df3 = pd.read_excel(input_path + "/corr_checker3.xlsx", index_col=0)
        
        msg = "Equality checker did not correct identify the dataframes as equal."
        assert corrected_attenuation_calc_equality_checker(df1, df2, df3), msg
    
    def test_attenuation_calc_equality_checker_wrong_book_subset(self):
        ''' Checks whether a ValueError is raised when subsets of the true and false irradiated dataframes do not match. '''
        
        df1 = pd.read_excel(input_path + "/checker_book_wrong_subset_true.xlsx", index_col=0)
        df2 = pd.read_excel(input_path + "/checker_book_wrong_subset_false.xlsx", index_col=0)
                
        assert not attenuation_calc_equality_checker(df1, df2)
        
    @pytest.mark.parametrize("filename", [f'att_batch_subset_{i}.xlsx' for i in range(1, 5)])
    def test_attenuation_calc_equality_checker_wrong_batch_subsets(self, filename):
        ''' Checks whether a ValueError is raised when the sample_or_control, replicate, proton_peak_index, and sat_time values in the true and false irradiated dataframes do not match. '''
        
        df1 = pd.read_excel(input_path + "/true_" + filename, index_col=0)
        df2 = pd.read_excel(input_path + "/false_" + filename, index_col=0)

        assert not attenuation_calc_equality_checker(df1, df2, 'batch')
        
    @pytest.mark.parametrize("filename", [f'att_book_true_subset{i}.xlsx' for i in range(1, 4)])
    def test_corrected_attenuation_calc_equality_checker_wrong_subsets(self, filename):
        ''' Checks whether a ValueError is raised when the replicate, 
        sat_time values, and concentration values in three subset dataframes do not match. '''
        
        df1 = pd.read_excel(input_path + "/1_" + filename, index_col=0)
        df2 = pd.read_excel(input_path + "/2_" + filename, index_col=0)
        df3 = pd.read_excel(input_path + "/3_" + filename, index_col=0)
            
        assert not corrected_attenuation_calc_equality_checker(df1, df2, df3)
    
    # testing assertions
    def test_attenuation_calc_equality_checker_unequal_shape(self):
        ''' Checks whether a ValueError is raised when the number of true and false irradiated samples in dataframe do not match (i.e. missing data). 
        
        Notes
        -----
        The values in the dataframes passed to attenuation_calc_equality_checker don't matter, as long as the shapes are different so that the assertion is raised.
        '''
        
        df1 = pd.DataFrame({1:2, 2:4, 3:2}, index=(1, 2, 3)) 
        df2 = pd.DataFrame({1:3, 2:4}, index=(1, 2))
        
        with pytest.raises(ValueError) as e:
            attenuation_calc_equality_checker(df1, df2)
        
        assert e.match("Error, irrad_false and irrad_true dataframes are not the same shape to begin with.")
      
    def test_corrected_attenuation_calc_equality_checker_unequal_shape(self):
        ''' Checks whether a ValueError is raised when the number of true and false irradiated samples in dataframe do not match (i.e. missing data). 
        
        Notes
        -----
        The values in the dataframes passed to attenuation_calc_equality_checker don't matter, as long as the shapes are different so that the assertion is raised.
        '''
        
        df1 = pd.DataFrame({1:2, 2:4, 3:2}, index=(1, 2, 3)) 
        df2 = pd.DataFrame({1:3, 2:4}, index=(1, 2))
        df3 = pd.DataFrame({1:2, 2:4, 3:2}, index=(1, 2, 3)) 
        
        with pytest.raises(ValueError) as e:
            corrected_attenuation_calc_equality_checker(df1, df2, df3)
        
        assert e.match("Error, corrected % attenuation input dataframes are not the same shape to begin with.")
           
class TestDofs:
    """ This class contains all the unit tests related to the function get_dofs."""
    
    def test_get_dofs(self):
        """ Given a list of peaks, checks for whether the expected dofs are returned."""
        
        peak_list = open(input_path + "/dof_input.txt").readlines()    
        peak_list = [int(word.rstrip()) for word in peak_list]
        
        peak_input = np.array(peak_list)
        
        actual = get_dofs(peak_input)
        
        expected = open(expected_path + "/dof_output.txt").readlines()    
        expected = [int(word.rstrip()) for word in expected]
        
        msg = "Dofs were not identifed as expected."
        assert actual == expected, msg
