import pytest
import sys
import os
import glob
import shutil

# appending path to access sibling directory - uncomment if local package setup doesn't work
sys.path.append(os.getcwd() + '/../src')

from discoprocess.data_wrangling_helpers import *

# global testng directories
path = os.path.dirname(__file__) + "/test-files/test_helpers"
input_path = path + "/input"
expected_path = path + "/expected"

class TestInitialize:
    """This class contains all the unit tests relating to the function initialize_excel_batch_replicates."""

    def test_initialize_excel_batch_replicates(self):
        """ Taking in a filepath to an Excel batch, this function checks for whether the expected polymer names, replicates indices, and sheets are returned."""

        b = input_path + "/batch_initialize_input.xlsx"

        actual_polymers, actual_replicates, actual_sheets = initialize_excel_batch_replicates(b)

        expected_polymers = ['CMC', 'CMC_ours', 'HEMAcMPC', 'HPMC E3', 'HPMC E4M', 'PDMA', 'PDMAcd', 'PEGHCO', 'PTA']

        expected_replicates = np.array([3, 1, 4, 3, 3, 3, 3, 1, 3])

        expected_sheets = ['CMC (2)', 'CMC (3)', 'CMC (4)', 'CMC_ours', 'HEMAcMPC (1)', 'HEMAcMPC (2)', 'HEMAcMPC (3)',
                            'HEMAcMPC (4)', 'HPMC E3', 'HPMC E3 (2)', 'HPMC E3 (3)', 'HPMC E4M', 'HPMC E4M (2)', 'HPMC E4M (3)', 'PDMA (1)',
                            'PDMA (2)', 'PDMA (3)', 'PDMAcd (1)', 'PDMAcd (2)', 'PDMAcd (3)', 'PEGHCO', 'PTA (1)', 'PTA (2)', 'PTA (3)']

        msg1 = "Polymer names were not extracted as expected."
        assert actual_polymers == expected_polymers, msg1

        msg2 = "Replicate indices were not extracted as expected."
        assert np.array_equal(actual_replicates, expected_replicates), msg2

        msg3 = "Excel sheets were not extracted as expected."
        assert actual_sheets == expected_sheets, msg3

    #New test for grab_conc that Matthew implemented (that's me!)
    def test_grab_conc(self):

        test_strings = ["CMC_131k_20uM (2)", "Batch-HPC_80k_20uM", "Batch-PVA99_85-105k_20uM", "Batch-PVP_55k_20uM"]

        for string in test_strings:
            assert grab_conc(string) == 20

class TestWrangle:
    """ This class contains all the unit tests relating to the wrangle functions."""

    def test_wrangle_batch(self):
        """ As part of batch initialization, checks for whether the expected polymers and its associated dataframes are returned in a list format.

        Notes
        -----
        Equality checking ignores datatype matching.
        """

        b = input_path + "/wrangle_batch_input.xlsx"

        name_sheets = ['CMC_9uM (2)', 'CMC_9uM (3)', 'CMC_9uM (4)', 'CMC_ours_9uM', 'HEMAcMPC_9uM (1)', 'HEMAcMPC_9uM (2)',
                       'HEMAcMPC_9uM (3)', 'HEMAcMPC_9uM (4)']

        replicate_index = [1, 2, 3, 1, 1, 2, 3, 4]

        actual = wrangle_batch(b, name_sheets, replicate_index)

        df_names = ['CMC_9uM (2)', 'CMC_ours_9uM', 'HEMAcMPC_9uM (1)'] # correspond to sheet_0.xlsx, sheet_1.xlsx, sheet_2.xlsx in wrangle_batch_output, respectively

        dfs = sorted(glob.glob(expected_path + "/wrangle_batch_output/*"), key=lambda x : int(os.path.basename(x)[6:-5])) # sort by numerical order
        dfs = [pd.read_excel(df, index_col=0) for df in dfs]

        expected = []

        for i in range(len(df_names)):
            expected.append((df_names[i], dfs[i]))

        msg1 = "{} dataframes were initialized, expected {}.".format(len(actual), len(expected))
        assert len(actual) == len(expected), msg1

        for i in range(len(actual)):
            msg2 = "Actual title of dataframe: {}\nExpected title of dataframe: {}".format(actual[i][0], expected[i][0])
            assert actual[i][0] == expected[i][0], msg2
            pd.testing.assert_frame_equal(actual[i][1], expected[i][1], check_dtype=False, check_exact=True)



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


class TestEqualityChecker:
    """ This class contains all the unit tests related to the attenuation equality checkers."""

    @pytest.mark.parametrize('path', ['batch'])
    def test_attenuation_calc_equality_checker(self, path):
        """ Checks whether two dataframes are equal according to the specifications of attenuation calculations; book path."""

        df1 = pd.read_excel(input_path + "/checker1_" + path + ".xlsx", index_col=0)
        df2 = pd.read_excel(input_path + "/checker2_" + path + ".xlsx", index_col=0)

        msg = "Equality checker did not correctly identify the dataframes as equal."
        assert attenuation_calc_equality_checker(df1, df2), msg

    def test_corrected_attenuation_calc_equality_checker(self):
        """ Checks whether the conditions for the corrected attenuation calculations are met between three dataframes."""

        df1 = pd.read_excel(input_path + "/corr_checker1.xlsx", index_col=0)
        df2 = pd.read_excel(input_path + "/corr_checker2.xlsx", index_col=0)
        df3 = pd.read_excel(input_path + "/corr_checker3.xlsx", index_col=0)

        msg = "Equality checker did not correct identify the dataframes as equal."
        assert corrected_attenuation_calc_equality_checker(df1, df2, df3), msg

    @pytest.mark.parametrize("filename", [f'att_batch_subset_{i}.xlsx' for i in range(1, 5)])
    def test_attenuation_calc_equality_checker_wrong_batch_subsets(self, filename):
        ''' Checks whether a ValueError is raised when the sample_or_control, replicate, proton_peak_index, and sat_time values in the true and false irradiated dataframes do not match. '''

        df1 = pd.read_excel(input_path + "/true_" + filename, index_col=0)
        df2 = pd.read_excel(input_path + "/false_" + filename, index_col=0)

        assert not attenuation_calc_equality_checker(df1, df2)

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

        peak_list = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3,
                        3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4,
                        4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                        7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                        1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]

        peak_input = np.array(peak_list)

        actual = get_dofs(peak_input)

        expected = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        msg = "Dofs were not identifed as expected."
        assert actual == expected, msg

    #Hey it works now :D
    def test_get_dofs_one_peak(self):

        df = pd.read_pickle("C:/Users/matth/OneDrive/Documents/GitHub/disco-data-processing/tests/test-files/test_helpers/input/get_dofs_one_peak_input.pkl")

        assert get_dofs_one_peak(df) == [2, 2, 2, 2, 2, 2, 2]

class TestCleansing:

    def test_DropComplete(self):

        false_string = "I am now complete, perhaps even immortal"

        true_string = "I am not yet ready to battle my demons"

        assert DropComplete(false_string) == False

        assert DropComplete(true_string) == True

    def test_clean_string(self):

        input_string = "()(((()()()))))() )))(A)"

        actual = clean_string(input_string)

        expected = "a"

        assert actual == expected
