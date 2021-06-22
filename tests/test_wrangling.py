import pytest
import sys
import os
import glob
import shutil
import matplotlib.pyplot as plt

# appending path to access sibling directory
sys.path.append(os.getcwd() + '/../src')

from data_wrangling_functions import *
from matplotlib.testing.compare import compare_images

# global testing directories
path = "./test-files/test_wrangling"
input_path = path + "/input"
expected_path = path + "/expected"

@pytest.fixture(scope='function')
def remove():
    
    output_dir = path + "/output"
    os.mkdir(output_dir)
    
    yield output_dir
    
    shutil.rmtree(output_dir)

class TestDataFrameConversion:
    """This class contains all the unit tests relating to the dataframe conversion functions, batch_to_dataframe and book_to_dataframe."""
    
    def test_batch(self):
        """Testing overall functionality. Takes in a batch Excel sheet and converts each sheet into a dataframe, returning a tuple of the form
        (df_name, df).
        
        Notes
        -----
        Equality checking ignores datatype matching.
        """
        
        batch = input_path + "/batch_to_dataframe_input.xlsx"
        
        # loop through names and assert equality of dataframes
        
        actual = batch_to_dataframe(batch)
        file = open(expected_path + "/batch_to_dataframe_output.txt")
        
        for i in range(len(actual)):
            
            # testing equality of sheet names
            expected_name = file.readline().rstrip()
            msg1 = "Actual name of sheet: {}, Expected name of sheet: {}".format(actual[i][0], expected_name)
            assert actual[i][0] == expected_name, msg1
            
            # testing equality of dataframes
            name = f"sheet_{i}"
            expected_df = pd.read_excel(expected_path + "/batch_to_dataframe/" + name + ".xlsx", index_col=0)
            pd.testing.assert_frame_equal(actual[i][1], expected_df, check_dtype=False, check_exact=True)

    def test_book_mock(self):
        """ Testing overall functionality. Takes in a book Excel sheet and converts it into a dataframe. The created 
        excel sheet is removed during teardown.
        
        Notes
        -----
        The equality check ignores datatype matching.
        """

        book = input_path + "/KHA.xlsx"
        
        actual = book_to_dataframe(book)
        actual_title = actual[0]
        actual_df = actual[1]
        
        expected_title = "KHA"
        expected_df = pd.read_excel(expected_path + "/book_to_dataframe_output.xlsx", index_col=0)
        
        msg = "Actual title: {}, Expected title: {}".format(actual_title, expected_title)
        assert actual_title == expected_title, msg
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False, check_exact=True)

    # testing assertions
    def test_book_unequal(self):
        ''' Checks whether a ValueError is raised when the number of samples and controls do not match. '''
        
        book = input_path + "/KHA_modified.xlsx" # Control (2) deleted
        
        with pytest.raises(ValueError) as e:
            raise book_to_dataframe(book)
        
        assert e.match('ERROR: The number of sample sheets is not equal to the number of control sheets in {} please confirm the data in the book is correct.'.format(book))
                
class TestCleanBatch:
    """ This class contains all the unit tests relating ti the function test_clean_batch_list. """
    
    def test_clean_batch_list(self):
        """ This function recreates the input dataframe list and checks whether the resulting cleaned dataframes match the expected results. 
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5 and ignores datatype matching.
        """
        
        input_list = glob.glob(input_path + "/clean_batch_input/*")
        
        # recreating input list
        clean_dfs = []
        
        for file in input_list:
            
            name = os.path.basename(file)
            name = name.split(sep=".")
            name = name[0]       
            
            df = pd.read_excel(file, index_col=0)
            
            clean_dfs.append((name, df))
            
        clean_dfs = [clean_dfs]
        
        actual = clean_the_batch_tuple_list(clean_dfs)
        
        output_list = glob.glob(expected_path + "/clean_batch_output/*")
        
        msg1 = "Too many or too few dataframes were cleaned and exported."
        assert len(actual) == len(output_list)
        
        # loop through and check dataframe equality
        for i in range(len(actual)):
            actual_df = actual[i]
            expected_df = pd.read_excel(output_list[i], index_col=0)
            pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)

class TestExport:
    
    def test_export_book(self, remove):
        
        name = 'test'
        df = pd.DataFrame({'a':1, 'b':2}, index=['a','b'])
        output_dir = remove
        
        export_clean_books(name, df, output_dir)
        
        assert os.path.isfile(output_dir + "/" + name + "/" + "test_clean_raw_df.xlsx"), "Exported file could not be found."
        
class TestAttenuation:
    """This class contains all the unit tests relating to the add_attenuation and add_corr_attenuation functions."""
    
    def test_add_attenuation_batch(self, mocker):
        """ Checks for expected attenuation calculations applied in the batch path.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5.
        The function attenuation_calc_equality_checker is mocked to return True.
        """
        
        df = pd.read_excel(input_path + "/att_batch_input.xlsx", index_col=0)
           
        mocker.patch("data_wrangling_functions.attenuation_calc_equality_checker", return_value=True)
        
        actual_true, actual_false = add_attenuation(df, 'batch')
        
        expected_true = pd.read_excel(expected_path + "/att_batch_true_output.xlsx", index_col=0)
        expected_false = pd.read_excel(expected_path + "/att_batch_false_output.xlsx", index_col=0)
                
        pd.testing.assert_frame_equal(actual_true, expected_true)
        pd.testing.assert_frame_equal(actual_false, expected_false)
    
    def test_add_attenuation_book(self, mocker):
        """ Checks for expected attenuation calculations applied in the book path.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5.
        The function attenuation_calc_equality_checker is mocked to return True.
        """

        df = pd.read_excel(input_path + "/att_book_input.xlsx", index_col=0)
        
        mocker.patch("data_wrangling_functions.attenuation_calc_equality_checker", return_value=True)
        
        actual_true, actual_false = add_attenuation(df)
        expected_true = pd.read_excel(expected_path + "/att_book_true.xlsx", index_col=0)
        expected_false = pd.read_excel(expected_path + "/att_book_false.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual_true, expected_true)
        pd.testing.assert_frame_equal(actual_false, expected_false)
            
    def test_add_corr_attenuation_batch(self, mocker):
        """ Checks for expected corrected attenuation calculations applied in the batch path.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5.
        The function corrected_attenuation_calc_equality_checker is mocked to return True.
        """  
        
        df_true = pd.read_excel(input_path + "/corr_att_batch_true_input.xlsx", index_col=0)
        df_false = pd.read_excel(input_path + "/corr_att_batch_false_input.xlsx", index_col=0)
        
        mocker.patch("data_wrangling_functions.corrected_attenuation_calc_equality_checker", return_value=True)
               
        actual = add_corr_attenuation(df_true, df_false, 'batch')
        expected = pd.read_excel(expected_path + "/corr_att_batch_output.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual, expected)
    
    def test_add_corr_attenuation_book(self, mocker):
        """ Checks for expected corrected attenuation calculations applied in the book path.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5.
        The function attenuation_calc_equality_checker is mocked to return True.
        """
        
        df_true = pd.read_excel(input_path + "/att_book_true.xlsx", index_col=0)
        df_false = pd.read_excel(input_path + "/att_book_false.xlsx", index_col=0)
        
        mocker.patch("data_wrangling_functions.corrected_attenuation_calc_equality_checker", return_value=True)
        
        actual = add_corr_attenuation(df_true, df_false)
        expected = pd.read_excel(expected_path + "/corr_att_book.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual, expected)

    # testing assertions for add_attenuation
    # in creating these unit tests, changes made to Excel file are highlighted in yellow
    
    def test_attenuation_diff_shape(self):
        ''' Checks whether a ValueError is raised when the number of true and false irradiated samples in dataframe do not match (i.e. missing data). '''
        
        df = pd.read_excel(input_path + "/att_diff_shape_input.xlsx", index_col=0) # last seven rows removed
        
        with pytest.raises(ValueError) as e:
            raise add_attenuation(df)
        
        assert e.match("Error, irrad_false and irrad_true dataframes are not the same shape to begin with.")

    def test_attenuation_book_subset(self):
        ''' Checks whether a ValueError is raised when subsets of the true and false irradiated dataframes do not match. '''
        
        df = pd.read_excel(input_path + "/att_book_subset_wrong.xlsx", index_col=0)
        
        with pytest.raises(ValueError) as e:
            raise add_attenuation(df)
            
        assert e.match("Error, intensity_irrad true and false dataframes are not equal, cannot compute signal attenutation in a one-to-one manner.")
    
    @pytest.mark.parametrize("filename", [f'/att_batch_subset_{i}.xlsx' for i in range(1, 5)])
    def test_attenuation_batch_subset(self, filename):
        ''' Checks whether a ValueError is raised when the sample_or_control, replicate, proton_peak_index, and sat_time values in the true and false irradiated dataframes do not match. '''
        
        df = pd.read_excel(input_path + filename, index_col=0)
        
        with pytest.raises(ValueError) as e:
            raise add_attenuation(df, 'batch')
            
        assert e.match("Error, intensity_irrad true and false dataframes are not equal, cannot compute signal attenutation in a one-to-one manner.")
                
    # testing assertions for add_corr_attenuation
    
    def test_corr_attenuation_diff_shape(self):
        ''' Checks whether a ValueError is raised when the number of true and false irradiated samples in dataframe do not match (i.e. missing data). '''
        
        df_true = pd.read_excel(input_path + "/att_book_true_diff.xlsx", index_col=0) # modified, last 7 rows removed
        df_false = pd.read_excel(input_path + "/att_book_false.xlsx", index_col=0) # not modified
        
        with pytest.raises(ValueError) as e:
            raise add_corr_attenuation(df_true, df_false)
        
        assert e.match("Error, corrected % attenuation input dataframes are not the same shape to begin with.")

    @pytest.mark.parametrize("filename", [f'/att_book_true_subset{i}.xlsx' for i in range(1, 4)])
    def test_corr_attenuation_subset(self, filename):
        
        df_true = pd.read_excel(input_path + filename, index_col=0) # modified
        df_false = pd.read_excel(input_path + "/att_book_false.xlsx", index_col=0) # not modified
                
        with pytest.raises(ValueError) as e:
            raise add_corr_attenuation(df_true, df_false)
        
        assert e.match("Error, input dataframes are not equal, cannot compute corrected signal attenutation in a one-to-one manner.")

class TestPlots:
    """ This class contains all the unit tests relating to the plot generation functions."""
    
    def test_concentration(self, remove):
        """ Checks for whether the expected concentration plot is generated and removes the plot upon teardown.
        
        Notes
        -----
        Simply checks for filepath existence, does not check whether the generated plot matches a baseline due to jitter.
        """
        
        # SETUP
        output_dir = remove
            
        current_df_title = "KHA"
        
        df = pd.read_excel(input_path + "/plot_input.xlsx", index_col=0)
        generate_concentration_plot(df, output_dir, current_df_title)
        
        actual = path + "/output/exploratory_concentration_plot_from_KHA.png"

        msg = "The generated plot could not be found."
        assert os.path.exists(actual), msg
    
    def test_ppm(self, remove):
        """ Checks for whether the expected ppm plot is generated and removes the plot upon teardown.
        
        Notes
        -----
        Simply checks for filepath existence, does not check whether the generated plot matches a baseline due to jitter.
        """
         
        # SETUP
        output_dir = remove
            
        current_df_title = "KHA"        
    
        df = pd.read_excel(input_path + "/plot_input.xlsx", index_col=0)
        generate_ppm_plot(df, output_dir, current_df_title)
        
        actual = path + "/output/exploratory_ppm_plot_from_KHA.png"

        msg = "The generated plot could not be found."
        assert os.path.exists(actual), msg
    
class TestPrep:
    """ This class contains all the unit tests relating to the prep functions. """
    
    def test_prep_mean_book(self):
        """ Checks whether an Excel book is prepped for statistical analysis on a "mean" basis.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5 and ignores datatype matching.
        """
        
        input_mean = pd.read_excel(input_path + "/prep_mean_book_input.xlsx", index_col=0)
        
        actual = prep_mean(input_mean)
        
        # preserve multi-index when reading in Excel file
        expected_mean_left = pd.read_excel(expected_path + "/prep_mean_book_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        expected_mean_right = pd.read_excel(expected_path + "/prep_mean_book_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    
    def test_prep_mean_batch(self):
        """ Checks whether an Excel batch is prepped for statistical analysis on a "mean" basis.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5 and ignores datatype matching.
        """
         
        input_mean = pd.read_excel(input_path + "/prep_mean_batch_input.xlsx", index_col=0)
        
        actual = prep_mean(input_mean, 'batch')
        
        # preserve multi-index when reading in Excel file
        expected_mean_left = pd.read_excel(expected_path + "/prep_mean_batch_output.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
        expected_mean_right = pd.read_excel(expected_path + "/prep_mean_batch_output.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    
    def test_prep_replicates_book(self):
        """ Checks whether an Excel book is prepped for statistical analysis per replicate.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5.
        """

        input_replicate = pd.read_excel(input_path + "/prep_replicate_book_input.xlsx", index_col=0)
        
        actual = prep_replicate(input_replicate)
        
        expected = pd.read_excel(expected_path + "/prep_replicate_book_output.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual, expected)
    
    def test_prep_replicates_batch(self):
        """ Checks whether an Excel batch is prepped for statistical analysis per replicate.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5.
        """
 
        input_replicate = pd.read_excel(input_path + "/prep_replicate_batch_input.xlsx", index_col=0)
        
        actual = prep_replicate(input_replicate, 'batch')
        
        expected = pd.read_excel(expected_path + "/prep_replicate_batch_output.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual, expected)

class TestT:
    """ This class contains all the unit tests relating to the t-test analysis function. """
    
    def test_t(self):
        """ Performa a sample t-test and checks whether the expected results were appended to the inputted dataframe.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-5. 
        """
        
        # preserve multi-index when reading in Excel file
        df = pd.read_excel(input_path + "/t_test_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        df_other = pd.read_excel(input_path + "/t_test_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
        input_df = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
        
        actual = t_test(input_df)

        # preserve multi-index when reading in Excel file
        expected_left = pd.read_excel(expected_path + "/t_test_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        expected_right = pd.read_excel(expected_path + "/t_test_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        expected_right.columns = pd.MultiIndex.from_product([expected_right.columns, ['']])
        expected = pd.merge(expected_left, expected_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        pd.testing.assert_frame_equal(actual, expected)
    
class TestAF:
    """ This class contains all the unit tests relating to the function compute_af. """
    
    def test_af(self):
        """ Checks whether the expected amplication factor was calculated and appended to the dataframe. """
         
        # preserve multi-index when reading in Excel file
        df_mean = pd.read_excel(input_path + "/af_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        df_mean_other = pd.read_excel(input_path + "/af_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
        mean = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
        
        df_replicate = pd.read_excel(input_path + "/af_replicates_input.xlsx", index_col=0)
        
        actual_mean, actual_replicates = compute_af(mean, df_replicate, 10)
        
        # preserve multi-index when reading in Excel file
        expected_mean_left = pd.read_excel(expected_path + "/af_mean_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        expected_mean_right = pd.read_excel(expected_path + "/af_mean_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        expected_replicates = pd.read_excel(expected_path + "/af_replicates_output.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual_mean, expected_mean, check_exact=True)

        pd.testing.assert_frame_equal(actual_replicates, expected_replicates, check_exact=True)

class TestDropBadPeaks:
    """This class contains all the unit tests relating to the execute_curvefit function."""
    
    def test_drop_peaks_book(self, remove):
        """ Checks whether the expected peaks were dropped and removes any generated files upon teardown. """
        
        # SETUP
        output_dir = remove
        df_title = "KHA"
        
        # Preserve multi-index when reading in Excel file
        df_mean = pd.read_excel(input_path + "/drop_mean_peaks_book_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        df_mean_other = pd.read_excel(input_path + "/drop_mean_peaks_book_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
        mean = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        df_replicates = pd.read_excel(input_path + "/drop_replicates_peaks_book_input.xlsx", index_col=0)
        
        actual_mean, actual_replicates = drop_bad_peaks(mean, df_replicates, df_title, output_dir)
        
        # Preserve multi-index when reading in Excel file
        expected_mean_left = pd.read_excel(expected_path + "/drop_mean_peaks_book_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        expected_mean_right = pd.read_excel(expected_path + "/drop_mean_peaks_book_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        expected_replicates = pd.read_excel(expected_path + "/drop_replicates_peaks_book_output.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual_mean, expected_mean, check_exact=True)
        
        pd.testing.assert_frame_equal(actual_replicates, expected_replicates, check_exact=True)
        
    def test_drop_peaks_batch(self, remove):     
        
        # SETUP
        output_dir = remove
        
        df_title = "CMC"
        
        # Preserve multi-index when reading in Excel file
        df_mean = pd.read_excel(input_path + "/drop_mean_peaks_batch_input.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
        df_mean_other = pd.read_excel(input_path + "/drop_mean_peaks_batch_input.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
        df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
        mean = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

        df_replicates = pd.read_excel(input_path + "/drop_replicates_peaks_batch_input.xlsx", index_col=0)
        
        actual_mean, actual_replicates = drop_bad_peaks(mean, df_replicates, df_title, output_dir, 'batch')
        
        # Preserve multi-index when reading in Excel file
        expected_mean_left = pd.read_excel(expected_path + "/drop_mean_peaks_batch_output.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
        expected_mean_right = pd.read_excel(expected_path + "/drop_mean_peaks_batch_output.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

        expected_replicates = pd.read_excel(expected_path + "/drop_replicates_peaks_batch_output.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual_mean, expected_mean, check_exact=True)
        
        pd.testing.assert_frame_equal(actual_replicates, expected_replicates, check_exact=True)

# shorten
class TestCurveFit:
    """This class contains all the unit tests relating to the execute_curvefit function."""
    # can split to check for figures separately from df modification
    def test_curvefit_batch(self, remove):
        """ Checks for whether the curvefit was executed as expected; batch path. Removes all generated plots during teardown.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-3.
        """
        
        # SETUP
        output_dir = remove
        df_title = "CMC"
        output_curve = "{}/curve_fit_plots_from_{}".format(output_dir, df_title)
        output_table = "{}/data_tables_from_{}".format(output_dir, df_title)
        
        os.mkdir(output_curve)
        os.mkdir(output_table)
        
        # Preserve multi-index when reading in Excel file
        df_mean_left = pd.read_excel(input_path + "/batch_curve_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
        df_mean_right = pd.read_excel(input_path + "/batch_curve_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
        df_mean_right.columns = pd.MultiIndex.from_product([df_mean_right.columns, ['']])
        df_mean = pd.merge(df_mean_left, df_mean_right, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))
                
        df_replicates = pd.read_excel(input_path + "/batch_curve_replicate_input.xlsx", index_col=0)
        
        actual_mean, actual_replicates = execute_curvefit(df_mean, df_replicates, output_curve, output_table, df_title, 'batch')
        
        # Preserve multi-index when reading in Excel file
        expected_mean_left = pd.read_excel(expected_path + "/batch_curve_mean_output.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
        expected_mean_right = pd.read_excel(expected_path + "/batch_curve_mean_output.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

        expected_replicates = pd.read_excel(expected_path + "/batch_curve_replicate_output.xlsx", index_col=0)
    
        pd.testing.assert_frame_equal(df_mean, expected_mean, rtol=1e-3)
        pd.testing.assert_frame_equal(df_replicates, expected_replicates, rtol=1e-3)

        # check if the same plots are generated (can only compare filepath/name)

        actual_curve = glob.glob(output_curve + "/*")
        actual_table = glob.glob(output_table + "/*")
        
        expected_curve = glob.glob(expected_path + "/curve_fit_plots_from_CMC/*")
        expected_table = glob.glob(expected_path + "/data_tables_from_CMC/*")
        
        if len(actual_curve) != len(expected_curve):
            assert len(actual_curve) == len(expected_curve)
        
        if len(actual_table) != len(expected_table):
            assert len(actual_table) == len(expected_table)
        
        for i in range(len(actual_curve)): # uncomment the following and comment the uncommented lines to simple check for existence
            #actual_curve[i] = os.path.basename(actual_curve[i])
            #expected_curve[i] = os.path.basename(expected_curve[i])
            
            #assert actual_curve[i] == expected_curve[i]
            msg3 = "The generated plot {} does not match the expected plot.".format(actual_curve[i])
            assert compare_images(actual_curve[i], expected_curve[i], tol=0.1) is None, msg3 # compare pixel differences in plot
        
        for i in range(len(actual_table)):
            if "mean" in actual_table[i] and "mean" in expected_table[i]:
                
                # Preserve multi-index when reading in Excel file
                df_mean = pd.read_excel(actual_table[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
                df_mean_other = pd.read_excel(actual_table[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
                df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
                actual = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

                # Preserve multi-index when reading in Excel file
                expected_mean_left = pd.read_excel(expected_table[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
                expected_mean_right = pd.read_excel(expected_table[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
                expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
                expected = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))
    
                pd.testing.assert_frame_equal(actual, expected, rtol=1e-3)
                        
            elif "replicate" in actual_table[i] and "replicate" in expected_table[i]:
                actual_table[i] = pd.read_excel(actual_table[i], index_col=0)
                expected_table[i] = pd.read_excel(expected_table[i], index_col=0)
            
                pd.testing.assert_frame_equal(actual_table[i], expected_table[i], rtol=1e-3)
                
            else:
                
                msg4 = "Not all data tables were generated and exported."
                assert False, msg4
    
    #def test_curvefit_batch_figures(self, remove):
    
    def test_curvefit_book(self, remove):  
        """ Checks for whether the curvefit was executed as expected; book path. Removes all generated plots during teardown.
        
        Notes
        -----
        Equality checking uses a relative tolerance of 1e-3. 
        Simply checks for filepath existence. 
        """
        
        # SETUP
        output_dir = remove
        df_title = "KHA"
        output_curve = "{}/curve_fit_plots_from_{}".format(output_dir, df_title)
        output_table = "{}/data_tables_from_{}".format(output_dir, df_title)
        
        os.mkdir(output_curve)
        os.mkdir(output_table)
 
        # Preserve multi-index when reading in Excel file
        df_mean = pd.read_excel(input_path + "/book_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        df_mean_other = pd.read_excel(input_path + "/book_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
        mean = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
        
        df_replicates = pd.read_excel(input_path + "/book_replicates_input.xlsx", index_col=0)
        
        actual_mean, actual_replicates = execute_curvefit(mean, df_replicates, output_curve, output_table, df_title)
        
        # Preserve multi-index when reading in Excel file
        expected_mean_left = pd.read_excel(expected_path + "/book_meancurve.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        expected_mean_right = pd.read_excel(expected_path + "/book_meancurve.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        expected_replicates = pd.read_excel(expected_path + "/book_replicatescurve.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual_mean, expected_mean, rtol=1e-3)
        pd.testing.assert_frame_equal(actual_replicates, expected_replicates, rtol=1e-3)
        
        # check if the same plots are generated (can only compare filepath/name)

        actual_curve = glob.glob(output_curve + "/*")
        actual_table = glob.glob(output_table + "/*")
        
        expected_curve = glob.glob(expected_path + "/curve_fit_plots_from_KHA/*")
        expected_table = glob.glob(expected_path + "/data_tables_from_KHA/*")
        
        if len(actual_curve) != len(expected_curve):
            msg1 = "Not all curve plots were generated."
            assert len(actual_curve) == len(expected_curve), msg1
        
        if len(actual_table) != len(expected_table):
            msg2 = "Not all data tables were generated."
            assert len(actual_table) == len(expected_table), msg2
        
        for i in range(len(actual_curve)): # uncomment the following and comment the uncommented lines to simple check for existence
            #actual_curve[i] = os.path.basename(actual_curve[i])
            #expected_curve[i] = os.path.basename(expected_curve[i])
            
            #assert actual_curve[i] == expected_curve[i]
            msg3 = "The generated plot {} does not match the expected plot.".format(actual_curve[i])
            assert compare_images(actual_curve[i], expected_curve[i], tol=0.1) is None, msg3 # compare pixel differences in plot
        
        for i in range(len(actual_table)):
            if "mean" in actual_table[i] and "mean" in expected_table[i]:
                
                # Preserve multi-index when reading in Excel file
                df_mean = pd.read_excel(actual_table[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
                df_mean_other = pd.read_excel(actual_table[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
                df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
                actual = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

                # Preserve multi-index when reading in Excel file
                expected_mean_left = pd.read_excel(expected_table[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
                expected_mean_right = pd.read_excel(expected_table[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
                expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
                expected = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
    
                pd.testing.assert_frame_equal(actual, expected, rtol=1e-3)
                        
            elif "replicate" in actual_table[i] and "replicate" in expected_table[i]:
                actual_table[i] = pd.read_excel(actual_table[i], index_col=0)
                expected_table[i] = pd.read_excel(expected_table[i], index_col=0)
            
                pd.testing.assert_frame_equal(actual_table[i], expected_table[i], rtol=1e-3)
                
            else:
                
                msg4 = "Not all data tables were generated and exported."
                assert False, msg4

    #def test_curvefit_book_figures(self, remove):
        
        