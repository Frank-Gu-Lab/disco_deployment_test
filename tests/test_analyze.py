import pytest 
import os
import sys
import shutil

# appending path to access sibling directory
sys.path.append(os.getcwd() + '\\..\\src')

from data_analyze import *

class TestGenerateDirectory:
    """ This class contains all the unit tests relating to the function generate_directories()"""
    
    def test_generate_directories(self):
        
        current_df_title = "KHA"
        global_output_directory = "./test-files/test_visualize/output"
        
        if not os.path.exists(global_output_directory):
            os.mkdir(global_output_directory)
        
        try:
        
            output_directory_exploratory, output_directory_curve, output_directory_tables, output_directory = generate_directories(current_df_title, global_output_directory)
            
            assert os.path.exists(output_directory)
            assert os.path.exists(output_directory_curve)
            assert os.path.exists(output_directory_exploratory)
            assert os.path.exists(output_directory_tables)
            
        finally:
            
            # TEARDOWN

            shutil.rmtree(global_output_directory + "/KHA")

class TestModelling:
    """This class contains all the unit tests relating to the function modelling()."""
    
    def test_modelling_book(self):
        """ Ideally, test should be independent of nested functions. Try to incorporate mocking."""
        
        path = "./test-files/test_visualize"
        global_output_directory = path + "/output"
        current_df_title = "KHA"
        output_directory = global_output_directory + "/KHA"
        output_directory_curve = output_directory + "/curve_fit_plots_from_KHA"
        output_directory_tables = output_directory + "/exploratory_plots_from_KHA"
        
        if not os.path.exists(global_output_directory):
            os.mkdir(global_output_directory)
            
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        if not os.path.exists(output_directory_curve):
            os.mkdir(output_directory_curve)
            
        if not os.path.exists(output_directory_tables):
            os.mkdir(output_directory_tables)
        
        try:
            
            current_df_attenuation = pd.read_excel(path + "/input/book_modelling_input.xlsx", index_col=0)

            actual_mean, actual_replicates = modelling_data(current_df_attenuation, current_df_title, output_directory, output_directory_curve, output_directory_tables)
        
            expected_mean_left = pd.read_excel(path + "/expected/book_modelling_mean.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
            expected_mean_right = pd.read_excel(path + "/expected/book_modelling_mean.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
            expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
            expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

            expected_replicates = pd.read_excel(path + "/expected/book_modelling_replicates.xlsx", index_col=0)
            
            pd.testing.assert_frame_equal(actual_mean, expected_mean, rtol=1e-3, check_dtype=False)
            pd.testing.assert_frame_equal(actual_replicates, expected_replicates, rtol=1e-3)
            
        finally:
            
            # TEARDOWN
            
            shutil.rmtree(output_directory)
        
        
    #def test_modelling_batch(self):

# incomplete
#class TestAnalyze:
    
    #def test_analyze(self):