import pandas as pd
import pytest 
import os
import sys
import shutil

# appending path to access sibling directory
sys.path.append(os.getcwd() + '/../src')

from data_analyze import *

# global testing directories
path = "./test-files/test_analyze"

@pytest.fixture(scope='function')
def remove():
    
    output_dir = path + "/output"
    os.mkdir(output_dir)
    
    yield output_dir
    
    shutil.rmtree(output_dir)

class TestGenerateDirectory:
    """ This class contains all the unit tests relating to the function generate_directories."""
    
    def test_generate_directories(self, remove):
        """This function tests for overall functionality. Takes in a dataframe title (polymer)
        and a global output directory, and returns custom directories for the specific polymer. Removes all generated directories in teardown. """
        
        #SETUP
        current_df_title = "KHA"
        
        global_output_directory = remove
        
        output_directory_exploratory, output_directory_curve, output_directory_tables, output_directory = generate_directories(current_df_title, global_output_directory)
        
        assert os.path.exists(output_directory)
        assert os.path.exists(output_directory_curve)
        assert os.path.exists(output_directory_exploratory)
        assert os.path.exists(output_directory_tables)

class TestModeling:
    """This class contains all the unit tests relating to the function modeling_data."""
    
    @pytest.mark.parametrize('path', ['book', 'batch'])
    def test_modeling_data(self, remove, path, mocker):
        ''' As modeling_data is not a pure function, these unit tests simply check whether the expected dependencies are called. Pytest parametrize tests for both book and batch paths.
        
        Notes
        -----
        Mocked dependencies were given specified return values. These return values are simply a placeholder to maintain functionality and are not the actual 
        outputs of each dependency (should already have a separate unit test).
        '''
        
        df = pd.DataFrame({1:1, 2:2}, index=(1, 2))
        df_title = "Mock"
        global_output_directory = remove
        output_directory = global_output_directory + "/Mock"
        output_directory_curve = output_directory + "/Curve" 
        output_directory_tables = output_directory + "/Table"
        
        dirs = [output_directory, output_directory_curve, output_directory_tables]
        
        for d in dirs:
            os.mkdir(d)
        
        mock1 = mocker.patch("data_analyze.prep_mean")
        mock2 = mocker.patch("data_analyze.prep_replicate")
        mock3 = mocker.patch("data_analyze.t_test")
        mock4 = mocker.patch("data_analyze.compute_af", return_value=(df, df))
        mock5 = mocker.patch("data_analyze.drop_bad_peaks", return_value=(df, df))
        mock6 = mocker.patch("data_analyze.execute_curvefit", return_value=(df, df))
        
        modeling_data(df, df_title, output_directory, output_directory_curve, output_directory_tables, batch_or_book = path)
        
        mocks = [mock1, mock2, mock3, mock4, mock5, mock6]
        
        for mock in mocks:
            mock.assert_called_once()
            
        # check if outputs were successfully created in output_directory_tables
        assert os.path.exists(output_directory_tables + "/stats_analysis_output_replicate_all_Mock.xlsx"), "Analysis output for replicates was not exported to an Excel file!"
        assert os.path.exists(output_directory_tables + "/stats_analysis_output_mean_all_Mock.xlsx"), "Analysis output for means was not exported to an Excel file!"
        
        # check that book path was passed
        assert mock1.call_args_list[0][0][-1] == path, "{} path was not passed into prep_mean.".format(path.capitalize())
        assert mock2.call_args_list[0][0][-1] == path, "{} path was not passed into prep_replicate.".format(path.capitalize())
        assert mock5.call_args_list[0][0][-1] == path, "{} path was not passed into drop_bad_peaks.".format(path.capitalize())
        assert mock6.call_args_list[0][0][-1] == path, "{} path was not passed into execute_curvefit.".format(path.capitalize())
    
class TestAnalyze:
    ''' This class contains all the unit tests relating to the function data_analyze. '''
    
    @pytest.mark.parametrize('path', ['book', 'batch'])
    def test_analyze_data(self, remove, path, mocker):
        ''' As data_analyze is not a pure function, these unit tests simply check whether the expected dependencies are called. Pytest parametrize tests for both book and batch paths.
        
        Notes
        -----
        Mocked dependencies were given specified return values. These return values are simply a placeholder to maintain functionality and are not the actual 
        outputs of each dependency (should already have a separate unit test).
        '''
        
        df = pd.DataFrame({1:1, 2:2}, index=(1, 2))
        tuple_list = [('Mock', df)]
        global_output_directory = remove
            
        mock1 = mocker.patch("data_analyze.generate_directories", return_value=[global_output_directory + "/Mock"]*4)
        mock2 = mocker.patch("data_analyze.add_attenuation", return_value = (df, df))
        mock3 = mocker.patch("data_analyze.add_corr_attenuation")
        mock4 = mocker.patch("data_analyze.generate_concentration_plot")
        mock5 = mocker.patch("data_analyze.generate_ppm_plot")
        mock6 = mocker.patch("data_analyze.modeling_data", return_value = (df, df))
        
        mocks = [mock1, mock2, mock3, mock4, mock5, mock6]
        
        analyze_data(tuple_list, global_output_directory, batch_or_book = path)
                
        # check that dependencies are called only once
        for mock in mocks:
            mock.assert_called_once()
            
        # check that the 'batch' path was passed as expected
        assert mock2.call_args_list[0][0][-1] == path, "{} path was not passed into add_attenuation.".format(path.capitalize())
        assert mock3.call_args_list[0][0][-1] == path, "{} path was not passed into add_corr_attenuation.".format(path.capitalize())
        assert mock6.call_args_list[0][0][-1] == path, "{} path was not passed into modelling_data.".format(path.capitalize())
  