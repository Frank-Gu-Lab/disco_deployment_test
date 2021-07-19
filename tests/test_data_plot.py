import pytest
import sys
import os
import glob
import shutil
import matplotlib.pyplot as plt
import pandas as pd

from contextlib import contextmanager

# appending path to access sibling directory
sys.path.append(os.getcwd() + '/../src')

from data_plot import *
from matplotlib.testing.compare import compare_images

# global testing directories
path = "./test-files/test_data_plot"
input_path = path + "/input"
expected_path = path + "/expected"

@pytest.fixture(scope='function')
def remove():
    
    output_dir = path + "/output"
    os.mkdir(output_dir)
    
    yield output_dir
    
    shutil.rmtree(output_dir)

@contextmanager
def assert_plot_added():
    ''' Context manager that checks whether a plot was created (by Matplotlib) by comparing the total number of figures before and after. Referenced from [1].
    
    References
    ----------
    [1] https://towardsdatascience.com/unit-testing-python-data-visualizations-18e0250430
    '''
    
    plots_before = plt.gcf().number
    yield
    plots_after = plt.gcf().number
    assert plots_before < plots_after, "Error! Plot was not successfully created."

class TestGeneratePlot:
    
    def test_generate_concentration_plot(self, remove):
        """ Checks for whether the expected concentration plot is generated and removes the plot upon teardown.
        
        Notes
        -----
        Simply checks for filepath existence, does not check whether the generated plot matches a baseline due to jitter.
        """
        
        # SETUP
        output_dir = remove
            
        current_df_title = "KHA"
        
        df = pd.read_excel(input_path + "/plot_input.xlsx", index_col=0)
        
        with assert_plot_added(): # checks that a plot was created
            generate_concentration_plot(df, output_dir, current_df_title)
        
        actual = path + "/output/exploratory_concentration_plot_from_KHA.png"

        msg = "The generated plot could not be found."
        assert os.path.exists(actual), msg
    
    def test_generate_ppm_plot(self, remove):
        """ Checks for whether the expected ppm plot is generated and removes the plot upon teardown.
        
        Notes
        -----
        Simply checks for filepath existence, does not check whether the generated plot matches a baseline due to jitter.
        """
         
        # SETUP
        output_dir = remove
            
        current_df_title = "KHA"        
    
        df = pd.read_excel(input_path + "/plot_input.xlsx", index_col=0)
        
        with assert_plot_added(): # checks that a plot was created
            generate_ppm_plot(df, output_dir, current_df_title)
        
        actual = path + "/output/exploratory_ppm_plot_from_KHA.png"

        msg = "The generated plot could not be found."
        assert os.path.exists(actual), msg

    '''
    def test_generate_curvefit_plot(self, remove):
        
        actual_curve = glob.glob(output_curve + "/*")
        expected_curve = glob.glob(expected_path + "/curve_fit_plots_from_CMC/*")
        
        for i in range(len(actual_curve)): # uncomment the following and comment the uncommented lines to simple check for existence
            #actual_curve[i] = os.path.basename(actual_curve[i])
            #expected_curve[i] = os.path.basename(expected_curve[i])
            
            #assert actual_curve[i] == expected_curve[i]
            msg3 = "The generated plot {} does not match the expected plot.".format(actual_curve[i])
            assert compare_images(actual_curve[i], expected_curve[i], tol=0.1) is None, msg3 # compare pixel differences in plot
        return
    '''