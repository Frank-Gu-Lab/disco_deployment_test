import pytest
import sys
import os
import glob
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from contextlib import contextmanager

# appending path to access sibling directory
sys.path.append(os.getcwd() + '/../src')

from discoprocess.data_plot import *
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

    @pytest.mark.parametrize('path', ['mean', 'rep'])
    def test_generate_curvefit_plot(self, path, remove):
        
        output_dir = remove
        
        sat_time = np.array([0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75])
        
        if path == "mean":
            y_ikj_df = pd.read_excel(input_path + "/generate_curvefit_plots_yijk_mean_input.xlsx", index_col=[0, 1, 2])
            param_vals = np.array([-0.03367754, 2.43837912])
            ppm = 4.21
            filename = output_dir + "/mean_conc9.5625_ppm4.21.png"
            c = 9.5625
            r = None
        else:
            y_ikj_df = pd.read_excel(input_path + "/generate_curvefit_plots_yijk_rep_input.xlsx", index_col=0)
            param_vals = np.array([-3.34786840e-02, 9.24356468e+01])
            ppm = 4.21
            filename = output_dir + "/replicate1_conc9.5625_ppm4.21.png"
            c = 9.5625
            r = 1
                    
        with assert_plot_added():
            generate_curvefit_plot(sat_time, y_ikj_df, param_vals, ppm, filename, c, r, mean_or_rep = path)
        
        actual_curve = filename
        expected_curve = expected_path + "/" + os.path.basename(actual_curve)
        
        msg = "The generated plot {} does not match the expected plot.".format(actual_curve)
        assert compare_images(actual_curve, expected_curve, tol=0.1) is None, msg # compare pixel differences in plot
        