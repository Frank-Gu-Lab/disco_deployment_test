#commenting this out because apparently we don't need these.

import pytest
import sys
import os
import glob
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from contextlib import contextmanager

# appending path to access sibling directory - uncomment if local package setup doesn't work
sys.path.append(os.getcwd() + '/../src')

from discoprocess.data_plot import *
from matplotlib.testing.compare import compare_images

# global testing directories
path = os.path.dirname(__file__) + "/test-files/test_data_plot"
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
        '''
         Checks for whether the expected concentration plot is generated and removes the plot upon teardown.

        Notes
        -----
        Simply checks for filepath existence, does not check whether the generated plot matches a baseline due to jitter.
        '''

        # SETUP
        output_dir = remove

        current_df_title = "CMC_20uM"

        df = pd.read_excel(input_path + "/plot_input.xlsx", index_col=0)

        with assert_plot_added(): # checks that a plot was created
            generate_concentration_plot(df, output_dir, current_df_title)

        actual = path + "/output/conc_CMC_20uM.png"

        msg = "The generated plot could not be found."
        assert os.path.exists(actual), msg


    def test_generate_ppm_plot(self, remove):
        '''
        Checks for whether the expected ppm plot is generated and removes the plot upon teardown.

        Notes
        -----
        Simply checks for filepath existence, does not check whether the generated plot matches a baseline due to jitter.
        '''

        # SETUP
        output_dir = remove

        current_df_title = "CMC_20uM"

        df = pd.read_excel(input_path + "/plot_input.xlsx", index_col=0)

        with assert_plot_added(): # checks that a plot was created
            generate_ppm_plot(df, output_dir, current_df_title)

        actual = path + "/output/ppm_CMC_20uM.png"

        msg = "The generated plot could not be found."
        assert os.path.exists(actual), msg


    @pytest.mark.parametrize('path', ['mean', 'rep'])
    def test_generate_curvefit_plot(self, path, remove):

        output_dir = remove

        sat_time = np.array([0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75])

        if path == "mean":
            y_ikj_df = pd.read_pickle("C:/Users/matth/OneDrive/Documents/GitHub/disco-data-processing/tests/test-files/test_data_plot/input/gen_plot_mean_input_df.pkl")
            print(y_ikj_df)
            param_vals = np.array([-0.03367754, 2.43837912])
            ppm = 1.58145
            filename = output_dir + "/mean_conc_plot.png"
            c = 20
            r = None
            expected_curve = expected_path + "/" + "mean_conc_expected.png"
        else:
            y_ikj_df = pd.read_pickle("C:/Users/matth/OneDrive/Documents/GitHub/disco-data-processing/tests/test-files/test_data_plot/input/gen_plot_rep_input_df.pkl")
            print(y_ikj_df)
            param_vals = np.array([-3.34786840e-02, 9.24356468e+01])
            ppm = 1.58145
            filename = output_dir + "/rep_conc_plot.png"
            c = 20
            r = 1
            expected_curve = expected_path + "/" + "rep_conc_expected.png"

        with assert_plot_added():
            generate_curvefit_plot(sat_time, y_ikj_df, param_vals, ppm, filename, c, r, mean_or_rep = path)

        actual_curve = filename

        msg = "The generated plot {} does not match the expected plot.".format(actual_curve)
        assert compare_images(actual_curve, expected_curve, tol=0.1) is None, msg # compare pixel differences in plot
