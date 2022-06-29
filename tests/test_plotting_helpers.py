import pytest
import sys
import os
import re
import glob
import shutil
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

# appending path to access sibling directory - uncomment if local package setup doesn't work
sys.path.append(os.getcwd() + '/../src')

from discoprocess.plotting_helpers import *
from matplotlib.testing.compare import compare_images

# global testing directories
path = os.path.dirname(__file__) + "/test-files/test_plotting_helpers"
input_path = path + "/input"
expected_path = path + "/expected"

def test_grab_peak_subset_data():

    input = f"{input_path}/" + "stats_analysis_output_replicate_all_PEG_20k_20uM.xlsx"

    replicate_all = pd.read_excel(input, index_col=[0], header=[0]).reset_index(drop=True)

    expected  = replicate_all.loc[replicate_all['proton_peak_index'] == 1].copy()

    actual = grab_peak_subset_data(replicate_all, 1)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

def test_grab_disco_effect_data():

    input = f"{input_path}/" + "stats_analysis_output_replicate_all_PEG_20k_20uM.xlsx"

    replicate_all = pd.read_excel(input, index_col=[0], header=[0]).reset_index(drop=True)

    subset_df = replicate_all.loc[replicate_all['proton_peak_index'] == 1].copy()

    expected_1 = subset_df.groupby(by=['sat_time', 'proton_peak_index']).mean()
    expected_2 = subset_df.groupby(by=['sat_time', 'proton_peak_index']).mean()['corr_%_attenuation']
    expected_3 = subset_df.groupby(by=['sat_time', 'proton_peak_index']).std()['corr_%_attenuation']
    expected_4 = subset_df['replicate'].max()

    actual_1, actual_2, actual_3, actual_4 = grab_disco_effect_data(subset_df)

    assert(expected_4 == actual_4)
    pd.testing.assert_frame_equal(actual_1, expected_1)
    pd.testing.assert_series_equal(actual_2, expected_2)
    pd.testing.assert_series_equal(actual_3, expected_3)


def test_assemble_peak_buildup_df():

    input = f"{input_path}/" + "stats_analysis_output_replicate_all_PEG_20k_20uM.xlsx"

    replicate_all = pd.read_excel(input, index_col=[0], header=[0]).reset_index(drop=True)

    subset_df = replicate_all.loc[replicate_all['proton_peak_index'] == 1].copy()

    grouped_df = subset_df.groupby(by=['sat_time', 'proton_peak_index']).mean()
    mean_disco = subset_df.groupby(by=['sat_time', 'proton_peak_index']).mean()['corr_%_attenuation']
    std_disco = subset_df.groupby(by=['sat_time', 'proton_peak_index']).std()['corr_%_attenuation']
    n = subset_df['replicate'].max()

    expected_df = grouped_df.copy()
    expected_df = expected_df.rename(columns={"corr_%_attenuation": "corr_%_attenuation_mean"})
    expected_df['corr_%_attenuation_std'] = std_disco
    expected_df['sample_size'] = n
    expected_df = expected_df.drop(columns='replicate') # replicate data column doesn't make sense after grouping by mean
    expected_df = expected_df.reset_index()

    actual = assemble_peak_buildup_df(replicate_all, 1)

    pd.testing.assert_frame_equal(actual, expected_df)

def test_generate_correlation_coefficient():

    input_df = pd.read_pickle("C:/Users/matth/OneDrive/Documents/GitHub/disco-data-processing/tests/test-files/test_plotting_helpers/input/r_2_input_frame.pkl")

    expected_df = pd.read_pickle("C:/Users/matth/OneDrive/Documents/GitHub/disco-data-processing/tests/test-files/test_plotting_helpers/output/r_2_output_frame.pkl")

    actual_df = generate_correlation_coefficient(input_df)

    pd.testing.assert_frame_equal(actual_df, expected_df)
