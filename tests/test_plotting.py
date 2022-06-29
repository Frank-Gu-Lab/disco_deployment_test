import pytest
import sys
import os
import re
import glob
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

from contextlib import contextmanager

from matplotlib.ticker import FormatStrFormatter

sys.path.append(os.getcwd() + '/../src')

from discoprocess.plotting_helpers import annotate_sig_buildup_points
from discoprocess.wrangle_data import flatten_multicolumns, calculate_abs_buildup_params

from discoprocess.plotting import *
from discoprocess.wrangle_data import *

from discoprocess.plotting_helpers import assemble_peak_buildup_df

from matplotlib.testing.compare import compare_images

# global testing directories
path = os.path.dirname(__file__) + "/test-files/test_plotting"
input_path = path + "/input"
expected_path = path + "/expected"
output_directory = path + "/output"

#high mw CMC
high_cmc_mean_all = pd.read_excel(input_path + "/stats_analysis_output_mean_all_CMC_131k_20uM.xlsx", index_col=[0, 1, 2, 3], header=[0, 1]).reset_index()
high_cmc_mean_bindingonly = pd.read_excel(input_path + "/stats_analysis_output_mean_CMC_131k_20uM.xlsx", index_col=[0, 1, 2, 3], header=[0, 1]).reset_index()
high_cmc_replicate_all = pd.read_excel(input_path + "/stats_analysis_output_replicate_all_CMC_131k_20uM.xlsx", index_col=[0], header=[0]).reset_index(drop=True)
high_cmc_replicate_bindingonly = pd.read_excel(input_path + "/stats_analysis_output_replicate_CMC_131k_20uM.xlsx", index_col=[0], header=[0]).reset_index(drop=True)

#low mw CMC
low_cmc_mean_all = pd.read_excel(input_path + "/stats_analysis_output_mean_all_CMC_90k_20uM.xlsx", index_col=[0, 1, 2, 3], header=[0, 1]).reset_index()
low_cmc_mean_bindingonly = pd.read_excel(input_path + "/stats_analysis_output_mean_CMC_90k_20uM.xlsx", index_col=[0, 1, 2, 3], header=[0, 1]).reset_index()
low_cmc_replicate_all = pd.read_excel(input_path + "/stats_analysis_output_replicate_all_CMC_90k_20uM.xlsx", index_col=[0], header=[0]).reset_index(drop=True)
low_cmc_replicate_all = pd.read_excel(input_path + "/stats_analysis_output_replicate_CMC_90k_20uM.xlsx", index_col=[0], header=[0]).reset_index(drop=True)


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

def test_add_fingerprint_toax():

    with assert_plot_added():

        mosaic = """
        AA
        BB
        """

        gs_kw = dict(width_ratios=[1, 1.5], height_ratios=[1, 1.5])

        fig, axd = plt.subplot_mosaic(mosaic, gridspec_kw=gs_kw, figsize=(3.3, 4), constrained_layout=False, dpi=150)

        add_fingerprint_toax(high_cmc_replicate_bindingonly, axd['B'])
        axd['B'].set_ylabel("DISCO AFo")
        axd['B'].set_xlabel("1H Chemical Shift (Δ ppm)")
        axd['B'].axhline(y =0.0, color = "0.8", linestyle = "dashed")
        axd['B'].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axd['B'].tick_params(axis = 'x', labelsize = 6)
        axd['B'].tick_params(axis = 'y', labelsize = 6)


        props = dict(facecolor = "white", linewidth = 0.3)
        output_filename = f"{output_directory}/CMC_131k_20uM_fingerprint.png"
        plt.tight_layout()
        fig.patch.set_facecolor('white')
        fig.savefig(output_filename, dpi = 500, transparent = False)

        expected_filename = f"{expected_path}/CMC_131k_20uM_fingerprint_expected.png"

def test_buildup_toax():

    with assert_plot_added():

        mosaic = """
        AA
        BB
        """

        gs_kw = dict(width_ratios=[1, 1.5], height_ratios=[1, 1.5])

        fig2, axd2 = plt.subplot_mosaic(mosaic, gridspec_kw=gs_kw, figsize=(3.3, 4), constrained_layout=False, dpi=150)

        display_frame = high_cmc_mean_bindingonly
        add_buildup_toax(high_cmc_mean_bindingonly, axd2['A'])
        axd2['A'].set_ylabel("DISCO Effect")
        axd2['A'].set_xlabel("Saturation Time (s)")
        axd2['A'].axhline(y =0.0, color = "0.8", linestyle = "dashed")
        axd2['A'].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axd2['A'].xaxis.set_ticks(np.arange(0.25, 2.0, 0.25))
        axd2['A'].tick_params(axis = 'x', labelsize = 6)
        axd2['A'].tick_params(axis = 'y', labelsize = 6)


        props = dict(facecolor = "white", linewidth = 0.3)
        legA = axd2['A'].legend(loc = 'upper left', title = "Δ ppm", prop = {'size':5})
        legA.get_frame().set_edgecolor('k')
        legA.get_title().set_fontsize('6')
        plt.rcParams['legend.fontsize'] = 7
        legA.get_frame().set_linewidth(0.3)


        output_filename = f"{output_directory}/CMC_131k_20uM_buildup.png"
        plt.tight_layout()
        fig2.patch.set_facecolor('white')
        fig2.savefig(output_filename, dpi = 500, transparent = False)

        expected_filename = f"{expected_path}/CMC_131k_20uM_buildup_expected.png"

def test_add_difference_plot_transposed():

    with assert_plot_added():
        x =  pd.read_excel(input_path + "/stats_analysis_output_replicate_all_" + "CMC_90k_20uM" + ".xlsx", index_col=[0], header=[0]).reset_index(drop=True)

        y =  pd.read_excel(input_path + "/stats_analysis_output_replicate_all_" + "CMC_131k_20uM" + ".xlsx", index_col=[0], header=[0]).reset_index(drop=True)

        effect_size_df = generate_disco_effect_mean_diff_df(y, x)
        subset_sattime_df = generate_subset_sattime_df(effect_size_df, 0.25)

        figure, axy = plt.subplots(1, figsize = (16, 5))

        add_difference_plot_transposed(df = subset_sattime_df, ax = axy, dy = 0.3)

        axy.set_ylabel("CMC Standardized Effect Size \n(Hedges G, t=0.25s)", fontsize = 8)
        axy.set_ylim(-3, 2.5)
        axy.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axy.set_xlabel("1H Chemical Shift (Δ ppm)", fontsize = 6)
        axy.tick_params(axis = 'x', labelsize = 6)
        axy.tick_params(axis = 'y', labelsize = 6)


        output_filename_2 = f"{output_directory}/" + "CMC" + "_diff_T" + ".png"
        figure.patch.set_facecolor("white")
        plt.tight_layout(pad = 1)
        figure.savefig(output_filename_2, dpi = 500, transparent = False)

def test_add_difference_plot():

    with assert_plot_added():
        x =  pd.read_excel(input_path + "/stats_analysis_output_replicate_all_" + "CMC_90k_20uM" + ".xlsx", index_col=[0], header=[0]).reset_index(drop=True)

        y =  pd.read_excel(input_path + "/stats_analysis_output_replicate_all_" + "CMC_131k_20uM" + ".xlsx", index_col=[0], header=[0]).reset_index(drop=True)

        effect_size_df = generate_disco_effect_mean_diff_df(y, x)
        subset_sattime_df = generate_subset_sattime_df(effect_size_df, 0.25)

        figure, axy = plt.subplots(1, figsize = (16, 5))

        add_difference_plot(df = subset_sattime_df, ax = axy, dy = 0.3)

        axy.set_ylabel("CMC Standardized Effect Size \n(Hedges G, t=0.25s)", fontsize = 8)
        axy.set_ylim(-3, 2.5)
        axy.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axy.set_xlabel("1H Chemical Shift (Δ ppm)", fontsize = 6)
        axy.tick_params(axis = 'x', labelsize = 6)
        axy.tick_params(axis = 'y', labelsize = 6)


        output_filename_2 = f"{output_directory}/" + "CMC" + "_diff" + ".png"
        figure.patch.set_facecolor("white")
        plt.tight_layout(pad = 1)
        figure.savefig(output_filename_2, dpi = 500, transparent = False)


def test_add_overlaid_buildup_toax_customlabels():

    with assert_plot_added():

        df_list = []

        figure2, axywo = plt.subplots(1, figsize = (16, 5))

        x =  pd.read_excel(input_path + "/stats_analysis_output_replicate_all_" + "CMC_90k_20uM" + ".xlsx", index_col=[0], header=[0]).reset_index(drop=True)

        y =  pd.read_excel(input_path + "/stats_analysis_output_replicate_all_" + "CMC_131k_20uM" + ".xlsx", index_col=[0], header=[0]).reset_index(drop=True)

        ppi_1_low = assemble_peak_buildup_df(x, 2)
        ppi_1_high = assemble_peak_buildup_df(y, 2)

        cmc_effect_size_df = generate_disco_effect_mean_diff_df(x, y)

        kwargs = {"labels": ["90", "131"],
            "dx": 0.003,
            "dy": 0.010,
            "change_significance": cmc_effect_size_df,
            "annot_color": "#000000",
            "custom_colors": ['#b3cde3', '#377eb8']}

        df_list.append(ppi_1_low)
        df_list.append(ppi_1_high)

        add_overlaid_buildup_toax_customlabels(df_list, axywo, **kwargs)

        output_filename_3 = f"{output_directory}/" + "buildup_test_CMC" + ".png"
        figure2.patch.set_facecolor("white")
        plt.tight_layout(pad = 1)
        figure2.savefig(output_filename_3, dpi = 500, transparent = False)
