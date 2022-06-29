from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns
import os
import numpy as np

# to customize DISCO palette:
# find .matplotlib dir on your machine
# >>> import matplotlib as mpl
# >>> mpl.get_configdir()
# show hidden files, navigate to dir/stylelib, customize disco file

# uncomment below to enable custom DISCO science plot style
# plt.style.use(['science', 'disco'])
plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.size':12})
# rc("text", usetex=False)

# uncomment below to enable LaTex mode - must install LaTex before Science Plots
# os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

# define colour palette
# colors = ['#f781bf', '#e41a1c', '#ff7f00',
#           '#984ea3', '#377eb8', '#4575b4','#74add1', '#abd9e9',
#           '#e0f3f8', '#fee090', '#fdae61', '#f46d43']

# TO DO: add to instructions the custom DISCO colour package
# TO DO: fix fingerprint plots to accomodate outliers

# ['#377eb8', '#984ea3', '#ff7f00', '#e41a1c', '#f781bf',
#           '#f46d43', '#fdae61', '#fee090',
#           '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']

try:
    from .data_wrangling_helpers import y_hat_fit
    from .data_plot_helpers import *
except:
    from data_wrangling_helpers import y_hat_fit
    from data_plot_helpers import *

def generate_concentration_plot(current_df_attenuation, output_directory_exploratory, current_df_title):
    '''This function generates a basic exploratory stripplot of polymer sample attenuation vs saturation time using
    concentration as a "hue" to differentiate points. This function also saves the plot to a custom output folder.

    Parameters
    ----------
    current_df_attenuation : Pandas.DataFrame
        Dataframe after attenuation and corrected % attenuation have been calculated and added as columns
        (output from add_attenuation_and_corr_attenuation_to_dataframe).

    output_directory_exploratory : str
        File path to directory that contains attenuation data for all polymer samples.

    cuurent_df_title : str
        Title of DataFrame.
    '''
    a4_dims = (11.7, 8.27)
    fig1, ax = plt.subplots(figsize = a4_dims)
    sns.stripplot(ax = ax, x = 'sat_time', y = 'corr_%_attenuation', data = current_df_attenuation, hue = 'concentration', palette = 'viridis')

    plt.title("Polymer Sample Attenuation vs Saturation Time")
    plt.ylabel("Corrected Signal Intensity Attenuation (%)")
    plt.xlabel("NMR Pulse Saturation Time (s)")

    # define file name for the concentration plot
    output_file_name_conc = "{}/conc_{}.png".format(output_directory_exploratory, current_df_title)

    # export to file
    fig1.tight_layout()
    fig1.savefig(output_file_name_conc, dpi=300)

    return

def generate_ppm_plot(current_df_attenuation, output_directory_exploratory, current_df_title):
    '''This function generates a basic exploratory scatterplot of polymer sample attenuation vs saturation time using
    ppm as a "hue" to differentiate points. This function also saves the plot to a custom output folder.

    Parameters
    ----------
    current_df_attenuation : Pandas.DataFrame
        Dataframe after attenuation and corrected % attenuation have been calculated and added as columns
        (output from add_attenuation_and_corr_attenuation_to_dataframe).

    output_directory_exploratory : str
        File path to directory that contains attenuation data for all polymer samples.

    cuurent_df_title : str
        Title of DataFrame.
    '''

    a4_dims = (11.7, 8.27)
    fig2, ax2 = plt.subplots(figsize = a4_dims)
    sns.scatterplot(ax = ax2, x = 'sat_time', y = 'corr_%_attenuation', data = current_df_attenuation, hue ='ppm', palette = 'viridis', y_jitter = True, legend = 'brief')

    # a stripplot looks nicer than this, but its legend is unneccessarily long with each individual ppm, need to use rounded ppm to use the below line
    # sns.stripplot(ax = ax2, x = 'sat_time', y = 'corr_%_attenuation', data = corr_p_attenuation_df, hue ='ppm', palette = 'viridis', dodge = True)

    plt.title("Polymer Sample Attenuation vs Saturation Time")
    plt.ylabel("Corrected Signal Intensity Attenuation  (%) by ppm")
    plt.xlabel("NMR Pulse Saturation Time (s)")
    ax2.legend()

    # define file name for the concentration plot
    output_file_name_ppm = "{}/ppm_{}.png".format(output_directory_exploratory, current_df_title)

    # export to file
    fig2.tight_layout()
    fig2.savefig(output_file_name_ppm, dpi=300)

    return

def generate_curvefit_plot(sat_time, df, param_vals, ppm, filename, c, r=None, mean_or_rep = 'mean'):
    ''' This function generates the curve-fitted plots of STD intensity vs saturation time on both a mean and replicate basis.

    Parameters
    ----------
    sat_time : NumPy.ndarray
        Dataframe containing saturation time values.

    df : Pandas.DataFrame
        Dataframe containing one graph data, mean

    param_vals : array-like
        Array-like containing the fitted parameters for curve fitting.

    r : int, default=None
        Unique replicate, only used in "rep" path.

    c : float
        Unique concentration, used in both the "mean" and "rep" paths.

    ppm : float
        Chemical shift of a particular proton.

    filename : str
        File path to the output directory where the figures are saved.

    mean_or_rep : str, {'mean', 'rep'}
        String indicating which path to take.

    Notes
    -----
    Exports plot to file.
    '''

    fig, (ax) = plt.subplots(1, figsize=(8, 4))

    if mean_or_rep == "mean":
        fig, (ax) = plt.subplots(2, figsize=(8, 4))

        # PLOT MEAN DF CURVE FITS with the original data and save to file
        model = y_hat_fit(sat_time, *param_vals)
        disco_effect = df['corr_%_attenuation']['mean']
        std_err = df['corr_%_attenuation']['std'] / np.sqrt(df['sample_size'])

        ax[0].plot(sat_time, model, 'g-', label='model_w_significant_params')
        ax[0].plot(sat_time, df['yikj_bar'], 'g.', label='all_raw_data',
                markeredgecolor='k', markeredgewidth=0.25)

        # y_hat_fit(significant_sat_time, *best_param_vals_bar) - significant_yikj_bar
        ax[0].set_title('Mean Curve Fit, Concentration = {} µmolar, ppm = {}'.format(c, ppm))
        ax[0].set_xlabel('NMR Saturation Time (s)')
        ax[0].set_ylabel('I/Io')

        # v2
        ax[1].plot(sat_time, disco_effect, '.', markeredgecolor='k', markeredgewidth=0.25, label='model_w_significant_params')
        ax[1].fill_between(sat_time, disco_effect+std_err,
                         disco_effect-std_err, facecolor='#377eb8', alpha=0.25)
        ax[1].set_title('Buildup Curve, Concentration = {} µmolar, ppm = {}'.format(c, ppm))
        ax[1].set_xlabel('NMR Saturation Time (s)')
        ax[1].set_ylabel('Disco Effect')


    else:
        # PLOT CURVE FITS with original data per Replicate and save to file
        ax.plot(sat_time, y_hat_fit(sat_time, *param_vals), 'b-', label='data')
        ax.plot(sat_time, df, 'b.', label='data')
        ax.set_title('Replicate = {} Curve Fit, Concentration = {} µmolar, ppm = {}'.format(r, c, ppm))
        ax.set_xlabel('NMR Saturation Time (s)')
        ax.set_ylabel('I/Io')

    plt.rcParams.update({'figure.max_open_warning': 0})
    fig.tight_layout()
    # export to file
    fig.savefig(filename, dpi=300)

    return

def generate_buildup_curve(df, polymer_name, output_directory):
    '''Generates the formal Disco Effect STD buildup curve for the figure.

    Parameters:
    -----------
    df : Pandas.Dataframe
        dataframe containing one polymer's DISCO Effect information for the build up curve

    polymer_name: string
        string containing the identifier of the polymer

    output_directory: string
        file path to desired output directory for saved plots

    Returns:
    -------
    None, outputs formal figures
    '''

    fig, (ax) = plt.subplots(1, figsize=(8, 4))
    polymer_name_plot = polymer_name.replace("_", " ")

    # for grouping information to plot
    concentration = df.index.get_level_values(0)
    ppm = df.index.get_level_values(3)

    # colour build up curves differently by ppm
    labels = np.round(ppm, 2)
    groups = df.groupby(labels)

    # plot DISCO effect build up curve
    for name, group in groups:
        sat_time = group.index.get_level_values(1)
        disco_effect = group['corr_%_attenuation']['mean'].values
        std = group['corr_%_attenuation']['std'].values
        n = group['sample_size']['Unnamed: 7_level_1'].values
        std_err = std/np.sqrt(n)

        y1 = np.subtract(disco_effect, std_err)
        y2 = np.add(disco_effect, std_err)

        ax.plot(sat_time, disco_effect, markeredgecolor='k', markeredgewidth=0.25,
                marker='o', linestyle='', ms=8, label=name)
        ax.fill_between(sat_time, y1, y2,
                        alpha=0.25)
        ax.legend(loc='best', bbox_to_anchor=(
            0.6, 0.3, 0.6, 0.6), title="Δ ppm")

    ax.set_title(f'DISCO Effect Buildup Curve - {polymer_name_plot}')
    ax.set_xlabel('NMR Saturation Time (s)')
    ax.set_ylabel('Disco Effect')

    # define file name
    output_file_name = f"{output_directory}/{polymer_name}-disco-curve.png"

    # export to file
    fig.tight_layout()
    fig.savefig(output_file_name, dpi=300)

    plt.close('all')

    return

def generate_fingerprint(df, polymer_name, output_directory):
    ''' Plots characteristic DISCO fingerprint of the binding polymer.

    Parameters:
    -----------
    df : Pandas.DataFrame
        dataframe containing one polymer's data with a normalized column

    output_directory: string
        file path to desired output directory for saved plots

    Returns:
    -------
    None, outputs formal figures
    '''

    fig, (ax) = plt.subplots(1, figsize=(4, 4))

    # reset index
    df = df.reset_index(drop = True)

    if df['concentration'].nunique() > 1: # if more than one conc per polymer, choose 20 uM version (for PAA data)
        df = df.loc[df['concentration'] == 20].copy()

    # clean polymer name string for plotting
    polymer_name_plot = polymer_name.replace("_", " ")

    # grab variables for bar plot
    # afo_bar = np.unique(df['AFo_bar'].values)
    # sse_bar = np.unique(df['SSE_bar'].values)
    # sse_datapoints = df.groupby(by = 'proton_peak_index')['SSE'].max()

    # afo_bar_norm = np.unique(df['AFo_bar_norm'].values)
    ppm = np.round(df.copy().groupby(by='proton_peak_index')['ppm'].mean(),2)
    ppm_ix = np.unique(df['proton_peak_index'].values)

    # map ppm to proton peak index incase multi ppms per proton peak index
    ppm_mapper = dict(zip(ppm_ix,ppm))
    df['ppm'] = df['proton_peak_index'].map(ppm_mapper)
    df['point_size'] = df['outlier_prob'].map({False: 4.0, True: 2.0})
    sizes = df['point_size'].values
    df['AFo'] = df['AFo'].abs() # use absolute values for fingerprints
    # print(df['AFo_norm'])
    # print(df['point_size'])
    # print(df)
    # print(type(df['ppm']))
    # print(type(sizes))
    # temp = df[some_filter]
    # temp.plot.scatter(x='x', y='y', s=temp['s'])
    # generate barplot

    sns.barplot(data = df, x = 'ppm', y = 'AFo', ax = ax, edgecolor = 'k')
    # df.plot.scatter(x = 'ppm', y = 'AFo_norm', s ='point_size')
    # print(df['point_size'].isna())
    sns.stripplot(data= df, x= 'ppm', y= 'AFo', ax= ax, edgecolor='k', linewidth=0.5)

    # sns.boxplot(data = df, x = 'ppm', y = 'AFo_norm', ax = ax)

    # # annotate outliers - PAA does this work?
    # for ix, flag in df['outlier_prob'].items():
    #     if flag == True:
    #         # print("Worked", flag)
    #         # print(df)
    #         ax.annotate("$\\diamond$", (df.loc[ix, 'proton_peak_index'], df.loc[ix, 'AFo_norm']+0.0001), size=7)


    # format plot
    ax.set_title(f'Binding Fingerprint \n{polymer_name_plot}')
    ax.set_xlabel("Peak Δ ppm)")
    ax.set_ylabel("DISCO AF_0")
    # ax.set_ylabel("Normalized DISCO $AF_0$")

    # reformat x axis
    ax.invert_xaxis()  # invert to match NMR spectrum

    # define file name
    output_file_name = f"{output_directory}/{polymer_name}-fingerprint.png"

    # export fig to file
    fig.tight_layout()
    fig.savefig(output_file_name, dpi=500, bbox_inches='tight')

    plt.close('all')


    return
