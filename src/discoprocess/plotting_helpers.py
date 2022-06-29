# -*- coding: utf-8 -*-
"""
Helper functions used to simplify code for visualization of results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# DATA GRABBING FUNCTIONS
def grab_peak_subset_data(df, ppi):
    '''Subsets a df to only the desired proton peak index.

    Parameters:
    ----------
    df: Pandas.DataFrame
        replicate all data for one polymer ("all" flag signals raw data is from all protons, not just binding ones)
    ppi: int
        the desired proton peak index to subset to

    Returns:
    -------
    subset_df: Pandas.DataFrame
        copy of the dataframe after subset to the desired peak
    '''
    subset_df = df.loc[df['proton_peak_index'] == ppi].copy()

    return subset_df

def grab_disco_effect_data(subset_df):
    '''Grabs basic statistical summary of each disco effect(t)
    datapoint for a given peak in a given polymer.

    Parameters:
    -----------
    subset_df: Pandas.Dataframe
        subset_df comes is the desired peak subset of the replicate all raw data for one polymer

    Returns:
    --------
    grouped_df: pandas.DataFrame
        snapshot of all mean statistical parameters for each sat time and selected peak

    mean_disco: pd.Series
        mean disco effect value for the given peak at all sat times

    std_disco: pd.Series
        std dev disco effect value for the given peak at all sat times

    n: int
        number of replicates associated with statistical summary
    '''

    grouped_df = subset_df.groupby(by=['sat_time', 'proton_peak_index']).mean()
    mean_disco = subset_df.groupby(by=['sat_time', 'proton_peak_index']).mean()['corr_%_attenuation']
    std_disco = subset_df.groupby(by=['sat_time', 'proton_peak_index']).std()['corr_%_attenuation']
    n = subset_df['replicate'].max()

    return grouped_df, mean_disco, std_disco, n

def assemble_peak_buildup_df(df, ppi):
    '''Assembles a directly indexable dataframe of individual proton Disco Effect(t) data,
    that can be used to plot an individual peak buildup curve for the desired proton.

    Parameters:
    -----------
    df: pandas.Dataframe
        replicate all data file from one polymer, contains data for all peaks

    ppi: int
        proton peak index of the desired polymer proton for the buildup curve

    Returns:
    --------
    plot_df: pandas.DataFrame
        the assembled df containing data required to plot the buildup curve of the selected peak

    '''
    # grab data for desired peak
    subset_df = grab_peak_subset_data(df, ppi)

    # manipulate raw data and reassemble into a flat single column index plot df
    grouped_df, mean_disco, std_disco, n = grab_disco_effect_data(subset_df)
    plot_df = grouped_df.copy()
    plot_df = plot_df.rename(columns={"corr_%_attenuation": "corr_%_attenuation_mean"})
    plot_df['corr_%_attenuation_std'] = std_disco
    plot_df['sample_size'] = n
    plot_df = plot_df.drop(columns='replicate') # replicate data column doesn't make sense after grouping by mean
    plot_df = plot_df.reset_index()

    return plot_df

# ANNOTATION FUNCTIONS
def annotate_sig_buildup_points(ax, significance, sat_time, disco_effect, dx, dy, color):
    '''Adds markers to the buildup curve plot provided for by ax where points are flagged as
    statistically significant in the passed "significance" series.

    Parameters:
    ----------
    ax: matplotlib plot axis
        indicates the plot to be annotated

    significance: pd.Series or array-like
        iterable the length of the plot domain containing boolean flags for domain indices that should
        or should not be annotated as significant

    sat_time: np.array
        plot domain values

    disco_effect: np.array
        plot range values

    dx: float
        amount to shift the annotation marker away from datapoint on x axis

    dy: float:
        amount to shift the annotation marker away from datapoint on y axis

    color: string
        color-code or other matplotlib compatible signifier of marker colour

    Returns:
    -------
    None, performs annotation action on the plot ax

    '''

    sig_annotation_markers = significance.map({True: "*", False: " "}).reset_index(drop=True)

    for ix, marker in enumerate(sig_annotation_markers):
        ax.annotate(marker, xy=(
            sat_time[ix]+dx, disco_effect[ix]+dy), c=color)

    return

# QUALITY OF FIT CHECKS
def generate_errorplot(df, ax):
    '''Creates a proton-wise residual standard error plot for examining the quality of nonlinear regression fit that occured
    during AF0 calculation originally during disco data processing.
    Theory:
    -------
    RSS = sum((ymodel - yobs)**2)
    RSE = sqrt((1/n-2)*RSS)
    Notes:
    ------
    Use this plot to visually inspect the RSE per significant peak, to identify any false positive
    peak binding signals (i.e. high error indicates a poor quality nonlinear regression fit, such as a horizontal line AF0 buildup curve).
    This error plot should be complemented by a visual inspection of the original nonlinear regression data during disco
    data processing (outside the scope of this repo).
    '''
    # calculate error
    df['y_model'] = df['alpha']*(1-np.exp(-df['sat_time'] * df['beta']))
    df['RSS'] = (df['y_model'] - df['yikj'])**2
    rss = df.groupby(by=['sat_time', 'proton_peak_index']).sum()['RSS'].reset_index()
    n = df['replicate'].max()
    df['RSE'] = np.sqrt((1/(n-2))*rss['RSS'])


    # make plot
    sns.boxplot(data=df, x='ppm', y='RSE', ax=ax)
    plt.title("Residual Squared Error - {}".format(df['polymer_name'].values[0].replace("_", " ")))
    ax.invert_xaxis() # replicates NMR spectrum axis
    plt.xlabel("1H Chemical Shift (Δ ppm)")

    return

def generate_correlation_coefficient(df):

    df['y_model'] = df['alpha']*(1-np.exp(-df['sat_time'] * df['beta']))

    ppm_list = []

    for index, row in df.iterrows():
        if row['ppm'] not in ppm_list:
            ppm_list.append(row['ppm'])

    coeff_df = pd.DataFrame(columns = ["Sum of Squares of Residuals", "Total Sum of Squares", "Coefficient of Determination"], index = ppm_list)

    mean_frame = df.groupby(by = ['ppm', 'sat_time']).sum()[['yikj', 'y_model']].reset_index()

    for index, row in coeff_df.iterrows():
        row["Sum of Squares of Residuals"] = 0
        row["Total Sum of Squares"] = 0
        row["Coefficient of Determination"] = 0
        for index2, row2 in mean_frame.iterrows():
                if row2['ppm'] == index:
                    row["Sum of Squares of Residuals"] += (row2['y_model'] - row2['yikj'])**2
                    row["Total Sum of Squares"] += (row2['yikj'] - mean_frame['yikj'].mean())**2
        row["Coefficient of Determination"] = 1 - (row["Sum of Squares of Residuals"] / row["Total Sum of Squares"])

    return coeff_df
