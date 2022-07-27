# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# DATA GRABBING FUNCTIONS
def grab_peak_subset_data(df, ppi):

    subset_df = df.loc[df['proton_peak_index'] == ppi].copy()

    return subset_df

def grab_disco_effect_data(subset_df):


    grouped_df = subset_df.groupby(by=['sat_time', 'proton_peak_index']).mean()
    mean_disco = subset_df.groupby(by=['sat_time', 'proton_peak_index']).mean()['corr_%_attenuation']
    std_disco = subset_df.groupby(by=['sat_time', 'proton_peak_index']).std()['corr_%_attenuation']
    n = subset_df['replicate'].max()

    return grouped_df, mean_disco, std_disco, n

def assemble_peak_buildup_df(df, ppi):

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


    sig_annotation_markers = significance.map({True: "*", False: " "}).reset_index(drop=True)

    for ix, marker in enumerate(sig_annotation_markers):
        ax.annotate(marker, xy=(
            sat_time[ix]+dx, disco_effect[ix]+dy), c=color)

    return

# QUALITY OF FIT CHECKS
def generate_errorplot(df, ax):

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
