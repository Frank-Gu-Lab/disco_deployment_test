# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from discoprocess.plotting_helpers import annotate_sig_buildup_points
from discoprocess.wrangle_data import flatten_multicolumns, calculate_abs_buildup_params

def add_fingerprint_toax(df, ax, **kwargs):

    ppm = np.round(df.copy().groupby(by='proton_peak_index')['ppm'].mean(), 2)
    ppm_ix = np.unique(df['proton_peak_index'].values)

    # map ppm to proton peak index incase multi ppms per proton peak index
    ppm_mapper = dict(zip(ppm_ix, ppm))
    df['ppm'] = df['proton_peak_index'].map(ppm_mapper)

    # remove duplicates from df
    df_plot = df[['ppm', 'AFo']].drop_duplicates()

    # take absolute value of AFo
    df_plot['AFo'] = df_plot['AFo'].abs()

    #create plot, use custom palette if there is one
    try:
        custom_palette = kwargs.pop("custom_palette")
        sns.barplot(data=df_plot, x='ppm', y='AFo', ax=ax,
                    edgecolor='k',errcolor='black',palette=custom_palette)
        sns.stripplot(data=df_plot, x='ppm', y='AFo', ax=ax, edgecolor='k',
                      linewidth=0.5, jitter=False, palette=custom_palette)

    except KeyError: # executes if there is no custom palette
        pass
        sns.barplot(data=df_plot, x='ppm', y='AFo', ax=ax, edgecolor='k', errcolor='black')
        sns.stripplot(data=df_plot, x='ppm', y='AFo', ax=ax,
                      edgecolor='k', linewidth=0.5, jitter=False)

    # reformat x axis
    ax.invert_xaxis()  # invert to match NMR spectrum

    return

def add_buildup_toax(df, ax):

    if type(df.columns) == pd.MultiIndex:
        df = flatten_multicolumns(df)  # make data indexable if it is not already

    ppm_labels = np.round(df['ppm'], 2)
    df_plot = df.copy()
    groups = df_plot.groupby([ppm_labels])

    # plot DISCO effect build up curve, absolute values
    for ppm, group in groups:
        sat_time, disco_effect, y1, y2 = calculate_abs_buildup_params(group)
        ax.plot(sat_time, disco_effect, markeredgecolor='k', markeredgewidth=0.35,
                marker='o', linestyle='', ms=5, label="%.2f" % ppm)
        ax.fill_between(sat_time, y1, y2, alpha=0.25)
        ax.legend(loc='best', title="Peak ($\delta$, ppm)")
        ax.axhline(y=0.0, color="0.8", linestyle='dashed')

    return

def add_overlaid_buildup_toax_customlabels(df_list, ax, **kwargs):

    # extract custom properties
    custom_labels = kwargs.pop("labels")
    dx = kwargs.pop("dx")
    dy = kwargs.pop("dy")
    change_sig_df = kwargs.pop("change_significance")
    buildup_colors = kwargs.pop("custom_colors")
    annot_color = kwargs.pop("annot_color")

    # plot overlaid buildups using the correct custom properties
    color_count = 0
    for ix, df in enumerate(df_list):
        plot_label = custom_labels[ix]

        if type(df.columns) == pd.MultiIndex:
            df = flatten_multicolumns(df)  # make data indexable if it is not already

        ppm_labels = np.round(df['ppm'], 2)
        df_plot = df.copy()
        groups = df_plot.groupby([ppm_labels])

        # plot DISCO effect build up curve, absolute values
        for _, group in groups:
            sat_time, disco_effect, y1, y2 = calculate_abs_buildup_params(group)

            full_plot_label = f"{plot_label}"
            ax.plot(sat_time, disco_effect, markeredgecolor='k', markeredgewidth=0.35, color=buildup_colors[color_count],
                    marker='o', linestyle='', ms=5, label=full_plot_label)
            ax.fill_between(sat_time, y1, y2, color=buildup_colors[color_count],
                            alpha=0.25)
            ax.axhline(y=0.0, color="0.8", linestyle='dashed')
        color_count += 1

    # annotate significance of change in disco effect (NOT disco adhesion interaction significance)
    key = group['proton_peak_index'].unique()[0]
    print(key)
    print(change_sig_df)
    change_sig_subset = change_sig_df.loc[change_sig_df['proton_peak_index'] == key]

    # annotate change sig points
    significance = change_sig_subset['changed_significantly']
    annotate_sig_buildup_points(ax, significance, sat_time, disco_effect, dx, dy, color=annot_color)

    return

def add_difference_plot(df, ax, dy, **kwargs):


    plot_range = range(1, (df.shape[0])+1)

    # zero line
    ax.axvline(x=0.0, color="0.8", linestyle='dashed')

    # error bars
    ax.hlines(y=plot_range, xmin=df['effect_sem_lower'],
              xmax=df['effect_sem_upper'], color='black', linewidth=2, zorder=1)

    # data
    ax.scatter(df['effect_size'], plot_range, s=(40),
               alpha=1, label='Effect Size', marker='o', linewidths=0.35, edgecolors='k', zorder = 2)
    ax.set_yticks(list(plot_range))

    # annotate significance
    df['annotation'] = df['changed_significantly'].map({True: "*", False: ""})

    for ix, value in enumerate(list(plot_range)):
        x = df['effect_size'].iloc[ix]
        y = value + dy
        marker = df['annotation'].iloc[ix]
        ax.annotate(marker, (x, y), c='#000000')

    return

def add_difference_plot_transposed(df, ax, dy, **kwargs):


    plot_domain = range(1, df.shape[0]+1)

    # zero line
    ax.axhline(y=0.0, color="0.8", linestyle='dashed')

    # error bars
    ax.vlines(x=plot_domain, ymin=df['effect_sem_lower'],
              ymax=df['effect_sem_upper'], color='black', linewidth=2, zorder = 1)

    # data
    ax.scatter(plot_domain, df['effect_size'], s = (40,), alpha=1, label='effect size', marker = 'o', linewidths = 0.35, edgecolors = 'k', zorder = 2)


    ax.set_xticks(list(plot_domain))

    ax.set_xticklabels(np.round(df['ppm'].values, 2))

    # annotate significance
    df['annotation'] = df['changed_significantly'].map({True: "*", False: ""})

    for ix, value in enumerate(list(plot_domain)):
            y = df['effect_size'].iloc[ix] + dy
            x = value + 0.05
            marker = df['annotation'].iloc[ix]
            ax.annotate(marker, (x,y), c = '#000000')

    return
