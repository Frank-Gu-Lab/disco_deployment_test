# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import t
import pingouin as pg

# DATA PREPROCESSING FUNCTIONS
def flatten_multicolumns(mean_df):


    # clean up multi index for both
    colnames = mean_df.columns.get_level_values(0).values
    mean_df = mean_df.droplevel(1, axis=1)
    colnames[4] = "corr_%_attenuation_mean"
    colnames[5] = "corr_%_attenuation_std"
    mean_df.columns = colnames

    return mean_df

# CALCULATE DISCO EFFECT PARAMS
def calculate_abs_buildup_params(df):


    # plot DISCO effect build up curve, absolute values
    sat_time = df['sat_time'].values
    disco_effect = abs(df['corr_%_attenuation_mean'].values)
    std = abs(df['corr_%_attenuation_std'].values)
    n = df['sample_size'].values
    std_err = std/np.sqrt(n)

    y1 = np.subtract(disco_effect, std_err)
    y2 = np.add(disco_effect, std_err)

    return sat_time, disco_effect, y1, y2

def calculate_buildup_params(df):


    # plot DISCO effect build up curve, absolute values
    sat_time = df['sat_time'].values
    disco_effect = df['corr_%_attenuation_mean'].values
    std = df['corr_%_attenuation_std'].values
    n = df['sample_size'].values
    std_err = std/np.sqrt(n)

    y1 = np.subtract(disco_effect, std_err)
    y2 = np.add(disco_effect, std_err)

    return sat_time, disco_effect, y1, y2

# CHANGE PROFILE DISCO EFFECT STAT TESTING, AND RESULTING DATA WRANGLING FUNCTIONS
def shapiro_wilk(effect):


    stat, p = stats.shapiro(effect)

    return p

def bartlett(effect1, effect2):


    stat, p = stats.bartlett(effect1, effect2)

    return p

def change_significance(group_1, group_2, alt_hyp="two-sided"):

    df_list = []

    for first, second, in zip(group_1, group_2):

        # assign data and indexes of each group included in statistical testing
        first_ix = first[0]
        first_df = first[1]
        second_ix = second[0]
        second_df = second[1]

        if first_ix == second_ix:  # validates that the same sat time and proton peak are being compared btw two groups
            ppm = first_df['ppm'].values[0]
            ppi = first_df['proton_peak_index'].values[0]

            disco_1 = abs(first_df['corr_%_attenuation'])
            disco_2 = abs(second_df['corr_%_attenuation'])

            # test for normality
            norm_p1 = shapiro_wilk(disco_1)  # if fail to reject null, they are treated as normal
            norm_p2 = shapiro_wilk(disco_2)

            # test for equal variance
            p_eq_var = bartlett(disco_2, disco_1)

            # if norm dist and equal variance do parametric test
            if np.logical_and(np.logical_and(norm_p2, norm_p1), p_eq_var) > 0.05:

                # calc and flag significance results
                stat, p_result = stats.ttest_ind(disco_2, disco_1, equal_var=True, alternative=alt_hyp)

                # calc 95 confidence intervals of datapoints
                # python code ref: https://stats.stackexchange.com/questions/475289/confidence-interval-for-2-sample-t-test-with-scipy
                disco_1_mean = np.mean(disco_1)
                v1 = np.var(disco_1, ddof = 1)
                n1 = len(disco_1)

                disco_2_mean = np.mean(disco_2)
                v2 = np.var(disco_2, ddof = 1)
                n2 = len(disco_2)

                delta_disco = disco_2_mean - disco_1_mean
                pooled_se = np.sqrt(v1 / n1 + v2 / n2)
                dof = (v1 / n1 + v2 / n2)**2 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))

                # upper and lower bounds 95 CI
                delta_CI_lower = delta_disco - t.ppf(0.95, dof)*pooled_se
                delta_CI_upper = delta_disco + t.ppf(0.95, dof)*pooled_se

                if float(p_result) <= 0.05:
                    sig_bool = True
                    print(f"Sig Point is: {first_ix[0]}, {ppm}, p = {p_result}, n = {len(disco_1)}")

                else:
                    sig_bool = False

                # calc effect size
                hedges_g = pg.compute_effsize(disco_2, disco_1, paired=False, eftype="hedges")

                # calc effect size sem for error bars
                # see pingouin docs for formula https://pingouin-stats.org/generated/pingouin.compute_esci.html
                effect_se = np.sqrt(((n1+n2) / (n1*n2)) + ((hedges_g**2) / (2*(n1+n2))))

                # write results to output
                current_dict = {"sat_time": first_ix[0],
                                "proton_peak_index": ppi,
                                "ppm": ppm,
                                "changed_significantly": sig_bool,
                                "mean_difference":delta_disco,
                                "delta_95_ci_lower": delta_CI_lower,
                                "delta_95_ci_upper": delta_CI_upper,
                                "delta_sem_lower": delta_disco - pooled_se,
                                "delta_sem_upper": delta_disco + pooled_se,
                                "effect_size": hedges_g,
                                "effect_sem_lower":hedges_g - effect_se,
                                "effect_sem_upper":hedges_g + effect_se}

                df_list.append(current_dict)

            else:
                print("Observations do not pass normality and equal variance test!")
                print("Shapiro P's: ", norm_p1, norm_p2)
                print("Equal Variance P:", p_eq_var)

    results_df = pd.DataFrame(df_list)

    return results_df

def generate_subset_sattime_df(effect_size_df, sat_time):


    subset_sattime_df = effect_size_df.loc[effect_size_df['sat_time'] == sat_time].copy().drop(columns="sat_time")

    return subset_sattime_df

def generate_disco_effect_mean_diff_df(replicate_df_low, replicate_df_high):


    # take absolute values of everything

    grouped_low = replicate_df_low.groupby(
        by=["sat_time", "proton_peak_index"])
    grouped_high = replicate_df_high.groupby(
        by=["sat_time", "proton_peak_index"])

    # perform t test per peak per sat time to see if sig change w increased property
    effect_size_df = change_significance(group_1=grouped_low, group_2=grouped_high)

    return effect_size_df
