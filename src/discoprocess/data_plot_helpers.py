# -*- coding: utf-8 -*-
"""
Created on Tuesday Nov 23rd 13:17 2021

@author: hellosamstuart

Performs statistical tests and other operations on datasets used during plotting.

"""
import pandas as pd
import numpy as np

def tukey_fence(df, variable):
    ''' Applies Tukey's fence method of outlier detection to flag outliers in a dataset.

    Parameters:
    -----------
    df: Pandas.Dataframe    
        dataset to be checked for outliers

    variable: string
        column name in df containing univariate variable to be assessed
    
    Returns:
    --------
    df: Pandas.Dataframe
        same dataset with a boolean column flagging outliers

    Notes:
    ------
    Presumed that df has been preformatted to only contain univariate data in var. Ensure
    this occurs before passing df to tukey_fence.

    Example:
    --------

    Reference:
    ----------
    Original code credit to Alicia Horsch, "Detecting and Treating Outliers in Python - Part 1"
    * https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755

    '''
    #Takes two parameters: dataframe & variable of interest as string
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr

    #inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence

    #outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence


    # assign outlier status back to df
    for index, x in (df[variable].items()):

        if x <= outer_fence_le or x >= outer_fence_ue: # flag if probable outlier
            df.loc[index, ('outlier_prob')] = True
            df.loc[index, ('outlier_poss')] = False 
            
        elif x <= inner_fence_le or x >= inner_fence_ue: # flag if possible outlier
            df.loc[index, ('outlier_poss')] = True
        
        else:
            df.loc[index, ('outlier_prob')] = False # indicate if neither
            df.loc[index, ('outlier_poss')] = False 

    return df


    