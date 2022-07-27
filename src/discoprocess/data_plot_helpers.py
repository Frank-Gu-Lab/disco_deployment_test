# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def tukey_fence(df, variable):

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
            df.loc[index, ('outlier_prob')] = False

        else:
            df.loc[index, ('outlier_prob')] = False # indicate if neither
            df.loc[index, ('outlier_poss')] = False

    return df
