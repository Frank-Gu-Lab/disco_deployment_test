# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 21 14:39 2021

@author: Samantha Stuart

Standard publication plots generated from processed DISCO data. 
"""
import pandas as pd
import os
import glob

from discoprocess.data_plot import generate_buildup_curve

polymer_library = glob.glob('../data/output/merged/stats_analysis_output_mean_all_*')

for polymer in polymer_library:

    polymer_df = pd.read_excel(polymer, index_col = [0,1,2,3], header = [0,1])
    
    
    generate_buildup_curve(polymer_df)

    
    


