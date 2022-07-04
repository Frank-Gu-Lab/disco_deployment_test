####Importing Dependencies####
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.stats import t
import scipy.stats as stats
from scipy.optimize import curve_fit
import os
import glob
import shutil as sht

###Import from modules###
from discoprocess.data_wrangling_functions import *
from discoprocess.data_merging import merge, etl_per_replicate, etl_per_sat_time, etl_per_proton
from discoprocess.data_analyze import *
from discoprocess.data_checker import *

from sklearn.preprocessing import MaxAbsScaler
from discoprocess.data_plot import generate_buildup_curve
from discoprocess.data_plot import generate_fingerprint
from discoprocess.data_plot_helpers import tukey_fence

from discoprocess.wrangle_data import generate_disco_effect_mean_diff_df, generate_subset_sattime_df
from discoprocess.plotting import add_fingerprint_toax, add_buildup_toax, add_difference_plot_transposed, add_overlaid_buildup_toax_customlabels, add_difference_plot
from discoprocess.plotting_helpers import assemble_peak_buildup_df, generate_errorplot, generate_correlation_coefficient
from PIL import Image
import copy as cp

from streamlit import caching

import random
import string

from tempfile import TemporaryDirectory

import time as t

idx = pd.IndexSlice

#Setting up the page name
st.set_page_config(page_title = "DISCO Data Processing")
st.title("DISCO Dashboard")

##Some helper Functions##

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))

    return result_str

def grab_polymer_name(full_filepath, common_filepath = " "):
    '''Grabs the polymer name from file path.

    Parameters:
    -----------
    full_filepath: string
        path to the datasheet for that polymer

    common_filepath: string
        portion of the filepath that is shared between all polymer inputs, excluding their custom names

    Returns:
    -------
    polymer_name: string
        the custom portion of the filepath with the polymer name and any other custom info
    '''

    #Necessary for some windows operating systems
    for char in full_filepath:
        if char == "\\":
            full_filepath = full_filepath.replace("\\", "/")

    for char in common_filepath:
        if char == "\\":
            common_filepath = full_filepath.replace("\\", "/")

    polymer_name = full_filepath.split(common_filepath)[1]
    polymer_name = polymer_name[:-5] # remove the .xlsx

    return polymer_name

def grab_polymer_weight(polymer_name):

    polymer_name = polymer_name.split("_")
    polymer_weight = polymer_name[1][:-1]

    return (polymer_name[0], int(polymer_weight))

def data_checking(list_of_raw_books):
    st.info("Checking names of polymers in datasets for correct formatting")

    i = 0

    if name_checker(list_of_raw_books):
        st.success("Names are formatted correctly!")
        i += 1

    if resonance_and_column_checker(list_of_raw_books):
        st.success("Data formatted correctly!")
        i += 1

    if range_checker(list_of_raw_books):
        st.success("Data ranges are all correct!")
        i += 1

    if i == 3:
        st.info("Great! Now it is time for data analysis!")
        i += 1

def analyzer(list_of_raw_books):

    batch_tuple_list = []

    clean_batch_tuple_list = []
    clean_batch_list = []

    for book in list_of_raw_books:
        #append tuples from the list output of the batch processing function, so that each unique polymer tuple is assigned to the clean_batch_tuple_list
        batch_tuple_list.append([polymer for polymer in batch_to_dataframe(book)])

    # PERFORM DATA CLEANING ON ALL BOOKS PROCESSED VIA BATCH PROCESSING ----------------

    #if there has been batch data processing, call the batch cleaning function
    if len(batch_tuple_list) != 0:

        clean_batch_list = clean_the_batch_tuple_list(batch_tuple_list)

        # convert clean batch list to a clean batch tuple list format (polymer_name, df) for further processing
        clean_batch_tuple_list = [(clean_batch_list[i]['polymer_name'].iloc[0], clean_batch_list[i]) for i in range(len(clean_batch_list))]

    # LOOP THROUGH AND PROCESS EVERY CLEAN DATAFRAME IN THE BATCH LIST GENERATED ABOVE, IF ANY ----------------------------------

    if len(clean_batch_tuple_list) != 0:
        with st.spinner("Analyzing data..."):
            analyze_data(clean_batch_tuple_list, st.session_state["global_output_directory"])
            

###Down to business###


###Session management###

if "session_code" not in st.session_state:
    st.session_state["session_code"] = get_random_string(12)

st.sidebar.title("DISCO Navigation Sidebar")

if "session_dir" not in st.session_state and "session_code" in st.session_state:

    st.session_state["global_output_directory"] = "../data/output/" + st.session_state["session_code"]
    if not os.path.exists(st.session_state["global_output_directory"]):
        os.makedirs(st.session_state["global_output_directory"])

    st.session_state["merged_output_directory"] = "{}/merged".format(st.session_state["global_output_directory"])

    if not os.path.exists(st.session_state["merged_output_directory"]):
        os.makedirs(st.session_state["merged_output_directory"])
    # define data source path and data destination path to pass to data merging function
    source_path = '{}/*/tables_*'.format(st.session_state["global_output_directory"])
    destination_path = '{}'.format(st.session_state["merged_output_directory"])

past_dir = open("past_user.txt", "r")

past_dirs = past_dir.read()

list_of_past_dirs = past_dirs.split("\n")

while "" in list_of_past_dirs:
    list_of_past_dirs.remove("")

while " " in list_of_past_dirs:
    list_of_past_dirs.remove(" ")


if "merged_output_directory" in st.session_state and "time" not in st.session_state:
    if len(list_of_past_dirs) > 0:
        dirs_to_keep = []
        for dir in list_of_past_dirs:
            directory, time = dir.split(" ")
            if os.path.exists("../data/output/" + directory) and t.time() - time >= 600:
                sht.rmtree("../data/output/" + directory)
            if os.path.exists("../data/output/" + directory) and t.time() - time <= 600:
                dirs_to_keep.append(dir + "\n")
        past_dir.close()
        past_dir = open("past_user.txt", "w")
        past_dir.write("")
        past_dir.close()

        past_dir = open("past_user.txt", "a")
        past_dir.writelines(dirs_to_keep)
        past_dir.close()

    st.session_state["time"] = t.time()

    past_dir = open("past_user.txt", "a")
    past_dir.write(st.session_state["global_output_directory"] + " " + str(st.session_state["time"]) + "\n")
    past_dir.close()

###Page management###

list_of_raw_books = []
choice = st.sidebar.radio("Would you like to upload data for data analysis, or plot data?", ["Read me", "Upload and analyze (Step 1)", "Plot data (Step 2)", "Window-viewer (Step 3)"])

if choice == "Read me":
    st.header("README")
    st.caption("This repository contains code by the Frank Gu Lab at UofT for DISCO-NMR data processing turned into an interactive GUI")
    st.image("aesthetic/disco_ball.gif")
    st.markdown("This code transforms outputs from individual **DISCO NMR** experiments in MestreNova into a statistically validated, clean Pandas DataFrame of true positive and true negative polymer-proton binding observations for further use in machine learning.")
    st.subheader("Data Process Description")
    st.write('''
        - Part 1: Reading and Cleaning Data (reads and prepares data for statistical analysis)
        - Part 2: Statistical Analysis (classifies true positive binding proton observations, generates AFo plots, writes outputs to file)
        - Part 3: Generate DataFrame + Export to Excel (merges true positive and true negative observations classified from Part 2 into ML-ready dataset)
    ''')
    st.subheader("Graphical Interface Workflow")
    st.write('''
        - Step 1: Upload data for analysis under the "Upload and analyze" tab.
        - Step 2: Go to the plotting tab to see plots of the data from the analysis tab
        - Step 3: Compare plots under the Window-viewer tab
    ''')
    st.subheader("Special Data Input Formatting Requirements")
    st.markdown("For Batch Format inputs, please ensure unique polymer replicates intended to be analyzed together follow the same naming format. For example, if there are 4 total CMC replicates, 3 from one experiment to be analyzed together, and 1 from a separate experiment that is NOT intended as a replicate of the other three, the sheet tabs should be named as follows:")
    st.write("""
        - CMC (1), CMC (2), CMC (3) (These 3 will be analyzed together, as their name string is the same, and all have a space and brackets as delimeters.)
        - CMC_other (The 4th CMC tab will be treated separately, as it is named with either a different string or delimeter (both in this case)
    """)
    st.markdown("Data tables should be formatted as follows in EXCEL:")
    st.image("aesthetic/table_exemplar.png")
    st.markdown("With the range keyword indicating to the program a datatable to be processed.  Please note how the on and off resonance tables for each saturation time must be formatted exactly as in the photo for each saturation time.  Also note the control column and the protein column (in this case BSM)")

if choice == "Upload and analyze (Step 1)":
    st.info("Please upload your data files to begin data processing!")
    list_of_raw_books = st.sidebar.file_uploader("Please provide input files", accept_multiple_files = True)

    MAX_BOOKS = 7
    if len(list_of_raw_books) > MAX_BOOKS:
        st.warning("Warning: Max file upload limit of 7.  Only the first 7 books will be processed")
        list_of_raw_books = list_of_raw_books[:8]

    i = 0


    if len(list_of_raw_books) > 0:
        data_checking(list_of_raw_books)
        analyzer(list_of_raw_books)
        i += 6

    if i == 6:
        try:

            replicate_summary_df = etl_per_replicate(source_path, destination_path)
            rep_sum = replicate_summary_df.to_excel(os.path.join(st.session_state["merged_output_directory"], "merged_binding_dataset.xlsx"))

            quality_check_df = etl_per_sat_time(source_path, destination_path)
            qual_check = quality_check_df.to_excel(os.path.join(st.session_state["merged_output_directory"], "merged_fit_quality_dataset.xlsx"))

            proton_summary_df = etl_per_proton(quality_check_df)

            prot_sum = proton_summary_df.to_excel(os.path.join(st.session_state["merged_output_directory"], "proton_binding_dataset.xlsx"))

            sht.make_archive(os.path.abspath(st.session_state["merged_output_directory"]), "zip", st.session_state["global_output_directory"], os.path.abspath(st.session_state["merged_output_directory"]))
            with open(st.session_state["merged_output_directory"] + ".zip", "rb") as f:
                st.download_button("Download Zip with Merged Datesets", f, file_name = "merged" + ".zip")
                i = i + 1

            with st.expander("Open to see Merged Binding Dataset"):
                st.table(replicate_summary_df)
            with st.expander("Open to see Merged Fit Quality Dataset"):
                st.table(quality_check_df)

        except ValueError:
            st.warning("There were no binding polymers, please rerun with a new dataset to try other samples! (simply upload more to begin the process)")

    if i == 7:
        st.info("Data analysis is complete.  If you would like to plot figures, please select the radio button above.")


if choice == "Plot data (Step 2)":
    pass
if choice == "Window-viewer (Step 3)":
    pass
