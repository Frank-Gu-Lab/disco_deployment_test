##GUI Module for Streamlit application###

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

def remove_conc(polymer_name):
    i = 0
    for pos, letter in enumerate(reversed(polymer_name)):
        if letter == "_":
            i = pos
            break
    polymer_name = polymer_name[:-1 * i - 1]

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

    return clean_batch_tuple_list



###Down to business###


###Session management###

if "session_code" not in st.session_state:
    st.session_state["session_code"] = get_random_string(12)

st.sidebar.title("DISCO Navigation Sidebar")

if "session_dir" not in st.session_state and "session_code" in st.session_state:

    st.session_state["global_output_directory"] = os.path.abspath("app/data/output/" + st.session_state["session_code"])
    if not os.path.exists(st.session_state["global_output_directory"]):
        os.makedirs(st.session_state["global_output_directory"])

    st.session_state["merged_output_directory"] = "{}/merged".format(st.session_state["global_output_directory"])

    if not os.path.exists(st.session_state["merged_output_directory"]):
        os.makedirs(st.session_state["merged_output_directory"])
    # define data source path and data destination path to pass to data merging function
    source_path = '{}/*/tables_*'.format(st.session_state["global_output_directory"])
    destination_path = '{}'.format(st.session_state["merged_output_directory"])
try:
    past_dir = open("past_user.txt", "r")
except FileNotFoundError:
    past_dir = open(os.path.abspath("src/past_user.txt"), "r")

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
            if os.path.exists("../output/" + directory) and t.time() - float(time) >= 600:
                sht.rmtree("../output/" + directory)
            if os.path.exists("../output/" + directory) and t.time() - float(time) <= 600:
                dirs_to_keep.append(dir + "\n")
        past_dir.close()
        try:
            past_dir = open("past_user.txt", "w")
        except FileNotFoundError:
            past_dir = open(os.path.abspath("src/past_user.txt"), "w")
        past_dir.write("")
        past_dir.close()

        try:
            past_dir = open("past_user.txt", "a")
        except FileNotFoundError:
            past_dir = open(os.path.abspath("src/past_user.txt"), "a")
        past_dir.writelines(dirs_to_keep)
        past_dir.close()

    st.session_state["time"] = t.time()

    try:
        past_dir = open("past_user.txt", "a")
    except FileNotFoundError:
        past_dir = open(os.path.abspath("src/past_user.txt"), "a")
    past_dir.write(st.session_state["global_output_directory"] + " " + str(st.session_state["time"]) + "\n")
    past_dir.close()

###Page management###

list_of_raw_books = []
if t.time() - st.session_state["time"] < 600:
    choice = st.sidebar.radio("Would you like to upload data for data analysis, or plot data?", ["Read me", "Upload and analyze (Step 1)", "Plot data (Step 2)"])
else:
    choice = "None"
    st.info("Please refresh the page to begin a new session.  Session time expires after 10 minutes.")


if choice == "Read me":
    st.header("README")
    st.caption("This repository contains code by the Frank Gu Lab at UofT for DISCO-NMR data processing turned into an interactive GUI")
    try:
        st.image("aesthetic/disco_ball.gif")
    except FileNotFoundError:
        st.image(os.path.abspath("src/aesthetic/disco_ball.gif"))
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
    ''')
    st.subheader("Special Data Input Formatting Requirements")
    st.markdown("For Batch Format inputs, please ensure unique polymer replicates intended to be analyzed together follow the same naming format. For example, if there are 4 total CMC replicates, 3 from one experiment to be analyzed together, and 1 from a separate experiment that is NOT intended as a replicate of the other three, the sheet tabs should be named as follows:")
    st.write("""
        - CMC (1), CMC (2), CMC (3) (These 3 will be analyzed together, as their name string is the same, and all have a space and brackets as delimeters.)
        - CMC_other (The 4th CMC tab will be treated separately, as it is named with either a different string or delimeter (both in this case)
    """)
    st.markdown("Data tables should be formatted as follows in EXCEL:")
    try:
        st.image("aesthetic/table_exemplar.png")
    except FileNotFoundError:
        st.image(os.path.abspath("src/aesthetic/table_exemplar.png"))
    st.markdown("With the range keyword indicating to the program a datatable to be processed.  Please note how the on and off resonance tables for each saturation time must be formatted exactly as in the photo for each saturation time.  Also note the control column and the protein column (in this case BSM)")

if choice == "Upload and analyze (Step 1)":
    st.info("Please upload your data files to begin data processing!")
    list_of_raw_books = st.sidebar.file_uploader("Please provide input files", accept_multiple_files = True)

    MAX_BOOKS = 6
    if len(list_of_raw_books) > MAX_BOOKS:
        st.warning("Warning: Max file upload limit of 6.  Only the first 4 books will be processed")
        list_of_raw_books = list_of_raw_books[:7]

    i = 0


    if len(list_of_raw_books) > 0:

        data_checking(list_of_raw_books)

        clean_batch_tuple_list = analyzer(list_of_raw_books)

        del list_of_raw_books

        # LOOP THROUGH AND PROCESS EVERY CLEAN DATAFRAME IN THE BATCH LIST GENERATED ABOVE, IF ANY ----------------------------------

        if len(clean_batch_tuple_list) != 0:
            with st.spinner("Analyzing data..."):
                analyze_data(clean_batch_tuple_list, st.session_state["global_output_directory"])
        i += 6

        del clean_batch_tuple_list

    if i == 6:
        try:

            replicate_summary_df = etl_per_replicate(source_path, destination_path)
            replicate_summary_df.to_excel(os.path.join(st.session_state["merged_output_directory"], "merged_binding_dataset.xlsx"))

            quality_check_df = etl_per_sat_time(source_path, destination_path)
            quality_check_df.to_excel(os.path.join(st.session_state["merged_output_directory"], "merged_fit_quality_dataset.xlsx"))

            proton_summary_df = etl_per_proton(quality_check_df)

            proton_summary_df.to_excel(os.path.join(st.session_state["merged_output_directory"], "proton_binding_dataset.xlsx"))

            del proton_summary_df

            with st.expander("Open to see Merged Binding Dataset"):
                st.table(replicate_summary_df)
            with st.expander("Open to see Merged Fit Quality Dataset"):
                st.table(quality_check_df)

            del replicate_summary_df
            del quality_check_df

            i += 1

        except ValueError:
            st.warning("There were no binding polymers, please rerun with a new dataset to try other samples! (simply upload more to begin the process)")
            i += 1



        if i == 7:
            with open(os.path.join(st.session_state["merged_output_directory"], "proton_binding_dataset.xlsx"), "rb") as f:
                st.download_button("Download Proton Binding Dataset (for ML)", f, file_name = "proton_binding_dataset" + ".xlsx")


if choice == "Plot data (Step 2)":

        try:
            i = 0

            st.info("Preparing supporting figures.")

            list_of_polymers = []

            if "plotting_exception" not in st.session_state:
                st.session_state["plotting_exception"] = False

            if os.path.exists('{}/publications'.format(st.session_state["merged_output_directory"])):
                sht.rmtree('{}/publications'.format(st.session_state["merged_output_directory"]))
                st.session_state["plotting_exception"] = False

            with st.spinner("Establishing directories for supporting figures"):

                # for only the peaks with a significant disco effect
                polymer_library_binding = set(glob.glob(st.session_state["merged_output_directory"] + "/stats_analysis_output_mean_*")) - set(st.session_state["merged_output_directory"] + "/stats_analysis_output_mean_all_*")

                # significant and zero peaks
                polymer_library_all = glob.glob(st.session_state["merged_output_directory"] + "/stats_analysis_output_mean_all_*")

                polymer_library_replicates = glob.glob(st.session_state["merged_output_directory"] + "/stats_analysis_output_replicate_*")

                polymber_replicate_libary_binding = set(glob.glob(st.session_state["merged_output_directory"] + "/stats_analysis_output_replicate_*")) - set(glob.glob(st.session_state["merged_output_directory"] + ".stats_analysis_output_replicate_all_*"))

                # Define a custom output directory for formal figures

                output_directory = '{}/publications'.format(st.session_state["global_output_directory"])

                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                binding_directory2 = f"{output_directory}/all_peaks"

                if not os.path.exists(binding_directory2):
                    os.makedirs(binding_directory2)

                if len(polymer_library_all) > 0:
                    for polymer in polymer_library_all:

                        polymer_name = grab_polymer_name(full_filepath=polymer,
                                                         common_filepath= st.session_state["merged_output_directory"] + "/stats_analysis_output_mean_all_")

                        # read polymer datasheet
                        polymer_df = pd.read_excel(polymer, index_col=[0, 1, 2, 3], header=[0, 1])

                        generate_buildup_curve(polymer_df, polymer_name, binding_directory2)
                i += 1

            if i == 1:

                for polymer in polymer_library_all:
                    list_of_polymers.append(grab_polymer_name(polymer, common_filepath = st.session_state["merged_output_directory"] + "/stats_analysis_output_mean_all_"))

                mean_all_list = []
                mean_bindonly_list = []
                replicate_all_list = []
                replicate_bindonly_list = []

                non_binding = []

                with st.spinner("Buildup curves being generated"):
                    # plot DISCO Effect build up curves with only significant peaks
                    for polymer in list_of_polymers:

                        mean_all = pd.read_excel(st.session_state["merged_output_directory"] + "/stats_analysis_output_mean_all_" + polymer+ ".xlsx", index_col=[0, 1, 2, 3], header=[0, 1]).reset_index()
                        mean_all_list.append((mean_all, polymer))

                        try:
                            mean_bindonly = pd.read_excel(st.session_state["merged_output_directory"] + "/stats_analysis_output_mean_" + polymer + ".xlsx", index_col=[0, 1, 2, 3], header=[0, 1]).reset_index()
                            mean_bindonly_list.append((mean_bindonly, polymer))
                        except FileNotFoundError:
                            pass

                        replicate_all = pd.read_excel(st.session_state["merged_output_directory"] + "/stats_analysis_output_replicate_all_" + polymer + ".xlsx", index_col=[0], header=[0]).reset_index(drop=True)
                        replicate_all_list.append((replicate_all, polymer))

                        try:
                            replicate_bindonly= pd.read_excel(st.session_state["merged_output_directory"] + "/stats_analysis_output_replicate_" + polymer + ".xlsx", index_col=[0], header=[0]).reset_index(drop=True)
                            replicate_bindonly_list.append((replicate_bindonly, polymer))
                        except FileNotFoundError:
                            non_binding.append(polymer)
                    i += 1

                if i == 2:

                    poly_choice = st.sidebar.radio("Please select a polymer to plot.", list_of_polymers)

                    mosaic = """
                    AA
                    BB
                    """

                    gs_kw = dict(width_ratios=[1, 1.5], height_ratios=[1, 1.5])

                    fig, axd = plt.subplot_mosaic(mosaic, gridspec_kw=gs_kw, figsize=(3.3, 4), constrained_layout=False, dpi=150)

                    isBinding = 0

                    display_frame = None

                    with st.spinner("graphing polymers"):
                        for tuple in mean_bindonly_list:
                            if poly_choice == tuple[1]:
                                isBinding += 1
                                add_buildup_toax(tuple[0], axd['A'])
                                axd['A'].set_ylabel("DISCO Effect", fontdict = {"fontsize": 7})
                                axd['A'].set_xlabel("NMR Saturation Time (s)", fontdict = {"fontsize": 7})
                                axd['A'].axhline(y =0.0, color = "0.8", linestyle = "dashed")
                                axd['A'].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                axd['A'].xaxis.set_ticks(np.arange(0.25, 2.0, 0.25))
                                axd['A'].tick_params(axis = 'x', labelsize = 6)
                                axd['A'].tick_params(axis = 'y', labelsize = 6)
                                axd['A'].set_title("DISCO Effect Buildup Curve - " + poly_choice, fontdict = {"fontsize": 7})

                        for tuple in replicate_bindonly_list:
                            if poly_choice == tuple[1]:
                                isBinding += 1
                                display_frame = tuple[0]
                                add_fingerprint_toax(tuple[0], axd['B'])
                                axd['B'].set_ylabel("DISCO AFo (Absolute Value)", fontdict = {"fontsize": 7})
                                axd['B'].set_xlabel("1H Chemical Shift (Δ ppm)", fontdict = {"fontsize": 7})
                                axd['B'].axhline(y =0.0, color = "0.8", linestyle = "dashed")
                                axd['B'].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                axd['B'].tick_params(axis = 'x', labelsize = 6)
                                axd['B'].tick_params(axis = 'y', labelsize = 6)
                                axd["B"].set_title("DISCO Fingerprint - " + poly_choice, fontdict = {"fontsize": 7})

                        if isBinding >= 2:

                            props = dict(facecolor = "white", linewidth = 0.3)
                            legA = axd['A'].legend(loc = 'upper left', title = "Δ ppm", prop = {'size':5})
                            legA.get_frame().set_edgecolor('k')
                            legA.get_title().set_fontsize('6')
                            plt.rcParams['legend.fontsize'] = 7
                            legA.get_frame().set_linewidth(0.3)

                            output_filename = f"{output_directory}{poly_choice}.png"
                            plt.tight_layout()
                            fig.patch.set_facecolor('white')
                            fig.savefig(output_filename, dpi = 500, transparent = False)

                            st.image(output_filename, use_column_width = True)


                            st.success("Binding detected, right click and click save image to begin download.")

                            i += 1

                        elif poly_choice in non_binding:
                            list_of_curves = glob.glob(binding_directory2 + "/*")

                            for curve in list_of_curves:
                                if poly_choice in curve:
                                    st.image(curve, use_column_width = True)

                            st.warning("No binding detected for this polymer, displaying the buildup curve only.")

                            i += 1

                        if i >= 3:

                            list_of_polymer_names = []

                            for polymer in list_of_polymers:
                                if remove_conc(polymer) not in list_of_polymer_names:
                                    list_of_polymer_names.append(remove_conc(polymer))

                            list_of_polymers_by_weight = []

                            for polymer in list_of_polymer_names:
                                list_of_polymers_by_weight.append(grab_polymer_weight(polymer))

                            possible_weights = []
                            possible_NP_weights = []

                            for polymer in list_of_polymers_by_weight:
                                if (polymer[0] == grab_polymer_weight(poly_choice)[0] and polymer[1] != grab_polymer_weight(poly_choice)[1]):
                                    print(polymer[0])
                                    possible_weights.append(polymer[1])
                                if (grab_polymer_weight(poly_choice)[0] in polymer[0] and "NP" in polymer[0] and polymer[1] not in possible_NP_weights and ((polymer[0] == grab_polymer_weight(poly_choice)[0] and polymer[1] != grab_polymer_weight(poly_choice)[1]) or (polymer[0] != grab_polymer_weight(poly_choice)[0]))):
                                    print(polymer[0])
                                    possible_NP_weights.append(polymer[1])

                            NP_choice = False

                            if "NP" in poly_choice:
                                NP_choice = True

                            weight_choice = st.radio("Please choose the molecular weight to compare with", possible_weights, key = 7)

                            if len(possible_weights) > 0:

                                temp = ["a", "b"]
                                list_of_replicates_for_diff = []
                                for polymer in list_of_polymers_by_weight:
                                    for polymer2 in list_of_polymers_by_weight:
                                        if polymer2[0] in polymer[0] and ((NP_choice == False and polymer[1] != polymer2[1]) or (NP_choice == True and (("NP" not in polymer2[0]) or polymer[1] != polymer2[1]) )) and polymer == grab_polymer_weight(poly_choice) and polymer2[1] == weight_choice and "NP" not in polymer2[0]:

                                            st.info([polymer, polymer2])

                                            for tuple in replicate_all_list:
                                                if polymer[1] > polymer2[1]:
                                                    if polymer[0] in tuple[1] and str(polymer[1]) + "k" in tuple[1] and ((NP_choice == False and "NP" not in tuple[1]) or (NP_choice == True and "NP" in tuple[1])):
                                                        temp[0] = tuple
                                                    if polymer2[0] in tuple[1] and str(polymer2[1]) + "k" in tuple[1] and "NP" not in tuple[1]:
                                                        temp[1] = tuple
                                                else:
                                                    if polymer[0] in tuple[1] and str(polymer[1]) + "k" in tuple[1] and ((NP_choice == False and "NP" not in tuple[1]) or (NP_choice == True and "NP" in tuple[1])):
                                                        temp[1] = tuple
                                                    if polymer2[0] in tuple[1] and str(polymer2[1]) + "k" in tuple[1] and "NP" not in tuple[1]:
                                                        temp[0] = tuple
                                            condition = 0
                                            for pair in list_of_replicates_for_diff:
                                                if pair[0][1] != temp[0][1]:
                                                    condition = 1
                                            if condition == 0:
                                                list_of_replicates_for_diff.append([temp[0], temp[1]])

                                if len(list_of_replicates_for_diff) > 0:

                                    effect_size_df = generate_disco_effect_mean_diff_df(list_of_replicates_for_diff[0][1][0], list_of_replicates_for_diff[0][0][0])
                                    subset_sattime_df = generate_subset_sattime_df(effect_size_df, 0.25)


                                    figure, axy = plt.subplots(1, figsize = (16, 5))

                                    add_difference_plot_transposed(df = subset_sattime_df, ax = axy, dy = 0.3)

                                    axy.set_ylabel(" Standardized Effect Size \n(Hedges G, t=0.25s)", fontsize = 8)
                                    axy.set_ylim(-3, 2.5)
                                    axy.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                    axy.set_xlabel("1H Chemical Shift (Δ ppm)", fontsize = 6)
                                    axy.tick_params(axis = 'x', labelsize = 6)
                                    axy.tick_params(axis = 'y', labelsize = 6)
                                    plt.title(list_of_replicates_for_diff[0][1][1] + " vs " + list_of_replicates_for_diff[0][0][1])


                                    output_filename_2 = f"{output_directory}/" + list_of_replicates_for_diff[0][1][1] + "_diff" + ".png"
                                    figure.patch.set_facecolor("white")
                                    plt.tight_layout(pad = 1)
                                    figure.savefig(output_filename_2, dpi = 500, transparent = False)

                                    st.image(output_filename_2, use_column_width = True)
                                    i += 1

                                    with st.expander("Look at fingerprint for another binding polymer"):

                                        binding_curves = []
                                        for polymer in list_of_polymers:
                                            if polymer not in non_binding and polymer != poly_choice:
                                                binding_curves.append(polymer)

                                        weight_choice = st.radio("Please choose a binding polymer", binding_curves, key = 2)

                                        mosaic = """
                                        AA
                                        BB
                                        """

                                        gs_kw = dict(width_ratios=[1, 1.5], height_ratios=[1, 1.5])

                                        fig, axd = plt.subplot_mosaic(mosaic, gridspec_kw=gs_kw, figsize=(3.3, 4), constrained_layout=False, dpi=150)

                                        for tuple in mean_bindonly_list:
                                            if weight_choice == tuple[1]:
                                                add_buildup_toax(tuple[0], axd['A'])
                                                axd['A'].set_ylabel("DISCO Effect", fontdict = {"fontsize": 7})
                                                axd['A'].set_xlabel("NMR Saturation Time (s)", fontdict = {"fontsize": 7})
                                                axd['A'].axhline(y =0.0, color = "0.8", linestyle = "dashed")
                                                axd['A'].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                                axd['A'].xaxis.set_ticks(np.arange(0.25, 2.0, 0.25))
                                                axd['A'].tick_params(axis = 'x', labelsize = 6)
                                                axd['A'].tick_params(axis = 'y', labelsize = 6)
                                                axd['A'].set_title("DISCO Effect Buildup Curve - " + weight_choice, fontdict = {"fontsize": 7})

                                        for tuple in replicate_bindonly_list:
                                            if weight_choice == tuple[1]:
                                                add_fingerprint_toax(tuple[0], axd['B'])
                                                axd['B'].set_ylabel("DISCO AFo (Absolute Value)", fontdict = {"fontsize": 7})
                                                axd['B'].set_xlabel("1H Chemical Shift (Δ ppm)", fontdict = {"fontsize": 7})
                                                axd['B'].axhline(y =0.0, color = "0.8", linestyle = "dashed")
                                                axd['B'].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                                axd['B'].tick_params(axis = 'x', labelsize = 6)
                                                axd['B'].tick_params(axis = 'y', labelsize = 6)
                                                axd["B"].set_title("DISCO Fingerprint - " + weight_choice, fontdict = {"fontsize": 7})

                                        props = dict(facecolor = "white", linewidth = 0.3)
                                        legA = axd['A'].legend(loc = 'upper left', title = "Δ ppm", prop = {'size':5})
                                        legA.get_frame().set_edgecolor('k')
                                        legA.get_title().set_fontsize('6')
                                        plt.rcParams['legend.fontsize'] = 7
                                        legA.get_frame().set_linewidth(0.3)

                                        output_filename = f"{output_directory}{weight_choice}.png"
                                        plt.tight_layout()
                                        fig.patch.set_facecolor('white')
                                        fig.savefig(output_filename, dpi = 500, transparent = False)

                                        st.image(output_filename, use_column_width = True)


                            #Now you just gotta graph em!
                            weight_np_choice = st.radio("Please choose a nanoparticle to compare with", possible_NP_weights, key = 5)

                            if len(possible_NP_weights) > 0:

                                list_of_NP = []
                                for polymer in list_of_polymers_by_weight:
                                    if "NP" in polymer[0]:
                                        list_of_NP.append(polymer)

                                print(list_of_NP)

                                temp = ["a", "b"]
                                list_of_np_replicates_for_diff = []
                                for polymer in list_of_polymers_by_weight:
                                    for polymer2 in list_of_NP:
                                        #print(polymer[0] == grab_polymer_weight(poly_choice)[0] and polymer[1] == grab_polymer_weight(poly_choice)[1] and grab_polymer_weight(poly_choice)[0] in polymer2[0] and weight_np_choice == polymer2[1])
                                        if polymer == grab_polymer_weight(poly_choice) and grab_polymer_weight(poly_choice)[0] in polymer2[0] and weight_np_choice == polymer2[1] and "NP" in polymer2[0]:

                                            for tuple in replicate_all_list:
                                                if polymer[1] > polymer2[1]:
                                                    if polymer[0] in tuple[1] and str(polymer[1]) + "k" in tuple[1] and ((NP_choice == False and "NP" not in tuple[1]) or (NP_choice == True and "NP" in tuple[1])):
                                                        temp[0] = tuple
                                                    if polymer2[0] in tuple[1] and str(polymer2[1]) + "k" in tuple[1] and "NP" in tuple[1]:
                                                        temp[1] = tuple
                                                else:
                                                    if polymer[0] in tuple[1] and str(polymer[1]) + "k" in tuple[1] and ((NP_choice == False and "NP" not in tuple[1]) or (NP_choice == True and "NP" in tuple[1])):
                                                        temp[1] = tuple
                                                    if polymer2[0] in tuple[1] and str(polymer2[1]) + "k" in tuple[1] and "NP" in tuple[1]:
                                                        temp[0] = tuple
                                            condition = 0
                                            for pair in list_of_np_replicates_for_diff:
                                                if pair[0][1] != temp[0][1]:
                                                    condition = 1
                                            if condition == 0:
                                                list_of_np_replicates_for_diff.append([temp[0], temp[1]])

                                if len(list_of_np_replicates_for_diff) > 0:


                                    effect_size_df = generate_disco_effect_mean_diff_df(list_of_np_replicates_for_diff[0][1][0], list_of_np_replicates_for_diff[0][0][0])
                                    subset_sattime_df = generate_subset_sattime_df(effect_size_df, 0.25)


                                    figure, axy = plt.subplots(1, figsize = (16, 5))

                                    add_difference_plot_transposed(df = subset_sattime_df, ax = axy, dy = 0.3)

                                    axy.set_ylabel(" Standardized Effect Size \n(Hedges G, t=0.25s)", fontsize = 8)
                                    axy.set_ylim(-3, 2.5)
                                    axy.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                    axy.set_xlabel("1H Chemical Shift (Δ ppm)", fontsize = 6)
                                    axy.tick_params(axis = 'x', labelsize = 6)
                                    axy.tick_params(axis = 'y', labelsize = 6)
                                    plt.title(list_of_np_replicates_for_diff[0][1][1] + " vs " + list_of_np_replicates_for_diff[0][0][1])


                                    output_filename_8 = f"{output_directory}/" + list_of_np_replicates_for_diff[0][1][1] + "_diff" + ".png"
                                    figure.patch.set_facecolor("white")
                                    plt.tight_layout(pad = 1)
                                    figure.savefig(output_filename_8, dpi = 250, transparent = False)

                                    st.image(output_filename_8, use_column_width = True)
                                    i += 1

                                    with st.expander("Look at fingerprint for another binding polymer"):

                                        binding_curves = []
                                        for polymer in list_of_polymers:
                                            if polymer not in non_binding and polymer != poly_choice:
                                                binding_curves.append(polymer)

                                        weight_choice = st.radio("Please choose a binding polymer", binding_curves, key = 11)

                                        mosaic = """
                                        AA
                                        BB
                                        """

                                        gs_kw = dict(width_ratios=[1, 1.5], height_ratios=[1, 1.5])

                                        fig, axd = plt.subplot_mosaic(mosaic, gridspec_kw=gs_kw, figsize=(3.3, 4), constrained_layout=False, dpi=150)

                                        for tuple in mean_bindonly_list:
                                            if weight_choice == tuple[1]:
                                                add_buildup_toax(tuple[0], axd['A'])
                                                axd['A'].set_ylabel("DISCO Effect", fontdict = {"fontsize": 7})
                                                axd['A'].set_xlabel("NMR Saturation Time (s)", fontdict = {"fontsize": 7})
                                                axd['A'].axhline(y =0.0, color = "0.8", linestyle = "dashed")
                                                axd['A'].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                                axd['A'].xaxis.set_ticks(np.arange(0.25, 2.0, 0.25))
                                                axd['A'].tick_params(axis = 'x', labelsize = 6)
                                                axd['A'].tick_params(axis = 'y', labelsize = 6)
                                                axd['A'].set_title("DISCO Effect Buildup Curve - " + weight_choice, fontdict = {"fontsize": 7})

                                        for tuple in replicate_bindonly_list:
                                            if weight_choice == tuple[1]:
                                                add_fingerprint_toax(tuple[0], axd['B'])
                                                axd['B'].set_ylabel("DISCO AFo (Absolute Value)", fontdict = {"fontsize": 7})
                                                axd['B'].set_xlabel("1H Chemical Shift (Δ ppm)", fontdict = {"fontsize": 7})
                                                axd['B'].axhline(y =0.0, color = "0.8", linestyle = "dashed")
                                                axd['B'].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                                axd['B'].tick_params(axis = 'x', labelsize = 6)
                                                axd['B'].tick_params(axis = 'y', labelsize = 6)
                                                axd["B"].set_title("DISCO Fingerprint - " + weight_choice, fontdict = {"fontsize": 7})

                                        props = dict(facecolor = "white", linewidth = 0.3)
                                        legA = axd['A'].legend(loc = 'upper left', title = "Δ ppm", prop = {'size':5})
                                        legA.get_frame().set_edgecolor('k')
                                        legA.get_title().set_fontsize('6')
                                        plt.rcParams['legend.fontsize'] = 7
                                        legA.get_frame().set_linewidth(0.3)

                                        output_filename = f"{output_directory}{weight_choice}.png"
                                        plt.tight_layout()
                                        fig.patch.set_facecolor('white')
                                        fig.savefig(output_filename, dpi = 500, transparent = False)

                                        st.image(output_filename, use_column_width = True)

                        if isinstance(display_frame, pd.DataFrame):

                            replicate_summary_df = etl_per_replicate(source_path, destination_path)
                            quality_check_df = etl_per_sat_time(source_path, destination_path)
                            proton_summary_df = etl_per_proton(quality_check_df)

                            new_summary = proton_summary_df.loc[proton_summary_df["polymer_name"] == poly_choice]

                            new_summary = new_summary[["polymer_name", "concentration", "ppm", "AFo"]]

                            with st.expander("Expand to see AFo by ppm for Fingerprint - "+ poly_choice):
                                st.table(new_summary)

            if isinstance(display_frame, pd.DataFrame):

                st.success("See below for information about fit quality and RSE")

                finger, axeruwu = plt.subplots(figsize = (6, 6))
                generate_errorplot(display_frame, axeruwu)

                coeff_df = generate_correlation_coefficient(display_frame)

                output_filename_3 = f"{output_directory}/" + poly_choice + "_Finger_RSE" + ".png"
                finger.patch.set_facecolor("white")
                plt.tight_layout(pad = 1)
                finger.savefig(output_filename_3, dpi = 500, transparent = False)

                rse_rep_dir = st.session_state["global_output_directory"] + "/" + poly_choice + "/plots_" + poly_choice + "/*"

                list_of_replicate_rse = glob.glob(rse_rep_dir)

                image_dir = []

                for image in list_of_replicate_rse:
                    if "mean" in image:
                        image_dir.append(image)

                with st.expander("Click to see Fit Quality Parameters for " + poly_choice + " Fingerprint"):
                    st.image(output_filename_3)
                    for image in image_dir:
                        st.image(image)
                    st.info("Please see table below for R^2 value by PPM for the non-linear fits above.")
                    st.table(coeff_df)
                    st.info("All plotting data remains available in the downloads above")


            with open(os.path.join(st.session_state["merged_output_directory"], "proton_binding_dataset.xlsx"), "rb") as f:
                st.download_button("Download Proton Binding Dataset (for ML)", f, file_name = "proton_binding_dataset" + ".xlsx")


        except NameError:
            st.warning("You do not have any datafiles to graph!")
        except NameError:
            st.warning("You do not have any datafiles to graph!")
