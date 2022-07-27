import os

try: # if packages are installed
    from .data_wrangling_functions import *
    from .data_plot import generate_concentration_plot, generate_ppm_plot
except: # if packages are not installed
    from data_wrangling_functions import *
    from data_plot import generate_concentration_plot, generate_ppm_plot

def generate_directories(current_df_title, global_output_directory):

    # DEFINE CUSTOM OUTPUT DIRECTORIES FOR THIS DATAFRAME ------------------------------------------

    # Define a custom output directory for the current df in the list
    output_directory = "{}/{}".format(global_output_directory, current_df_title)

    # make directory if there isn't already one for overall output for current df
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define an output exploratory directory for the exploratory plots
    output_directory_exploratory = "{}/exploratory_{}".format(output_directory, current_df_title)

    # make directory if there isn't already one for exploratory output
    if not os.path.exists(output_directory_exploratory):
        os.makedirs(output_directory_exploratory)

    # Define an output directory for the curve fit plots for the current df title
    output_directory_curve = "{}/plots_{}".format(output_directory, current_df_title)

    # make this directory if there isn't already one for the curve_fit_plots
    if not os.path.exists(output_directory_curve):
        os.makedirs(output_directory_curve)

    # Define an output directory for the final data tables after curve fitting and stats
    output_directory_tables = "{}/tables_{}".format(output_directory, current_df_title)

    # make this directory if there isn't already one for the data tables
    if not os.path.exists(output_directory_tables):
        os.makedirs(output_directory_tables)

    return output_directory_exploratory, output_directory_curve, output_directory_tables, output_directory

def modeling_data(current_df_attenuation, current_df_title, output_directory, output_directory_curve, output_directory_tables):

    # STEP 1 OF 5 - Prepare and generate mean dataframe of current data for stats with degrees of freedom and sample size included -----

    current_df_mean = prep_mean(current_df_attenuation)
    current_df_replicates = prep_replicate(current_df_attenuation)

    del current_df_attenuation

    # STEP 2 OF 5 - Perform t test for statistical significance -------------------------
    current_df_mean = t_test(current_df_mean, p=0.05)

    # STEP 3 OF 5 - Compute amplification factor -----------------------------------------
    # note, for batch: if AF denominators are different for each polymer, should make a list of all values for all polymers, then pass list[i] to af_denominator here

    current_df_mean, current_df_replicates = compute_af(current_df_mean, current_df_replicates, af_denominator = 10)

    # export current dataframes to excel with no observations dropped, for future reference in ML -----
    output_file_name = "stats_analysis_output_replicate_all_{}.xlsx".format(current_df_title) # replicates
    current_df_replicates.to_excel(os.path.join(output_directory_tables, output_file_name))
    output_file_name = "stats_analysis_output_mean_all_{}.xlsx".format(current_df_title) # mean
    current_df_mean.to_excel(os.path.join(output_directory_tables, output_file_name))

    # STEP 4 OF 5 - Drop proton peaks from further analysis that fail our acceptance criteria ---------------b--------------------------

    current_df_mean, current_df_replicates = drop_bad_peaks(current_df_mean, current_df_replicates, current_df_title, output_directory)

    # STEP 5 OF 5 - Perform curve fitting, generate plots, and export results to file  -----------------------------------------

    execute_curvefit(current_df_mean, current_df_replicates, output_directory_curve, output_directory_tables, current_df_title)

def analyze_data(tuple_list, global_output_directory):


        for i in range(len(tuple_list)):

            # BEGINNING PART 1 -------- Reading in Data and Visualizing the Results ------------------------

            #define current dataframe to be analyzed and its title from the tuple output of the data cleaning code
            current_df = tuple_list[i][1]
            current_df_title = tuple_list[i][0]

            # remove any spaces from title
            current_df_title = current_df_title.replace(' ', '')
            # DEFINE CUSTOM OUTPUT DIRECTORIES FOR THIS DATAFRAME ------------------------------------------

            output_directory_exploratory, output_directory_curve, output_directory_tables, output_directory = generate_directories(current_df_title, global_output_directory)

            # CALCULATE ATTENUATION & CORR ATTENUATION -----------------------------------

            df_true, df_false = add_attenuation(current_df)
            current_df_attenuation = add_corr_attenuation(df_true, df_false)


            modeling_data(current_df_attenuation, current_df_title, output_directory, output_directory_curve, output_directory_tables)
