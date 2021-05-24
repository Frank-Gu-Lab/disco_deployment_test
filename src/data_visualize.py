import os

def generate_directories(current_df_title, global_output_directory):
    # DEFINE CUSTOM OUTPUT DIRECTORIES FOR THIS DATAFRAME ------------------------------------------

    # Define a custom output directory for the current df in the list
    output_directory = "{}/{}".format(global_output_directory, current_df_title)

    # make directory if there isn't already one for overall output for current df
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define an output exploratory directory for the exploratory plots 
    output_directory_exploratory = "{}/exploratory_plots_from_{}".format(output_directory, current_df_title)

    # make directory if there isn't already one for exploratory output 
    if not os.path.exists(output_directory_exploratory):
        os.makedirs(output_directory_exploratory)

    # Define an output directory for the curve fit plots for the current df title
    output_directory_curve = "{}/curve_fit_plots_from_{}".format(output_directory, current_df_title)

    # make this directory if there isn't already one for the curve_fit_plots
    if not os.path.exists(output_directory_curve):
        os.makedirs(output_directory_curve)  

    # Define an output directory for the final data tables after curve fitting and stats
    output_directory_tables = "{}/data_tables_from_{}".format(output_directory, current_df_title)

    # make this directory if there isn't already one for the data tables
    if not os.path.exists(output_directory_tables):
        os.makedirs(output_directory_tables)    
        
    return output_directory_exploratory, output_directory_curve, output_directory_tables, output_directory
