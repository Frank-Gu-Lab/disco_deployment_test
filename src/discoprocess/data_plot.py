import matplotlib.pyplot as plt    
import seaborn as sns

from .data_wrangling_helpers import y_hat_fit
        
def generate_concentration_plot(current_df_attenuation, output_directory_exploratory, current_df_title):
    '''This function generates a basic exploratory stripplot of polymer sample attenuation vs saturation time using
    concentration as a "hue" to differentiate points. This function also saves the plot to a custom output folder.
    
    Parameters
    ----------
    current_df_attenuation : Pandas.DataFrame
        Dataframe after attenuation and corrected % attenuation have been calculated and added as columns 
        (output from add_attenuation_and_corr_attenuation_to_dataframe).
    
    output_directory_exploratory : str
        File path to directory that contains attenuation data for all polymer samples.
        
    cuurent_df_title : str
        Title of DataFrame.
    '''
    a4_dims = (11.7, 8.27)
    fig1, ax = plt.subplots(figsize = a4_dims)
    sns.stripplot(ax = ax, x = 'sat_time', y = 'corr_%_attenuation', data = current_df_attenuation, hue = 'concentration', palette = 'viridis')

    plt.title("Polymer Sample Attenuation vs Saturation Time")
    plt.ylabel("Corrected Signal Intensity Attenuation (%)")
    plt.xlabel("NMR Pulse Saturation Time (s)")
    
    # define file name for the concentration plot
    output_file_name_conc = "{}/exploratory_concentration_plot_from_{}.png".format(output_directory_exploratory, current_df_title)
    
    # export to file
    fig1.savefig(output_file_name_conc, dpi=300)

    return

def generate_ppm_plot(current_df_attenuation, output_directory_exploratory, current_df_title):
    '''This function generates a basic exploratory scatterplot of polymer sample attenuation vs saturation time using
    ppm as a "hue" to differentiate points. This function also saves the plot to a custom output folder.
    
    Parameters
    ----------
    current_df_attenuation : Pandas.DataFrame
        Dataframe after attenuation and corrected % attenuation have been calculated and added as columns
        (output from add_attenuation_and_corr_attenuation_to_dataframe).
    
    output_directory_exploratory : str
        File path to directory that contains attenuation data for all polymer samples.
        
    cuurent_df_title : str
        Title of DataFrame.
    '''
    
    a4_dims = (11.7, 8.27)
    fig2, ax2 = plt.subplots(figsize = a4_dims)
    sns.scatterplot(ax = ax2, x = 'sat_time', y = 'corr_%_attenuation', data = current_df_attenuation, hue ='ppm', palette = 'viridis', y_jitter = True, legend = 'brief')

    # a stripplot looks nicer than this, but its legend is unneccessarily long with each individual ppm, need to use rounded ppm to use the below line
    # sns.stripplot(ax = ax2, x = 'sat_time', y = 'corr_%_attenuation', data = corr_p_attenuation_df, hue ='ppm', palette = 'viridis', dodge = True)

    plt.title("Polymer Sample Attenuation vs Saturation Time")
    plt.ylabel("Corrected Signal Intensity Attenuation  (%) by ppm")
    plt.xlabel("NMR Pulse Saturation Time (s)")
    ax2.legend() 

    # define file name for the concentration plot
    output_file_name_ppm = "{}/exploratory_ppm_plot_from_{}.png".format(output_directory_exploratory, current_df_title)

    # export to file
    fig2.savefig(output_file_name_ppm, dpi=300)
        
    return

def generate_curvefit_plot(sat_time, y_ikj_df, param_vals, ppm, filename, c, r=None, mean_or_rep = 'mean'):
    ''' This function generates the curve-fitted plots of STD intensity vs saturation time on both a mean and replicate basis.
    
    Parameters
    ----------
    sat_time : NumPy.ndarray
        Dataframe containing saturation time values.
    
    y_ikj_df : Pandas.DataFrame
        Dataframe containing 
    
    param_vals : array-like
        Array-like containing the fitted parameters for curve fitting.
    
    r : int, default=None
        Unique replicate, only used in "rep" path.
    
    c : float
        Unique concentration, used in both the "mean" and "rep" paths.
    
    ppm : float
        Chemical shift of a particular proton.
    
    filename : str
        File path to the output directory where the figures are saved.
    
    mean_or_rep : str, {'mean', 'rep'}
        String indicating which path to take.
    
    Notes
    -----
    Exports plot to file.
    '''
    
    fig, (ax) = plt.subplots(1, figsize = (8, 4))
    
    if mean_or_rep == "mean":
        # PLOT MEAN DF CURVE FITS with the original data and save to file
        ax.plot(sat_time, y_hat_fit(sat_time, *param_vals), 'g-', label='model_w_significant_params')
        ax.plot(sat_time, y_ikj_df, 'g*', label='all_raw_data')
        ax.set_title('Mean Curve Fit, Concentration = {} µmolar, ppm = {}'.format(c, ppm))
        
    else:
        # PLOT CURVE FITS with original data per Replicate and save to file
        ax.plot(sat_time, y_hat_fit(sat_time, *param_vals), 'b-', label='data')
        ax.plot(sat_time, y_ikj_df, 'b*', label='data')
        ax.set_title('Replicate = {} Curve Fit, Concentration = {} µmolar, ppm = {}'.format(r, c, ppm))
        
    ax.set_xlabel('NMR Saturation Time (s)')
    ax.set_ylabel('I/Io')
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig.tight_layout()
    # export to file
    fig.savefig(filename, dpi=300)
    
    return
    