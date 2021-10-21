from math import sqrt
import matplotlib.pyplot as plt   
from matplotlib import rc
import seaborn as sns
import os
import numpy as np

plt.style.use(['science'])
plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.size':12})

# must install LaTex before Science Plots
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

try:
    from .data_wrangling_helpers import y_hat_fit
except:
    from data_wrangling_helpers import y_hat_fit

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
    output_file_name_conc = "{}/conc_{}.png".format(output_directory_exploratory, current_df_title)
    
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
    output_file_name_ppm = "{}/ppm_{}.png".format(output_directory_exploratory, current_df_title)

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
    
def generate_buildup_curve(df):
    '''Generates the formal STD buildup curve for the figure.
    
    Parameters:
    -----------
    df : Pandas.Dataframe
        the mean stats analysis output for one polymer in the library
   
    Returns:
    -------
    None, outputs formal figures
    '''

    fig, (ax) = plt.subplots(1, figsize=(8, 4))

    # Disco Effect
    y = df['corr_%_attenuation']['mean']

    y_stderr = df['corr_%_attenuation']['std'] / sqrt(df['sample_size']['Unnamed: 7_level_1'])

    # saturation times
    x = np.unique(df.index.get_level_values(1))

    # line
    # ax.plot(x, y, ls='-', color='#377eb8',
    #         label=f'{c}$\\mu$M - {ppm} ppm')

    # raw points
    ax.plot(x, y, marker='.', ls='', color='#377eb8',
            markeredgecolor='k', markeredgewidth=0.25, label=f'{c}$\\mu$M - {ppm} ppm')

    # TO DO: ADD STD ERROR
    ax.fill_between(x,np.array(y)+np.array(y_stderr),np.array(y)-np.array(y_stderr),facecolor = '#377eb8', alpha=0.25)

    plt.show()


    return

