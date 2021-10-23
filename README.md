# disco-data-processing
This repository contains the code by the Frank Gu Lab for DISCO-NMR data processing. 

<a>![](https://media.tenor.com/images/dedb6f501250b912f125112d6a04a26e/tenor.gif)</a>

This code transforms outputs from individual DISCO NMR experiments in MestreNova into a statistically validated, clean Pandas DataFrame of true positive and true negative polymer-proton binding observations for further use in machine learning. The purpose of this repository is to create a centralized location for lab members to: develop code for DISCO-NMR data processing, request additional features for the code, track bugs, and provide version control.

If you have a feature you would like to request, or have observed a bug in the code, please submit your comments through an "Issue" under the Issues tab.  

<h3> <b> Data Process Description </b> </h3>

- Part 1 : Reading and Cleaning Data      (reads and prepares data for statistical analysis)
- Part 2 : Statistical Analysis           (classifies true positive binding proton observations, generates AFo plots, writes outputs to file)
- Part 3 : Generate DataFrame + Export to Excel     (merges true positive and true negative observations classified from Part 2 into ML-ready dataset)

<b>[Read the Pseudocode + Stats Description for Part 1 and Part 2 Here](https://utoronto.sharepoint.com/:b:/r/sites/fase-che-fgl-nano/DISCOML/Shared%20Documents/Filesharing/disco-data-processing-pseudocode.pdf?csf=1&web=1&e=Ye55Bj)</b>

 
<h3><b> Run Setup </b></h3>

<b>Your Directory Should Look Like:  </b>    
- src/disco-data-processing.py <b> (main program) </b>
- src/discoprocess <b> (package) </b>
- src/setup.py <b> (package creator) </b>
- src/requirements.txt <b> (environment setup) </b>
- data/input/"raw_book_with_a_short_title_you_like.xlsx" (i.e. "PAA.xlsx")

<b> Modules inside discoprocess include: </b>
- \_\_init__.py <b> (package identifier) </b>
- data_wrangling_functions.py
- data_merging.py
- data_analyze.py
- data_plot.py
- data_wrangling_helpers.py

The other files in this hub are just exemplary outputs based on the inputs in the input folder.

Prior to running the script, please ensure you have all the required dependencies by navigating to the src directory and run ```pip install -r requirements.txt```. This repository also uses a custom package, discoprocess, which can be installed on your local computer/environment by running ```python setup.py install``` in the src directory, which can then be imported from any directory. This will install all the necessary packages to run this repository. Note that by installing discoprocess, any changes in the source code will not be reflected unless the package is uninstalled and reinstalled.

Insert all the books you would like to be analyzed inside the input directory. The code will create custom output folders per polymer experiment based on the name of the input books, so nothing overwrites. This code supports: 

- "Book" formatted experimental data (See PAA.xlsx in the input folder for example - one polymer per excel book) 
- "Batch" formatted experimental data (See (Batch 1), (Batch 2) files in the input folder - many polymers per excel book). 

<h3><b> Special Input Data Formatting Requirements </b></h3>
For Batch Format inputs, please ensure unique polymer replicates intended to be analyzed together follow the same naming format. 
For example, if there are 4 total CMC replicates, 3 from one experiment to be analyzed together, and 1 from a separate experiment that is NOT intended as 
a replicate of the other three, the sheet tabs should be named as follows: 

- CMC (1), CMC (2), CMC (3)               (These 3 will be analyzed together, as their name string is the same, and all have a space and brackets as delimeters.)
- CMC_other                               (The 4th CMC tab will be treated separately, as it is named with either a different string or delimeter (both in this case)

<h3><b> Running the Code </b></h3>   
Simply run the disco-data-processing.py file after preparing the input as described above on your local machine.

    python disco-data-processing.py

<h3><b> Results </b></h3>
The ultimate merged dataset will be available ("merged_binding_dataset.xlsx") as an Excel file in output/merged. 

## Unit Tests
<h3><b> Running Pytest </b></h3>
If discoprocess is installed on your local computer/environment, pytest can be run from tests or from its parent directory (so long as tests can be found in a recursive search).

To run all tests:

    python pytest

 To run a specific module:
 
    python pytest <<module.py>>
   
To run a specific class/function (substring matching, regex):
 
    python pytest -k "class_name or function_name"
   
<h3><b> Running Doctest </b></h3>
Doctests are written within the function docstring itself. The modules in the repository that contain doctests include make_dataset.py, crossval.py, build_features.py, and split.py. To run the doctests of a specific module:

    python <<module_name>>.py

If nothing is printed to the terminal, the tests all passed successfully.