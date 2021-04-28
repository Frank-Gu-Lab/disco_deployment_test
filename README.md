# disco-data-processing
This repository contains the code by the Frank Gu Lab for DISCO-NMR data processing. 

This code transforms outputs from individual DISCO NMR experiments in MaestraNova into a statistically validated, clean Pandas DataFrame of true positive and true negative polymer-proton binding observations for further use in machine learning. The purpose of this repository is to create a centralized location for lab members to: develop code for DISCO-NMR data processing, request additional features for the code, track bugs, and provide version control.

If you have a feature you would like to request, or have observed a bug in the code, please submit your comments through an "Issue" under the Issues tab.  

<h3><b> Disco Data Processing Setup </b></h3>

<b>Your Input Folder Should Look Like:</b>    
- disco-data-processing.py
- data_wrangling_functions.py
- input/"raw_book_with_a_title_you_like.xlsx" (i.e. "PAA.xlsx")

Prior to running the script, please ensure you have inserted all the books you would like to be analyzed inside the input directory. The code will create custom output folders per polymer experiment based on the name of the input books, so nothing overwrites. This code supports polymer data from the NMR analysis in a raw "Book Format" (See PAA.xlsx for example, one polymer per excel book) and a "Batch Format" (See Batch 1, Batch 2 files for an example of what this looks like, many polymers per excel book). 

For Batch Format inputs, please ensure unique observations intended to be analyzed together follow the same naming format. For example, if there are 4 total CMC results, 3 from one experiment to be analyzed together, and 1 from a separate experiment, the sheet tabs should be named as follows: 

CMC (1), CMC (2), CMC (3)               # These 3 will be analyzed together, as their name string is the same, and all have a space and brackets as delimeters.
CMC_other                               # The 4th CMC tab will be treated separately, as it is named with either a different string or delimeter (both in this case)
    
Then simply run this .py script to process the data. 
Part 1 : Reading and Cleaning Data      # prepares the data for statistical analysis
Part 2 : Statistical Analysis           # classify true positive binding proton observations, generate AFo plots (generates more Excel files)
(TO DO) Part 3 : Generate DataFrame     # true positive and true negative observations from Excel books generated in Part 1 and Part 2 are processed and merged into clean dataset 

<b>[Read the Pseudocode + Stats Outline for Part 1 and Part 2 of the Repository Here](https://utoronto.sharepoint.com/:b:/r/sites/fase-che-fgl-nano/DISCOML/Shared%20Documents/Filesharing/disco-data-processing-pseudocode.pdf?csf=1&web=1&e=Ye55Bj)</b>


<a>![](https://media.tenor.com/images/dedb6f501250b912f125112d6a04a26e/tenor.gif)</a>