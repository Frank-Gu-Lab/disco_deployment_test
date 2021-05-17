import pandas as pd 
import numpy as np
import os

def initialize_excel_batch_replicates(b):
    '''
    This function determines the unique polymers contained in an excel book,
    the number of replicates of those polymers, and also returns an iterable of 
    the sheet names without Complete in them.
    
    Parameters
    ----------
    b : str
    
    Returns
    -------
    unique_polymers : list
    
    unique_polymer_replicates : list
    
    all_sheets_complete_removed : list
    
    
    '''
    all_sheets_iterator = []
    all_data_sheets = []
    name_sheets = []
    unique_polymers = []
    unique_polymer_replicates = []
    all_sheets_complete_removed = []
    intermediate_string = []
    match_checker = []
    
    # load excel book into Pandas 
    current_book_title = os.path.basename(str(b))

    # determine the number and names of sheets in the book
    name_sheets = pd.ExcelFile(b).sheet_names
    
    # remove sheets with complete from the list
    all_sheets_iterable = [sheet for sheet in name_sheets]
    
    def DropComplete(x): 
        if "complete" in x: 
            return False 
        else: 
            return True
        
    all_sheets_complete_removed = list(filter(DropComplete, all_sheets_iterable))
    
    # sort all sheets with complete removed alphebetically for generalizable replicate ordering and processing
    all_sheets_complete_removed = sorted(all_sheets_complete_removed)
    
    # generate a list of unique polymers in the book 
    for sheet in range(len(all_sheets_complete_removed)):
        # drop string after brackets
        intermediate_string = all_sheets_complete_removed[sheet].split('(', 1)[0]
    
        #if there's a trailing space after dropping the bracket, remove it as well
        if intermediate_string[-1] == ' ':
            intermediate_string = intermediate_string[0:-1]
        
        unique_polymers.append(intermediate_string) 
    
    # drop duplicates to generate a unique polymers list
    unique_polymers = list(dict.fromkeys(unique_polymers))
    
    # initialize zero array that corresponds to each unique_polymer
    unique_polymer_replicates = np.zeros(len(unique_polymers))
    
    # calculate the number of replicates of the polymers in the book by matching unique polymers to sheet names
    for i in range(len(unique_polymers)):
        for j in range(len(all_sheets_complete_removed)):
        
            #calculate the current match checker the same way unique polymers were calculated
            match_checker = all_sheets_complete_removed[j].split('(', 1)[0]
            if match_checker[-1] == ' ':
                match_checker = match_checker[0:-1]

            #if unique polymer name matches the checker, increment replicate counter for that polymer
            if (unique_polymers[i] == match_checker):
                unique_polymer_replicates[i] +=1
    
    return unique_polymers, unique_polymer_replicates, all_sheets_complete_removed    
