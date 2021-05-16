import pandas as pd

def compare_excel(ex1, ex2):
    """
    Takes in two excel files (read as Pandas DataFrame) and checks whether they have the same content (are identical).
    
    Parameters
    ----------
    ex1, ex2 : Pandas.DataFrame
    
    Returns
    -------
    boolean
        True, if identical, otherwise False
        
    Notes
    -----
    Assumes one column for index and no stacking. Very very simple df -- maybe edit
    """
    
    # same shape
    
    if ex1.shape != ex2.shape:
        return False
    
    # same columns
    
    for i in range(len(ex1.columns)):
        if ex1.columns[i] != ex2.columns[i]:
            return False
    # same index
    
    for j in range(len(ex1.index)):
        if ex1.index[j] != ex2.index[j]:
            return False
        
    # compare values
    
    for i in range(ex1.shape[0]):
        for j in range(ex2.shape[1]):
            if ex1.iloc[i][j] != ex2.iloc[i][j]:
                return False
    
    return True