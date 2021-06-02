import pytest
import sys
import os
import shutil
import glob
import pandas as pd

# appending path to access sibling directory
sys.path.append(os.getcwd() + '\\..\\src')

from data_merging import move, merge

class TestMove:
    """This class contains all the unit tests relating to the move function."""
    
    def test_move(self):
        """Testing overall functionality, this function takes files from src_path and moves them to dst_path. The function then asserts whether
        the files exist in dst_path, subsequently removing the files as part of teardown."""
        
        # SETUP
        src_path = ".\\test-files\\test_move\\*"
        dst_path = ".\\test-files\\output"
        
        # grab file names from src
        directories = glob.glob(src_path)
        filenames = [glob.glob("{}/*".format(dir)) for dir in directories] # list of lists, each inner list represents files in a directory
        
        os.mkdir(dst_path)

        try:
            
            move(src_path, dst_path)
            
            for i in range(len(filenames)):
                for file in filenames[i]: # file = absolute path
                    filename = os.path.basename(file)
                    msg = "{} could not be found in {}!".format(filename, dst_path)
                    assert os.path.isfile(dst_path + "\\" + filename), msg
            
        finally:
            
            # TEARDOWN
            shutil.rmtree(dst_path)

class TestMerge:
    """This class contains all the unit tests relating to the merge function."""
    
    def test_merge(self):
        """Testing overall functionality. Takes all Excel sheets from src_path and moves to dst_path, from which the function concatenates all 
        sheets together into one Dataframe.
        
        Notes
        -----
        Equality checking ignores datatype matching.
        """
        
        #SETUP
        path = "./test-files/test_merge"
        src_path = path + "/data/KHA/data_tables_from_KHA"
        dst_path = path + "/output"

        os.mkdir(dst_path)

        try:

            actual = merge(src_path, dst_path)

            expected = pd.read_excel(path + "/expected/merged_binding_dataset.xlsx", index_col=0)
            expected.columns.name = 'index' # column.name attribute not saved when exported to Excel file

            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
        
        finally:
            
            # TEARDOWN
            shutil.rmtree(dst_path)