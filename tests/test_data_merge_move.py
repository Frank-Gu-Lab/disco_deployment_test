import pytest
import sys
import os
import shutil
import glob
import pandas as pd
from . import helpers

# modifying path to access sibling directory
sys.path.append(os.getcwd() + '\\..')

from src.data_merging import move, merge

# FIXTURES DON'T WORK WHEN CLASSES ARE DEFINED IDK WHY

################## TESTING MOVE ####################

# testing overall functionality

def test_move():
    
    # SETUP
    
    src_path = ".\\test-files\\test_move\\*"
    dst_path = ".\\test-files\\output"
    
    # grab file names from src
    
    directories = glob.glob(src_path)
    filenames = [glob.glob("{}/*".format(dir)) for dir in directories]
    
    os.mkdir(dst_path)

    try:
        
        move(src_path, dst_path)
        
        for i in range(len(filenames)):
            for file in filenames[i]: # file = absolute path
                filename = os.path.basename(file)
                assert os.path.isfile(dst_path + "\\" + filename)

    finally:
        
        # TEARDOWN
        shutil.rmtree(dst_path)

################## TESTING MERGE ####################
      
# testing overall functionality
  
def test_merge():

    src_path = ".\\test-files\\test_merge\\data\\*"
    dst_path = ".\\test-files\\test_merge\\output"
    merge_path = ".\\test-files\\test_merge\\actual"

    try:

        output = merge(src_path, dst_path)
        output_file_name = "merged_binding_dataset.xlsx"
        output.to_excel(os.path.join(merge_path, output_file_name))
        
        actual = pd.read_excel(merge_path + "\\" + output_file_name)
        expected = pd.read_excel(".\\test-files\\test_merge\\expected\\merged_binding_dataset.xlsx")
        
        assert helpers.compare_excel(actual, expected)
        
    
    finally:
        
        # TEARDOWN
        
        os.remove(merge_path + "\\" + output_file_name)
    