import pytest
import sys
import os
import shutil
import glob
import pandas as pd

from helpers import compare_excel

# modifying path to access sibling directory
sys.path.append(os.getcwd() + '\\..')

from src.data_merging import move, merge

################## TESTING MOVE ####################

# testing overall functionality

class TestMove:

    def test_move(self):
    
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
                    msg = "{} could not be found in {}!".format(filename, dst_path)
                    assert os.path.isfile(dst_path + "\\" + filename), msg

        finally:
            
            # TEARDOWN
            shutil.rmtree(dst_path)

################## TESTING MERGE ####################
      
# testing overall functionality

class TestMerge:
    
    def test_merge(self):
    
        src_path = ".\\test-files\\test_merge\\data\\KHA\\data_tables_from_KHA"
        dst_path = ".\\test-files\\test_merge\\output"
        merge_path = ".\\test-files\\test_merge\\actual"
        output_file_name = "merged_binding_dataset.xlsx"
        
        if not os.path.exists(merge_path):
            os.mkdir(merge_path)

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        try:

            output = merge(src_path, dst_path)
            output.to_excel(os.path.join(merge_path, output_file_name))
            
            actual = pd.read_excel(merge_path + "\\" + output_file_name)
            expected = pd.read_excel(".\\test-files\\test_merge\\expected\\merged_binding_dataset.xlsx")

            # check if file is as expected
            msg2 = "{} does not contain all the expected information.".format(merge_path + "\\" + output_file_name)
            assert compare_excel(actual, expected), msg2
        
        finally:
            
            # TEARDOWN
            
            # check if file exists
            msg1 = "{} was not successfully created.".format(merge_path + "\\" + output_file_name)
            assert os.path.exists(merge_path + "\\" + output_file_name), msg1
            
            os.remove(merge_path + "\\" + output_file_name)