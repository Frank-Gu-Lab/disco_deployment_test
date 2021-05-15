import pytest
import unittest
import sys
import os
import shutil
import glob


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
    
    os.mkdir(dst_path)
    
    # grab file names from src
    
    directories = glob.glob(src_path)
    filenames = [glob.glob("{}/*".format(dir)) for dir in directories]

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
  
#def test_merge():
