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

def test_move():
    
    # SETUP
    src_path = ".\\test-files\\test_move_normal"
    dst_path = ".\\test-files\\output"
    
    os.mkdir(dst_path)
    
    # grab file names from src
    
    filenames = glob.glob(src_path + "\\*")
    
    move(src_path, dst_path)
    
    try:
        
        for file in filenames: # file = absolute path
            filename = os.path.basename(file)
            assert os.path.isfile(dst_path + "\\" + filename)

    finally:
        
        # TEARDOWN
        shutil.rmtree(dst_path)

################## TESTING MERGE ####################
        
    