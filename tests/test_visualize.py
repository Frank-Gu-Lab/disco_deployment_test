import pytest 
import os
import sys
import shutil

# appending path to access sibling directory
sys.path.append(os.getcwd() + '\\..\\src')

from data_visualize import *

class TestGenerateDirectory:
    
    def test_generate_directories(self):
        
        current_df_title = "KHA"
        global_output_directory = "./test-files/test_visualize/output"
        
        try:
        
            output_directory_exploratory, output_directory_curve, output_directory_tables, output_directory = generate_directories(current_df_title, global_output_directory)
            
            assert os.path.exists(output_directory)
            assert os.path.exists(output_directory_curve)
            assert os.path.exists(output_directory_exploratory)
            assert os.path.exists(output_directory_tables)
            
        finally:
            
            # TEARDOWN

            shutil.rmtree(global_output_directory + "/KHA")