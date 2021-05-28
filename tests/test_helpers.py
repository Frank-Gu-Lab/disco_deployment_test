import pytest
import sys
import os
import glob
import shutil

# appending path to access sibling directory
sys.path.append(os.getcwd() + '\\..\\src')

from data_wrangling_functions import *

class TestDrop:
    
    def test_drop(self):
        
        s = "Complete"
        
        assert not DropComplete(s)
        
        t = "complete"
        
        assert not DropComplete(t)

        u = "another word"
        
        assert DropComplete(u)

#class TestInitialize:

#class TestWrangle:

    #def test_wrangle_batch(self):
    
    #def test_wrangle_book(self):

class TestCount:
    
    def test_count(self):
        
        path = "./test-files/test_helpers"
        
        input_list = []
        
        with open(path + "/input/count_input.txt") as file:
            
            for line in file.readlines():
                input_list.append(line.rstrip())
                
        num_samples, num_controls, sample_control_initializer, sample_replicate_initializer, control_replicate_initializer = count_sheets(input_list)

        assert num_samples == 3
        assert num_controls == 3
        
        expected_sample_control = []
        
        with open(path + "/input/sample_control.txt") as file:
            for line in file.readlines():
                expected_sample_control.append(line.rstrip())
                
        assert sample_control_initializer == expected_sample_control

        expected_sample_replicate = []

        with open(path + "/input/sample_replicate.txt") as file:
            for line in file.readlines():
                expected_sample_replicate.append(int(line.rstrip()))
                
        assert sample_replicate_initializer == expected_sample_replicate
        
        expected_control_replicate = []
        
        with open(path + "/input/control_replicate.txt") as file:
            for line in file.readlines():
                expected_control_replicate.append(int(line.rstrip()))
                
        assert control_replicate_initializer == expected_control_replicate

class TestEqualityChecker:

    def test_checker_book(self):

        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/checker1_book.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/checker2_book.xlsx", index_col=0)    
        
        assert attenuation_calc_equality_checker(df1, df2)
        
    def test_checker_batch(self):
        
        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/checker1_batch.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/checker2_batch.xlsx", index_col=0)    
        
        assert attenuation_calc_equality_checker(df1, df2, 'batch')
    
    def test_corrected_checker(self):
        
        path = "./test-files/test_helpers/input"
        
        df1 = pd.read_excel(path + "/corr_checker1.xlsx", index_col=0)
        df2 = pd.read_excel(path + "/corr_checker2.xlsx", index_col=0)
        df3 = pd.read_excel(path + "/corr_checker3.xlsx", index_col=0)
        
        assert corrected_attenuation_calc_equality_checker(df1, df2, df3)
    
#class TestDofs:

#class TestYHat: