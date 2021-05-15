import pytest
import sys
import unittest
import os

# modifying path to access sibling directory
sys.path.append(os.getcwd() + '\\..')

from src.data_merging import move, merge

class TestMerge(unittest.TestCase):
    
    def something(self):
        return
        
class TestMove(unittest.TestCase):
    
    """@pytest.fixture
    def setup(self, tmpdir):
        #
    
    def test_one_row(self, setup):
        assert 1 == 1
        
    def test_not_excel(self):
        
        with pytest.raises(ValueError):
            move(path, out)
        
        
    def test_multi_row(self, setup):
        
    def 
    """        