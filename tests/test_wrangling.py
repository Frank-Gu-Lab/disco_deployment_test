import pytest
import sys
import os
import glob

# modifying path to access sibling directory
sys.path.append(os.getcwd() + '\\..')

from src.data_wrangling_functions import *

list_of_raw_books = glob.glob("./test-files/test_wrangling/data/*.xlsx")

print(list_of_raw_books[0])