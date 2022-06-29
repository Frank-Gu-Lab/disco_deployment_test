import pytest
import sys
import os
import glob
import shutil
import unittest

# appending path to access sibling directory - uncomment if local package setup doesn't work
sys.path.append(os.getcwd() + '/../src')

from discoprocess.data_checker import *

class TestChecks:

    def test_name_checker(self):

        list_of_raw_books = []

        list_of_raw_books.append("C:/Users/matth/OneDrive/Documents/GitHub/disco-data-processing/tests/test-files/test_checker/test_checker_input/Batch-CMC_131k_20uM.xlsx")

        with pytest.raises(Exception) as e:
            name_checker(list_of_raw_books)

        assert e.match("Please add 'uM' to the end of the concentration in the polymer name")

    def test_resonance_and_column_checker(self):

                list_of_raw_books = []

                list_of_raw_books.append("C:/Users/matth/OneDrive/Documents/GitHub/disco-data-processing/tests/test-files/test_checker/test_checker_input/Batch-CMC_90k_20uM.xlsx")

                with pytest.raises(Exception) as e:
                    resonance_and_column_checker(list_of_raw_books)

                assert e.match("In the excel book Batch-CMC_90k_20uM.xlsx please ensure that all odd replicates are On resonance and Range keyword is only used in tables meant for data analysis in sheet CMC_90k_20uM \(1\)")

    def test_range_checker(self):

        list_of_raw_books = []

        list_of_raw_books.append("C:/Users/matth/OneDrive/Documents/GitHub/disco-data-processing/tests/test-files/test_checker/test_checker_input/Batch-PEG_2k_20uM.xlsx")

        with pytest.raises(Exception) as e:
            range_checker(list_of_raw_books)

        assert e.match("In the excel book Batch-PEG_2k_20uM.xlsx please ensure the ranges are equivalent across all tables in sheet PEG_2k_20uM \(1\)")
