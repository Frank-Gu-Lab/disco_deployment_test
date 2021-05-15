import sys
import os

import importlib  

sys.path.append(os.getcwd() + '\\..')

process = importlib.import_module("src.disco-data-processing", package="src")

from src.data_merging import move, merge
import process