#commenting this out because apparently we don't need these.

import pytest
import sys
import os
import glob
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from contextlib import contextmanager

# appending path to access sibling directory - uncomment if local package setup doesn't work
sys.path.append(os.getcwd() + '/../src')

import importlib
std = importlib.import_module("standard-figures")


def test_grab_polymer_name():

    full_filepath = "Documents/disco-data-processing\\input/hello\\Waffle.xlsx"
    common_filepath = "Documents/disco-data-processing/input/hello/"

    name = std.grab_polymer_name(full_filepath, common_filepath)

    assert name == "Waffle"
