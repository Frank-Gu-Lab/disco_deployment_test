import pytest
import sys
import os
import shutil
import glob
import pandas as pd

# appending path to access sibling directory - uncomment if local package setup doesn't work
#sys.path.append(os.getcwd() + '/../src')

from discoprocess.data_merging import *

# global testing directory
path = os.path.dirname(__file__) + "/test-files/test_merge_move"
merge_path = path + "/test_merge"
move_path = path + "test_move"

@pytest.fixture(scope='function')
def remove():
    
    output_dir = path + "/output"
    os.mkdir(output_dir)
    
    yield output_dir
    
    shutil.rmtree(output_dir)

class TestMove:
    """This class contains all the unit tests relating to the move function."""
    
    def test_move(self, remove):
        """Testing overall functionality, this function takes files from src_path and moves them to dst_path. The function then asserts whether
        the files exist in dst_path, subsequently removing the files as part of teardown."""
        
        # SETUP
        src_path = move_path + "/*"
        dst_path = remove
        
        # grab file names from src
        directories = glob.glob(src_path)
        filenames = [glob.glob("{}/*".format(dir)) for dir in directories] # list of lists, each inner list represents files in a directory
        
        move(src_path, dst_path)
        
        for i in range(len(filenames)):
            for file in filenames[i]: # file = absolute path
                filename = os.path.basename(file)
                msg = "{} could not be found in {}!".format(filename, dst_path)
                assert os.path.isfile(dst_path + "\\" + filename), msg
            
class TestClean:
    """ This class contains all unit tests relating to the clean function. """

    def test_clean_pos(self):
        """ Tests whether the dataframes with positive binding observations are cleaned as expected. """
        
        # recreate input list
        
        dfs = glob.glob(merge_path + "/input/clean_pos_input/*")

        for i in range(len(dfs)):
        
            try: # ppm in index
                # Preserve multi-index when reading in Excel file
                df = pd.read_excel(dfs[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
                df_other = pd.read_excel(dfs[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
                df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
                dfs[i] = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

            except: # ppm in column
                # Preserve multi-index when reading in Excel file
                df = pd.read_excel(dfs[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
                df_other = pd.read_excel(dfs[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
                df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
                dfs[i] = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        polymer_list = open(merge_path + "/input/clean_pos_polymer.txt").readlines()
        polymer_list = [l.rstrip() for l in polymer_list]
        
        # call function --> original list is modified (mutable)
        clean(dfs, polymer_list, True)        
        
        # recreate output list
        expected_dfspaths = glob.glob(merge_path + "/expected/clean_pos_output/*")
        expected_dfs = [pd.read_excel(df, index_col=[0, 1, 2, 3]) for df in expected_dfspaths]

        # compare
        assert len(dfs) == len(expected_dfs)

        for i in range(len(dfs)):
            pd.testing.assert_frame_equal(dfs[i], expected_dfs[i])
    
    def test_clean_neg(self):
        """ Tests whether the dataframes with negative binding observations are cleaned as expected. """

        # recreate input list        
        dfs = sorted(glob.glob(merge_path + "/input/clean_neg_input/*"), key=lambda x : int(os.path.basename(x)[6:-5]))

        for i in range(len(dfs)):
            
            try: # ppm in index
                # Preserve multi-index when reading in Excel file
                df = pd.read_excel(dfs[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
                df_other = pd.read_excel(dfs[i], header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
                df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
                dfs[i] = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

            except: # ppm in column
                # Preserve multi-index when reading in Excel file
                df = pd.read_excel(dfs[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
                df_other = pd.read_excel(dfs[i], header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
                df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
                dfs[i] = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        polymer_list = open(merge_path + "/input/clean_neg_polymer.txt").readlines()
        polymer_list = [l.rstrip() for l in polymer_list]
        
        # call function --> original list is modified (mutable)
        clean(dfs, polymer_list, False)        
        
        # recreate output list
        expected_dfspaths = sorted(glob.glob(merge_path + "/expected/clean_neg_output/*"), key=lambda x : int(os.path.basename(x)[6:-5]))
        expected_dfs = [pd.read_excel(df, index_col=[0, 1, 2, 3]) for df in expected_dfspaths]

        # compare
        assert len(dfs) == len(expected_dfs)
        
        for i in range(len(dfs)):
            pd.testing.assert_frame_equal(dfs[i], expected_dfs[i])

class TestReformat:
    """ This class contains all the unit tests relating to the reformt functions. """
    
    def test_reformat_pos(self):
        """ Takes in a list of dataframes containing positive binding data, concatenates them, and returns the reformatted dataframe. """
         
        # recreating input list
        df_list = glob.glob(merge_path + "/input/reformat_pos_input/*")
        df_list = [pd.read_excel(df, index_col=[0, 1, 2, 3]) for df in df_list]
        
        actual = reformat(df_list, True)
        
        expected = pd.read_excel(merge_path + "/expected/reformat_pos_output.xlsx", index_col=0)
        expected.columns.names = ['index'] # column name not included when reading in Excel file
        
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        
    def test_reformat_neg(self):  
        """ Takes in a list of dataframes containing negative binding data, concatenates them, and returns the reformatted dataframe. """
         
        # recreating input list
        df_list = sorted(glob.glob(merge_path + "/input/reformat_neg_input/*"), key=lambda x : int(os.path.basename(x)[6:-5]))
        df_list = [pd.read_excel(df, index_col=[0, 1, 2, 3]) for df in df_list]

        actual = reformat(df_list, False)

        expected = pd.read_excel(merge_path + "/expected/reformat_neg_output.xlsx", index_col=0)
        expected.columns.name = 'index' # column.name attribute not saved when exported to Excel file

        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        
class TestJoin:
    """ This class contains all the unit tests relating to the join function. """
    
    def test_join(self):
        """ Takes in two dataframes and joins them together. 
        
        Notes
        -----
        Equality checking ignores datatype matching.
        """
        
        df1 = pd.read_excel(merge_path + "/input/join_input1.xlsx", index_col=0)
        df2 = pd.read_excel(merge_path + "/input/join_input2.xlsx", index_col=0)
        
        actual = join(df1, df2)
        
        expected = pd.read_excel(merge_path + "/expected/join_output.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=False)
        
class TestMerge:
    """ This class contains all the unit tests relating to the merge function."""
        
    def test_merge(self, remove, mocker):
        ''' Checks whether all dependencies are called and that both positive (True) and negative (False) observations are read.
        
        Notes
        -----
        All dependencies are mocked with pytest-mock.
        '''
        source_path = merge_path + "/input/KHA/data_tables_from_KHA"
        destination_path = source_path
        
        mock1 = mocker.patch("discoprocess.data_merging.move")
        mock2 = mocker.patch("discoprocess.data_merging.clean")
        mock3 = mocker.patch("discoprocess.data_merging.reformat")
        mock4 = mocker.patch("discoprocess.data_merging.join")
                
        merge(source_path, destination_path)
        
        # checking call count
        mock1.assert_called_once()
        assert mock2.call_count == 2
        assert mock3.call_count == 2
        mock4.assert_called_once()
        
        # clean and reformat called with True and False each
        assert mock2.call_args_list[0][0][-1] == True
        assert mock2.call_args_list[1][0][-1] == False
        assert mock3.call_args_list[0][0][-1] == True
        assert mock3.call_args_list[1][0][-1] == False
        
        # somehow figure out how to check file names
        