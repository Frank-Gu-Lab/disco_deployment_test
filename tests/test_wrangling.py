import pytest
import sys
import os
import glob
import shutil
import numpy as np

# appending path to access sibling directory
sys.path.append(os.getcwd() + '\\..\\src')

from data_wrangling_functions import *

class TestDataFrameConversion:
    """This class contains all the unit tests relating to the dataframe conversion functions, batch_to_dataframe and book_to_dataframe."""
    
    def test_batch(self):
        """Testing overall functionality. Takes in a batch Excel sheet and converts each sheet into a dataframe, returning a tuple of the form
        (df_name, df)."""
        
        path = "./test-files/test_wrangling"

        try:
            
            batch = path + "/input/batch_to_dataframe_input.xlsx"
            
            # loop through names and assert equality of dataframes
            
            actual = batch_to_dataframe(batch)
            file = open(path + "/expected/batch_to_dataframe_output.txt")
            
            for i in range(len(actual)):
                
                # testing equality of sheet names
                expected_name = file.readline()
                assert actual[i][0] == expected_name
                
                # testing equality of dataframes
                name = f"sheet_{i}"
                expected_df = pd.read_excel(path + "/expected/" + name, index_col=0)
                
                assert actual[i][1].equals(expected_df)
            
        except:
            
            print("Program did not successfully execute!")
        
    def test_book(self):
        """Testing overall functionality. Takes in a book Excel sheet and converts it into a dataframe. The created 
        excel sheet is removed during teardown."""
        
        path = "./test-files/test_wrangling"
        output_dir = path +"/output"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        try:
            
            book = path + "/input/KHA.xlsx"
            
            actual = book_to_dataframe(book, output_dir)
            actual_title = actual[0]
            actual_df = actual[1]
            
            expected_title = "KHA"
            expected_df = pd.read_excel(path + "/expected/book_to_dataframe_output.xlsx", index_col=0)
            
            assert actual_title == expected_title 
            pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)
            
        finally:
            
            # TEARDOWN
            
            shutil.rmtree(output_dir)

class TestClean:
    
    def test_clean_batch_list(self):
             
        path = "./test-files/test_wrangling"
        
        input_list = glob.glob(path + "/input/clean_batch_input/*")
        
        # recreating input list
        clean_dfs = []
        
        for file in input_list:
            
            name = os.path.basename(file)
            name = name.split(sep=".")
            name = name[0]       
            
            df = pd.read_excel(file, index_col=0)
            
            clean_dfs.append((name, df))
            
        clean_dfs = [clean_dfs]
        
        actual = clean_the_batch_tuple_list(clean_dfs)
        
        output_list = glob.glob(path + "/expected/clean_batch_output/*")
        
        assert len(actual) == len(output_list)
        
        for i in range(len(actual)):
            actual_df = actual[i]
            expected_df = pd.read_excel(output_list[i], index_col=0)
            
            pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)
        
class TestAttenuation:
    """This class contains all the unit tests relating to the add_attenuation and add_corr_attenuation functions."""
    '''
    def test_add_attenuation_batch(self):
        
        df = pd.read_excel("./test-files/test_wrangling/input/att_batch_input.xlsx", index_col=0)
           
        actual = add_attenuation(df, 'batch')
        expected = pd.read_excel("./test-files/test_wrangling/expected/att_batch.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual, expected)
    '''
    def test_add_attenuation_book(self):
        """ MAKE NOTE OF PRECISION CHANGE """
        df = pd.read_excel("./test-files/test_wrangling/input/att_book_input.xlsx", index_col=0)
        
        actual_true, actual_false = add_attenuation(df)
        expected_true = pd.read_excel("./test-files/test_wrangling/expected/att_book_true.xlsx", index_col=0)
        expected_false = pd.read_excel("./test-files/test_wrangling/expected/att_book_false.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual_true, expected_true)
        pd.testing.assert_frame_equal(actual_false, expected_false)
    '''
    def test_add_both_att_batch(self):
        
        df = pd.read_excel("./test-files/test_wrangling/input/att_batch_input.xlsx", index_col=0)
           
        actual = add_attenuation_and_corr_attenuation_to_dataframe(df, 'batch')
        expected = pd.read_excel("./test-files/test_wrangling/expected/att_batch_output.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual, expected)
    '''
    def test_add_both_att_book(self):
        
        df = pd.read_excel("./test-files/test_wrangling/input/att_book_input.xlsx", index_col=0)
        
        actual_true, actual_false = add_attenuation(df)
        actual = add_corr_attenuation(actual_true, actual_false)
        expected = pd.read_excel("./test-files/test_wrangling/expected/corr_att_book.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual, expected)

    '''
    def test_add_corr_attenuation_batch(self):
                
        df = pd.read_excel("./test-files/test_wrangling/input/corr_att_batch_input.xlsx", index_col=0)
        
        actual = add_corr_attenuation(df, 'batch')
        expected = pd.read_excel("./test-files/test_wrangling/expected/corr_att_batch.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual, expected)
    '''
    def test_add_corr_attenuation_book(self):
        
        df_true = pd.read_excel("./test-files/test_wrangling/input/att_book_true.xlsx", index_col=0)
        df_false = pd.read_excel("./test-files/test_wrangling/input/att_book_false.xlsx", index_col=0)
           
        actual = add_corr_attenuation(df_true, df_false)
        expected = pd.read_excel("./test-files/test_wrangling/expected/corr_att_book.xlsx", index_col=0)
        
        pd.testing.assert_frame_equal(actual, expected)
    
#class TestPlots:
    
    #def test_concentration(self):
        
    #def test_ppm(self):
        
class TestPrep:
    
    def test_prep_mean_book(self):
        
        path = "./test-files/test_wrangling"
        
        input_mean = pd.read_excel(path + "/input/prep_mean_book_input.xlsx", index_col=0)
        
        actual = prep_mean(input_mean)
        
        expected_mean_left = pd.read_excel(path + "/expected/prep_mean_book_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        expected_mean_right = pd.read_excel(path + "/expected/prep_mean_book_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    
    def test_prep_mean_batch(self):
        
        path = "./test-files/test_wrangling"
        
        input_mean = pd.read_excel(path + "/input/prep_mean_batch_input.xlsx", index_col=0)
        
        actual = prep_mean(input_mean, 'batch')
        
        expected_mean_left = pd.read_excel(path + "/expected/prep_mean_batch_output.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
        expected_mean_right = pd.read_excel(path + "/expected/prep_mean_batch_output.xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    
    def test_prep_replicates_book(self):
        
        path = "./test-files/test_wrangling"
        
        input_replicate = pd.read_excel(path + "/input/prep_replicate_book_input.xlsx", index_col=0)
        
        actual = prep_replicate(input_replicate)
        
        expected = pd.read_excel(path + "/expected/prep_replicate_book_output.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual, expected)
    
    def test_prep_replicates_batch(self):
        
        path = "./test-files/test_wrangling"
        
        input_replicate = pd.read_excel(path + "/input/prep_replicate_batch_input.xlsx", index_col=0)
        
        actual = prep_replicate(input_replicate, 'batch')
        
        expected = pd.read_excel(path + "/expected/prep_replicate_batch_output.xlsx", index_col=0)

        pd.testing.assert_frame_equal(actual, expected)

class TestT:
    
    def test_t(self):
        
        path = "./test-files/test_wrangling"
        
        df = pd.read_excel(path + "/input/t_test_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        df_other = pd.read_excel(path + "/input/t_test_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        df_other.columns = pd.MultiIndex.from_product([df_other.columns, ['']])
        input_df = pd.merge(df, df_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
        
        actual = t_test(input_df)

        expected_left = pd.read_excel(path + "/expected/t_test_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        expected_right = pd.read_excel(path + "/expected/t_test_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        expected_right.columns = pd.MultiIndex.from_product([expected_right.columns, ['']])
        expected = pd.merge(expected_left, expected_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        pd.testing.assert_frame_equal(actual, expected)
    
class TestAF:
    
    def test_af(self):
        
        path = "./test-files/test_wrangling"
        
        df_mean = pd.read_excel(path + "/input/af_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        df_mean_other = pd.read_excel(path + "/input/af_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
        mean = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
        
        df_replicate = pd.read_excel(path + "/input/af_replicates_input.xlsx", index_col=0)
        
        actual_mean, actual_replicates = compute_af(mean, df_replicate, 10)
        
        expected_mean_left = pd.read_excel(path + "/expected/af_mean_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
        expected_mean_right = pd.read_excel(path + "/expected/af_mean_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
        expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
        expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

        expected_replicates = pd.read_excel(path + "/expected/af_replicates_output.xlsx", index_col=0)

        assert actual_mean.equals(expected_mean)
        assert actual_replicates.equals(expected_replicates)
        
class TestDropBadPeaks:
    """This class contains all the unit tests relating to the execute_curvefit function."""
    # testing overall functionality
    
    def test_drop_peaks_book(self):
        
        path = "./test-files/test_wrangling"
        output_dir = path + "/output"
        df_title = "KHA"
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        try:
   
            df_mean = pd.read_excel(path + "/input/drop_mean_peaks_book_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
            df_mean_other = pd.read_excel(path + "/input/drop_mean_peaks_book_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
            df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
            mean = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

            df_replicates = pd.read_excel(path + "/input/drop_replicates_peaks_book_input.xlsx", index_col=0)
            
            actual_mean, actual_replicates = drop_bad_peaks(mean, df_replicates, df_title, output_dir)
            
            expected_mean_left = pd.read_excel(path + "/expected/drop_mean_peaks_book_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
            expected_mean_right = pd.read_excel(path + "/expected/drop_mean_peaks_book_output.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
            expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
            expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
  
            expected_replicates = pd.read_excel(path + "/expected/drop_replicates_peaks_book_output.xlsx", index_col=0)
            
            assert actual_mean.equals(expected_mean)
            assert actual_replicates.equals(expected_replicates)
        
        finally:
            
            # TEARDOWN
            
            shutil.rmtree(output_dir)
        
    #def test_drop_peaks_batch(self):      
    
class TestCurveFit:
    """This class contains all the unit tests relating to the execute_curvefit function."""
    # testing overall functionality
    
    def test_curvefit_batch(self):
        
        path = "./test-files/test_wrangling"
        
        if not os.path.exists(path + "/output"):
            os.mkdir(path + "/output")
        
        try:
            
            input_mean = glob.glob(path + "/input/batch_curve_input/mean/*")
            input_replicates = glob.glob(path + "/input/batch_curve_input/replicates/*")
            
            names = open(path + "/input/batch_curve_input.txt")
            names = names.readlines()
            names = [name.rstrip() for name in names]
            
            empty = open(path + "/input/empty.txt")
            empty = empty.readlines()
            empty = [name.rstrip() for name in empty]
            empty = [name.replace(".xlsx", "") for name in empty]
            empty = [name.replace("sheet_", "") for name in empty]
            
            for i in range(len(names)):
                
                df_title = names[i]
                
                output_curve = "{}/output/curve_fit_plots_from_{}".format(path, df_title)
                output_table = "{}/output/data_tables_from_{}".format(path, df_title)
                
                if not os.path.exists(output_curve):
                    os.mkdir(output_curve)
                if not os.path.exists(output_table):
                    os.mkdir(output_table)
                
                # create empty Excel if df_title in empty
                
                if df_title in empty: # this raises an error -- empty df not properly created? bc of the multi-index column I think
                    continue
                    #df_mean = pd.DataFrame(columns = [("corr_%_attenuation", "mean"), ("corr_%_attenuation", "std"), ("dofs", ""), ("sample_size", ""), ("t_results", ""), ("significance", ""), ("amp_factor", "")], index=[])
                
                else:
                    df_mean_left = pd.read_excel(path + "/input/batch_curve_input/mean/sheet_" + df_title + ".xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
                    df_mean_right = pd.read_excel(path + "/input/batch_curve_input/mean/sheet_" + df_title + ".xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
                    df_mean_right.columns = pd.MultiIndex.from_product([df_mean_right.columns, ['']])
                    df_mean = pd.merge(df_mean_left, df_mean_right, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))
                        
                df_replicates = pd.read_excel(path + "/input/batch_curve_input/replicates/sheet_" + df_title +".xlsx", index_col=0)
                
                df_mean, df_replicates = execute_curvefit(df_mean, df_replicates, output_curve, output_table, df_title, 'batch')
                
                expected_mean_left = pd.read_excel(path + "/expected/batch_curve_output/mean/sheet_" + df_title + ".xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, :2]
                expected_mean_right = pd.read_excel(path + "/expected/batch_curve_output/mean/sheet_" + df_title + ".xlsx", header = [0, 1], index_col=[0, 1, 2, 3]).iloc[:, 2:].droplevel(1, axis=1)
                expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
                expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index", "ppm"), right_on=("concentration", "sat_time", "proton_peak_index", "ppm"))

                expected_replicates = pd.read_excel(path + "/expected/batch_curve_output/replicates/sheet_" + df_title + ".xlsx", index_col=0)
                
                pd.testing.assert_frame_equal(df_mean, expected_mean, rtol=1e-3)
                pd.testing.assert_frame_equal(df_replicates, expected_replicates, rtol=1e-3)
            
        finally:
            
            # TEARDOWN
            
            shutil.rmtree(path + "/output")
    
    def test_curvefit_book(self):  
        
        path = "./test-files/test_wrangling"
        df_title = "KHA"
        output_curve = "{}/output/curve_fit_plots_from_{}".format(path, df_title)
        output_table = "{}/output/data_tables_from_{}".format(path, df_title)
        
        if not os.path.exists(path + "/output"):
            os.mkdir(path + "/output")
        if not os.path.exists(output_curve):
            os.mkdir(output_curve)
        if not os.path.exists(output_table):
            os.mkdir(output_table)
        
        try:
            
            df_mean = pd.read_excel(path + "/input/book_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
            df_mean_other = pd.read_excel(path + "/input/book_mean_input.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
            df_mean_other.columns = pd.MultiIndex.from_product([df_mean_other.columns, ['']])
            mean = pd.merge(df_mean, df_mean_other, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))
            
            df_replicates = pd.read_excel(path + "/input/book_replicates_input.xlsx")
            
            actual_mean, actual_replicates = execute_curvefit(mean, df_replicates, output_curve, output_table, df_title)
            
            expected_mean_left = pd.read_excel(path + "/expected/book_meancurve.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, :4]
            expected_mean_right = pd.read_excel(path + "/expected/book_meancurve.xlsx", header = [0, 1], index_col=[0, 1, 2]).iloc[:, 4:].droplevel(1, axis=1)
            expected_mean_right.columns = pd.MultiIndex.from_product([expected_mean_right.columns, ['']])
            expected_mean = pd.merge(expected_mean_left, expected_mean_right, left_on=("concentration", "sat_time", "proton_peak_index"), right_on=("concentration", "sat_time", "proton_peak_index"))

            expected_replicates = pd.read_excel(path + "/expected/book_replicatescurve.xlsx")

            msg1 = "Curve fits for mean do not match."
            msg2 = "Curve fits for replicates do not match."
            
            pd.testing.assert_frame_equal(actual_mean, expected_mean, rtol=1e-3)
            pd.testing.assert_frame_equal(actual_replicates, expected_replicates, rtol=1e-3)
            
            # check if the same plots are generated (can only compare filepath/name)

            actual_curve = glob.glob(output_curve + "/*")
            actual_table = glob.glob(output_table + "/*")
            
            expected_curve = glob.glob(path + "/expected/curve_fit_plots_from_KHA/*")
            expected_table = glob.glob(path + "/expected/data_tables_from_KHA/*")
            
            if len(actual_curve) != len(expected_curve):
                assert len(actual_curve) == len(expected_curve)
            
            if len(actual_table) != len(expected_table):
                assert len(actual_table) == len(expected_table)
            
            for i in range(len(actual_curve)):
                actual_curve[i] = os.path.basename(actual_curve[i])
                expected_curve[i] = os.path.basename(expected_curve[i])
                
                assert actual_curve[i] == expected_curve[i]
            
            for i in range(len(actual_table)):
                actual_table[i] = os.path.basename(actual_table[i])
                expected_table[i] = os.path.basename(expected_table[i])
                
                assert actual_table[i] == expected_table[i]
                
        finally:
            
            # TEARDOWN
            
            shutil.rmtree(path + "/output")        
