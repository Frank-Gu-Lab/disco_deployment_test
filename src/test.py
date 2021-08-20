import os
import pandas as pd
from discoprocess.data_merging import merge

global_output_directory = "../data/output"
merge_output_directory = "{}/merged".format(global_output_directory)

source_path = '{}/*/data_tables_from_*'.format(global_output_directory)
destination_path = '{}'.format(merge_output_directory)

# call data merging function and write complete dataset to file
merged_dataset = merge(source_path, destination_path)
output_file_name = "merged_binding_dataset.xlsx"
merged_dataset.to_excel(os.path.join(merge_output_directory, output_file_name))
print("Data processing completed! Merged_binding_dataset.xlsx file available in the output directory under merged.")
