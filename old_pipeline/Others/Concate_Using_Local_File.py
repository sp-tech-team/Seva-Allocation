import numpy as np
import pandas as pd
from Gspread_Library import GoogleSheetHandler
from Concatenation_Library import Concatenation_Handler

input_file = 'D:/Seva-Allocation/old_pipeline/content/seva_data_2024_updated.xlsx'
output_file = 'D:/Seva-Allocation/old_pipeline/content/output_transformed.xlsx'

Concatenation_Main_Using_Local_Downloaded_File(input_file, output_file)