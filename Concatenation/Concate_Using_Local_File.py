import numpy as np
import pandas as pd
from Libraries.Gspread_Library import GoogleSheetHandler
from Libraries.Concatenation_Library import Concatenation_Handler

input_file = 'D:/Seva-Allocation/Concatenation/content/Input.xlsx'
output_file = 'D:/Seva-Allocation/Concatenation/content/output_transformed.xlsx'

Concatenation_Handler.Concatenation_Main_Using_Local_Downloaded_File(input_file, output_file)