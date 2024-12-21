import numpy as np
import pandas as pd
from Join_Library.Concatenation_Library import Concatenation_Handler

input_file = 'D:/Seva-Allocation/Concatenation/content/Input.xlsx'
output_file = 'D:/Seva-Allocation/Concatenation/content/output_transformed.xlsx'

Concatenation_Handler.Concatenation_Main_Using_Local_Downloaded_File(input_file, output_file)

# Load the first and third tabs
df_main = pd.read_excel(output_file, sheet_name = 0)  # Concatenated output - first tab
df_Right_Data = pd.read_excel(input_file, sheet_name = 1)  # Data that you need to join with the main - second tab

# Perform a left join of df_main with df_exported_filtered
df_joined = df_main.merge(df_Right_Data, on='SP ID', how='left')

# Write the output to a new Excel file with multiple tabs
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_joined.to_excel(writer, sheet_name='Joined Data', index=False)  # Write the joined data to another tab

print(f"Transformation complete. Output written to Joined Data tab")