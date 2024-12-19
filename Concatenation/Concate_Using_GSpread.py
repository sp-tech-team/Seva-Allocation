import numpy as np
import pandas as pd
from Libraries.Gspread_Library import GoogleSheetHandler
from Libraries.Concatenation_Library import Concatenation_Handler

# Parameters
credentials_path = "D:/Seva-Allocation/Concatenation/credentials.json"  # Path to service account credentials file
sheet_url = "https://docs.google.com/spreadsheets/d/1DtJEE7e43ePkrGefvn7nl21je2xDGZWxOCKOS8nAxYE/edit?gid=903292404#gid=903292404"
input_tab_name = "input"  # Input tab in the Google Sheet
output_tab_name = "Formatted Input"  # Output tab to write the processed data

print("\n You can download credentials.json from https://console.cloud.google.com/")
print("Google cloud console -> API & Services -> Credentials -> Add Service Account - Add Key and download as Json", end="\n \n")
print("Note: Make sure to give editor access in the sheet for the client_email mentioned in credentials.json", end="\n \n")

# Run the function
Concatenation_Handler.Concatenation_Main_Using_GSpread(sheet_url, input_tab_name, output_tab_name, credentials_path)
