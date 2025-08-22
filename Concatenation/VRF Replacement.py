import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from Libraries.Gspread_Library import GoogleSheetHandler
from Libraries.Concatenation_Library import Concatenation_Handler

def Concatenate_Skills_By_RequestName(sheet_url, input_tab_name, output_tab_name, credentials_path):
    """
    Fetches data from a specified tab in a Google Sheet,
    concatenates 'Skills/Keywords' grouped by 'Request Name',
    and writes the transformed data back to another tab in the same sheet.

    :param sheet_url: URL of the Google Sheet
    :param input_tab_name: Name of the tab to fetch data from
    :param output_tab_name: Name of the tab to write transformed data to
    :param credentials_path: Path to the Google API service account credentials file
    """
    # Initialize the Google Sheet Handler
    sheet_handler = GoogleSheetHandler(credentials_path)

    print("Fetching data from the Google Sheet...")
    try:
        df_in = sheet_handler.get_sheet_as_dataframe(sheet_url, input_tab_name)
        print(f"Data successfully read from tab '{input_tab_name}':")
        print(df_in.iloc[:4, :4])
    except Exception as e:
        print(f"Failed to retrieve data: {e}")
        return

    # Work on a copy
    df = df_in.copy()

    # Define columns to front-fill
    # columns_to_fill = [
    #     "Department",
    #     "Job Title",
    #     "Job Description",
    #     "Request Name",
    #     "Gender Preference",
    #     "# of Volunteers",
    #     "Work Experience Needed?",
    #     "Number of Years",
    #     "Comments"
    # ]
    columns_to_fill = [
        "Request Name"
    ]

    # Define columns to concatenate
    columns_to_concatenate = [
        "Skills/Keywords",
        "Educational Qualification"
    ]

    # Apply front-fill function
    df = Concatenation_Handler.front_fill_columns(df, columns_to_fill)

    # Ensure appropriate columns are converted to strings before concatenation
    df = Concatenation_Handler.convert_columns_to_string(df, columns_to_concatenate)

    # Apply concatenation function
    df = Concatenation_Handler.concatenate_method_for_Gspread(df, 'Request Name', columns_to_concatenate)


    # Drop duplicates based on 'Request Name' to keep only one row per request
    df_result = df.drop_duplicates(subset=["Request Name"]).reset_index(drop=True)

    print("Concatenation complete. Writing to output tab...")

    # Define the desired column order
    desired_column_order = [
        "Request Name",
        "Job Title",
        "Job Description",
        "Gender Preference",
        "# of Volunteers",
        "Work Experience Needed?",
        "Number of Years",
        "Skills/Keywords",
        "Educational Qualification",
        "Comments",
        "Please mention any other comments about Seva timings",
        "Department"
    ]

    # Reorder the DataFrame
    df_result = df_result[desired_column_order]

    # Write the final DataFrame to the Google Sheet
    try:
        sheet_handler.write_dataframe_to_sheet(sheet_url, df_result, output_tab_name)
        print(f"Output successfully written to '{output_tab_name}' tab.")
    except Exception as e:
        print(f"Failed to write data: {e}")

if __name__ == "__main__":
    SHEET_URL = "https://docs.google.com/spreadsheets/d/1i0ANT5-tamlo6YX9uuMTUwayzL31YEmpgC6qQ-0y7jI/edit?gid=1288139751#gid=1288139751"
    INPUT_TAB = "Input"  # Change this if the input tab has a different name
    OUTPUT_TAB = "Skills Combined Output"
    CREDENTIALS_PATH = "D:/Seva-Allocation/Concatenation/credentials.json"  # Path to service account credentials file

    Concatenate_Skills_By_RequestName(SHEET_URL, INPUT_TAB, OUTPUT_TAB, CREDENTIALS_PATH)