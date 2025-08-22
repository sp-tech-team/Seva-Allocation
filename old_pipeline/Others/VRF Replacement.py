import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

class GoogleSheetHandler:
    def __init__(self, credentials_file: str, scopes=None):
        """
        Initialize the handler with Google API credentials.
        :param credentials_file: Path to the credentials JSON file.
        :param scopes: Scopes required for the Google API.
        """
        if scopes is None:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        self.creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        self.client = gspread.authorize(self.creds)
    
    def get_sheet_as_dataframe(self, sheet_url: str, worksheet_name: str = None) -> pd.DataFrame:
        """
        Read a Google Sheet and convert it into a cleaned Pandas DataFrame.
        Replaces empty strings and 'nan' literals with np.nan, and converts column types.
        
        :param sheet_url: The URL of the Google Sheet.
        :param worksheet_name: The name of the worksheet/tab (optional, defaults to the first tab).
        :return: DataFrame containing the sheet's cleaned data.
        """
        sheet = self.client.open_by_url(sheet_url)
        if worksheet_name:
            worksheet = sheet.worksheet(worksheet_name)
        else:
            worksheet = sheet.get_worksheet(0)  # Default to the first worksheet
        
        # Fetch data and create DataFrame
        data = worksheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])  # Use the first row as headers
        
        # Replace empty strings and literal "nan" strings with np.nan
        df.replace("", np.nan, inplace=True)
        df.replace("nan", np.nan, inplace=True)
        
        # Ensure columns are properly typed
        df = df.convert_dtypes()
        
        return df

    def write_dataframe_to_sheet(self, sheet_url: str, df: pd.DataFrame, worksheet_name: str = None):
        """
        Write a Pandas DataFrame to a specific Google Sheet tab.
        If the tab does not exist, create it and add the content.
        :param sheet_url: The URL of the Google Sheet.
        :param df: The DataFrame to write.
        :param worksheet_name: The name of the worksheet/tab (optional, defaults to the first tab).
        """

        # Replace pd.NA or np.nan with empty strings to ensure JSON serializability
        df = df.fillna("").replace({pd.NA: ""})

        sheet = self.client.open_by_url(sheet_url)
        try:
            # Try to get the existing worksheet by name
            if worksheet_name:
                worksheet = sheet.worksheet(worksheet_name)
            else:
                worksheet = sheet.get_worksheet(0)  # Default to the first worksheet
        except:
            # If worksheet does not exist, create a new one
            if worksheet_name:
                worksheet = sheet.add_worksheet(title=worksheet_name, rows=str(len(df) + 1), cols=str(len(df.columns)))
            else:
                worksheet = sheet.get_worksheet(0)  # Fall back to first worksheet

        worksheet.clear()  # Clear existing data
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())  # Update with new data

    def append_to_sheet(self, sheet_url: str, df: pd.DataFrame, worksheet_name: str = None):
        """
        Append rows from a DataFrame to a specific Google Sheet tab without overwriting.
        If the tab does not exist, create it and add the content.
        :param sheet_url: The URL of the Google Sheet.
        :param df: The DataFrame to append.
        :param worksheet_name: The name of the worksheet/tab (optional, defaults to the first tab).
        """
        sheet = self.client.open_by_url(sheet_url)
        try:
            # Try to get the existing worksheet by name
            if worksheet_name:
                worksheet = sheet.worksheet(worksheet_name)
            else:
                worksheet = sheet.get_worksheet(0)  # Default to the first worksheet
        except:
            # If worksheet does not exist, create a new one
            if worksheet_name:
                worksheet = sheet.add_worksheet(title=worksheet_name, rows=str(len(df) + 1), cols=str(len(df.columns)))
            else:
                worksheet = sheet.get_worksheet(0)  # Fall back to first worksheet

        worksheet.append_rows(df.values.tolist(), value_input_option="RAW")

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

    # Convert column to string to ensure consistency
    if "Skills/Keywords" in df.columns:
        df["Skills/Keywords"] = df["Skills/Keywords"].astype(str)

        # Group by 'Request Name' and concatenate 'Skills/Keywords'
        df["Skills/Keywords"] = df.groupby("Request Name")["Skills/Keywords"] \
                                  .transform(lambda x: ', '.join(sorted(set(i.strip() for i in x if i.lower() != 'nan' and i.strip()))))

    else:
        print("'Skills/Keywords' column not found in the input data.")
        return

    # Drop duplicates based on 'Request Name' to keep only one row per request
    df_result = df.drop_duplicates(subset=["Request Name"]).reset_index(drop=True)

    print("Concatenation complete. Writing to output tab...")

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
    CREDENTIALS_PATH = "path/to/your/credentials.json"  # Update this to your local credentials file

    Concatenate_Skills_By_RequestName(SHEET_URL, INPUT_TAB, OUTPUT_TAB, CREDENTIALS_PATH)