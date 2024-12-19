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


