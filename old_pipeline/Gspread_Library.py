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
        Read a Google Sheet and convert it into a Pandas DataFrame.
        :param sheet_url: The URL of the Google Sheet.
        :param worksheet_name: The name of the worksheet (optional, defaults to the first worksheet).
        :return: DataFrame containing the sheet's data.
        """
        sheet = self.client.open_by_url(sheet_url)
        if worksheet_name:
            worksheet = sheet.worksheet(worksheet_name)
        else:
            worksheet = sheet.get_worksheet(0)  # Default to the first worksheet
        data = worksheet.get_all_values()
        return pd.DataFrame(data[1:], columns=data[0])  # Use the first row as headers

    def write_dataframe_to_sheet(self, sheet_url: str, df: pd.DataFrame, worksheet_name: str = None):
        """
        Write a Pandas DataFrame to a Google Sheet.
        :param sheet_url: The URL of the Google Sheet.
        :param df: The DataFrame to write.
        :param worksheet_name: The name of the worksheet (optional, defaults to the first worksheet).
        """
        sheet = self.client.open_by_url(sheet_url)
        if worksheet_name:
            worksheet = sheet.worksheet(worksheet_name)
        else:
            worksheet = sheet.get_worksheet(0)
        worksheet.clear()  # Clear existing data
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())  # Update with new data

    def append_to_sheet(self, sheet_url: str, df: pd.DataFrame, worksheet_name: str = None):
        """
        Append rows from a DataFrame to the Google Sheet without overwriting.
        :param sheet_url: The URL of the Google Sheet.
        :param df: The DataFrame to append.
        :param worksheet_name: The name of the worksheet (optional, defaults to the first worksheet).
        """
        sheet = self.client.open_by_url(sheet_url)
        if worksheet_name:
            worksheet = sheet.worksheet(worksheet_name)
        else:
            worksheet = sheet.get_worksheet(0)
        worksheet.append_rows(df.values.tolist(), value_input_option="RAW")

