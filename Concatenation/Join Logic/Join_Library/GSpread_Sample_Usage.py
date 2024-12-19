import pandas as pd
from Gspread_Library import GoogleSheetHandler

# Initialize the Google Sheet Handler
credentials_path = "D:/Seva-Allocation/old_pipeline/credentials.json"  # Path to your service account credentials file
sheet_handler = GoogleSheetHandler(credentials_path)

print("\n You can download credentials.json from https://console.cloud.google.com/")
print("Google cloud console -> API & Services -> Credentials -> Add Service Account - Add Key and download as Json", end="\n \n")
print("Note: Make sure to give editor access in the sheet for the client_email mentioned in credentials.json", end="\n \n")

# # Google Sheet URL
# sheet_url = "https://docs.google.com/spreadsheets/d/1GRuaVWZo0EWkbjrb5Tjs6MJYiYCXgWitr6jwjHhMdns/edit?gid=0#gid=0"

# Ask the user to input the Google Sheet URL
sheet_url = input("Please enter the Google Sheet URL: ")

# Fetch the data and convert it to a DataFrame
try:
    print('Fetching data from the Google Sheet...')
    df = sheet_handler.get_sheet_as_dataframe(sheet_url)
    print("DataFrame read from Google Sheet:")
    print(df)
except Exception as e:
    print(f"Failed to retrieve data: {e}")

# 2. Perform operations on the DataFrame (e.g., add a new column)
df["New Column"] = "Example Data"

# 3. Write updated DataFrame back to the Google Sheet
sheet_handler.write_dataframe_to_sheet(sheet_url, df)
print("Updated DataFrame written to the Google Sheet.")

# 4. Append additional data
additional_data = pd.DataFrame({
    "Column1": [1, 2],
    "Column2": [3, 4]
})
sheet_handler.append_to_sheet(sheet_url, additional_data)
print("Data appended to the Google Sheet.")