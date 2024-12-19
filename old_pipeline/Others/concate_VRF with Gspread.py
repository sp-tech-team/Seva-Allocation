import numpy as np
import pandas as pd
from Gspread_Library import GoogleSheetHandler
from Concatenation_Library import Concatenation_Handler

def front_fill_columns(df, columns):
    """Front fills specified columns in the given DataFrame."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].ffill()
    return df

def concatenate_columns(df, group_column, columns_to_concatenate, separator=','):
    """Concatenates specified columns within a DataFrame, grouped by another column."""
    # Replace literal '<NA>' and 'NA' strings with np.nan
    df.replace(['<NA>', 'NA'], np.nan, inplace=True)
    
    for column in columns_to_concatenate:
        if column in df.columns:
            # Transform group and exclude NA values explicitly
            df[column] = df.groupby(group_column)[column].transform(
                lambda x: separator.join(x.dropna().astype(str))
            )
    return df

def convert_columns_to_string(df, columns):
    """Converts specified columns in a DataFrame to string type."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df

def process_interviewer_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Interviewer Feedback data to extract summaries and comments for Work Experience and Education.
    Adds four new columns to the DataFrame:
        - Interviewer Work Experience Summary
        - Interviewer Work Experience Feedback
        - Interviewer Education Summary
        - Interviewer Education Feedback
    
    :param df: Input DataFrame containing SP ID and feedback columns.
    :return: Updated DataFrame with additional columns.
    """
    # Define the relevant columns
    question_col = "Interviewer Feedback/Summary/Question"
    summary_col = "Interviewer Feedback/Summary/Summary"
    comments_col = "Interviewer Feedback/Comments"

    # Filter the rows where 'Question' contains Work Experience or Education (Red Flags)
    filtered_df = df[df[question_col].isin(["Work Experience", "Education (Red Flags)"])]

    # Initialize empty dictionaries to store values for each SP ID
    work_experience_summary = {}
    work_experience_feedback = {}
    education_summary = {}
    education_feedback = {}

    # Iterate through the filtered rows to populate dictionaries
    for _, row in filtered_df.iterrows():
        sp_id = row["SP ID"]
        question = row[question_col]
        summary = row[summary_col]
        feedback = row[comments_col]

        if question == "Work Experience":
            work_experience_summary[sp_id] = summary
            work_experience_feedback[sp_id] = feedback
        elif question == "Education (Red Flags)":
            education_summary[sp_id] = summary
            education_feedback[sp_id] = feedback

    # Add new columns to the original DataFrame
    df["Interviewer Work Experience Summary"] = df["SP ID"].map(work_experience_summary)
    df["Interviewer Work Experience Feedback"] = df["SP ID"].map(work_experience_feedback)
    df["Interviewer Education Summary"] = df["SP ID"].map(education_summary)
    df["Interviewer Education Feedback"] = df["SP ID"].map(education_feedback)

    return df


# Initialize the Google Sheet Handler
credentials_path = "D:/Seva-Allocation/old_pipeline/credentials.json"  # Path to your service account credentials file
sheet_handler = GoogleSheetHandler(credentials_path)

print("\n You can download credentials.json from https://console.cloud.google.com/")
print("Google cloud console -> API & Services -> Credentials -> Add Service Account - Add Key and download as Json", end="\n \n")
print("Note: Make sure to give editor access in the sheet for the client_email mentioned in credentials.json", end="\n \n")

# # Google Sheet URL - Hardcoded Input
sheet_url = "https://docs.google.com/spreadsheets/d/1DtJEE7e43ePkrGefvn7nl21je2xDGZWxOCKOS8nAxYE/edit?gid=903292404#gid=903292404"
input_tab_name = 'input'

output_tab_name = ''

# Ask the user to input the Google Sheet URL
# sheet_url = input("Please enter the Google Sheet URL: ")
# input_tab_name = input("Please enter the input Tab Name: ")
# output_tab_name = input("Please enter the output Tab Name: ")

if output_tab_name.strip()=='':
    output_tab_name = 'Formatted Input'

# Fetch the data and convert it to a DataFrame
try:
    print('Fetching data from the Google Sheet...')
    df = sheet_handler.get_sheet_as_dataframe(sheet_url)
    print("DataFrame read from Google Sheet:")
    print(df.iloc[:4, :4])
except Exception as e:
    print(f"Failed to retrieve data: {e}")

df_in = sheet_handler.get_sheet_as_dataframe(sheet_url, input_tab_name)

# Load the data
# input_file = 'D:/Seva-Allocation/old_pipeline/content/VRF.xlsx'

# Load the first and third tabs
# df_main = pd.read_excel(input_file, sheet_name=0)  # First tab as df_main
# df_exported = pd.read_excel(input_file, sheet_name=2)  # Third tab as df_exported

# Replace current df with df_exported
# df = df_exported
df = df_in.copy()

# Columns to front-fill
columns_to_fill = [
    "SP ID",
    "Registration Batch",
    "Gender",
    "Age"
]

#definition
processed_interview_columns = []

# Columns to concatenate
columns_to_concatenate = [
    "Work Experience/Company",
    "Work Experience/Designation",
    "Work Experience/Tasks",
    "Work Experience/Industry",
    "Work Experience/From Date",
    "Work Experience/To Date",
    "Education/Qualifications",
    "Education/Institution's Name",
    "Education/City",
    "Education/Specialization",
    "Any Additional Skills",
    "Computer Skills",
    "Skills",
    "Languages",
    "Any Hobbies/Interests",
    "Hobbies/Interests/Type",
    "Hobbies/Interests/Name"
]

# Apply front-fill function
df = front_fill_columns(df, columns_to_fill)

# Ensure appropriate columns are converted to strings before concatenation
df = convert_columns_to_string(df, columns_to_concatenate)

# Apply concatenation function
df = concatenate_columns(df, 'SP ID', columns_to_concatenate)

df_with_concatenate = df.copy()

# print('Before Filter:\n', df_with_concatenate.iloc[:4, [0, 17]])

# process interview columns
processed_interview_columns = [
    "Interviewer Work Experience Summary", "Interviewer Work Experience Feedback", "Interviewer Education Summary",
    "Interviewer Education Feedback"
]

df_processed = process_interviewer_feedback(df)

# Display the updated DataFrame
print(df_processed.head(), '\n Processing interview columns...')

# Retain only relevant columns (Skills/Keywords, Add'l Skills, Languages, Request Name)
# columns_to_keep = ['Request Name', 'Skills/Keywords', "Add'l Skills", 'Languages']
columns_to_keep = columns_to_fill + columns_to_concatenate + processed_interview_columns
df_exported_filtered = df[columns_to_keep].drop_duplicates('SP ID').reset_index(drop=True)

print('Concatenation done:\n', df_exported_filtered.iloc[:4, [0, 17]], '\nWriting to sheet...')

df_out =  sheet_handler.write_dataframe_to_sheet(sheet_url, df_exported_filtered, output_tab_name)

print(f"Transformation complete. Output written to {output_tab_name}")


# Perform a left join of df_main with df_exported_filtered
# df_joined = df_main.merge(df_exported_filtered, on='Request Name', how='left')

# # Write the output to a new Excel file with multiple tabs
# output_file = 'D:/Seva-Allocation/old_pipeline/content/output_transformed.xlsx'
# with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
#     df_with_concatenate.to_excel(writer, sheet_name='Concatenated Export Data', index=False)  # Write df_with_concatenate to one tab
#     df_joined.to_excel(writer, sheet_name='Joined Data', index=False)  # Write the joined data to another tab

# print(f"Transformation complete. Output written to {output_file}")
