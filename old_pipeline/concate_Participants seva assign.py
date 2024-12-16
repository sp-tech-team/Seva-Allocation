import pandas as pd

def front_fill_columns(df, columns):
    """Front fills specified columns in the given DataFrame."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].ffill()
    return df

def concatenate_columns(df, group_column, columns_to_concatenate, separator=','):
    """Concatenates specified columns within a DataFrame, grouped by another column."""
    for column in columns_to_concatenate:
        if column in df.columns:
            df[column] = df.groupby(group_column)[column].transform(lambda x: separator.join(y for y in x if y != 'nan'))
    return df

def convert_columns_to_string(df, columns):
    """Converts specified columns in a DataFrame to string type."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df

def fill_and_drop_duplicates(df):
    """
    Loops through each row and column in the DataFrame. 
    - If a column in the current row is empty, it replaces the value with the corresponding value from the previous row.
    - Drops duplicate rows where all column values match.
    """
    # Fill empty values with the previous row's values
    for i in range(1, len(df)):  # Start from the second row
        for col in df.columns:
            if pd.isna(df.loc[i, col]) or df.loc[i, col] == '':
                df.loc[i, col] = df.loc[i-1, col]

    # Drop duplicate rows where all column values are identical
    df = df.drop_duplicates()

    return df

# Load the data
input_file = 'D:/Seva-Allocation/old_pipeline/content/seva_data_2024_updated.xlsx'

# Load the first and third tabs
df_main = pd.read_excel(input_file, sheet_name=0)  # First tab as df_main
# df_exported = pd.read_excel(input_file, sheet_name=2)  # Third tab as df_exported

# Columns to front-fill
columns_to_fill = [
    "SP ID",
    "Gender",
    "Age",
    "City",
    "State",
    "Nationality",
    "Country",
]

# Columns to concatenate
columns_to_concatenate = [
    "Languages",
    "Languages/Can read",
    "Languages/Can speak",
    "Languages/Can type",
    "Languages/Can write",
    "Education/Qualifications",
    "Education/Institution's Name",
    "Education/City",
    "Education/Specialization",
    "Education/Year of Passing/Graduation",
    "Work Experience/Company",
    "Work Experience/Designation",
    "Work Experience/Tasks",
    "Work Experience/Industry",
    "Work Experience/From Date",
    "Work Experience/To Date",
    "Marital status",
    "Program Tags",
    "Are you Coming with Laptop?",
    "Are you coming as couple?",
    "Any Additional Skills",
    "Computer Skills",
    "Skills",
    "Program History",
    "Local Volunteering",
    "Local Volunteering(days)",
    "Are you currently taking any medication? If yes, list medication & its purpose",
    "Have you taken any medication in the past? If yes, list medication and its purpose here",
    "Any highlights for SP Team",
    "Interviewer Feedback",
    "Interviewer Feedback/Summary/Question",
    "Interviewer Feedback/Summary/Summary",
    "Interviewer Feedback/Comments",
    "Concerns",
    "Please enter any concerns here",
    "Volunteering at IYC",
    "Volunteering at IYC/Volunteering Duration (No. of Days)",
    "Volunteering at IYC/Center Activity",
    "Volunteering at IYC/Description",
    "Local Volunteering/Volunteering Duration (No. of Days)",
    "Local Volunteering/Local center activity",
    "Any Hobbies/Interests",
    "Hobbies/Interests/Type",
    "Hobbies/Interests/Name",
    "Isha Connect/Name",
    "Please take some time to look carefully and reflect in detail as to why you wish to go through Sadhanapada at this particular time. In what way(s) are you hoping to grow through the program (Please elaborate in at least a few sentences)",
    "What are your thoughts on following the strict daily schedule, having very little personal time, no days off, and strictly adhering to the requirements and expectations of the program along with the guidelines of staying in the ashram?",
    "How do you feel about the physical demands of the program i.e) walking long distances, sitting cross legged and difficult activities like farming?",
    "How willing are you to be assigned to any kind of volunteering activity; which may be physically intense or office based, for the full duration of the program?",
    "How do you feel about sharing your space with many other volunteers? for example - dormitory stay area, shared bathroom facilities, and during volunteering activities ?",
    "How does your family feel about you staying at the Isha Yoga Center for the full duration of the program?",
    "What other questions do you have about the program?",
    "Now that you have more clarity on the program"
]

# Replace current df with df_exported
# df = df_exported
df = df_main.copy()  # Create an independent copy of df_main
df_filled = df_main.copy()  # Another independent copy for df_filled

# Fill empty cells for education and other field vlookups
df_filled = fill_and_drop_duplicates(df_filled)

# Apply front-fill function on df
df = front_fill_columns(df, columns_to_fill)

# Ensure appropriate columns are converted to strings before concatenation
df = convert_columns_to_string(df, columns_to_concatenate)

# Apply concatenation function
df = concatenate_columns(df, 'SP ID', columns_to_concatenate)

df_with_concatenate = df.copy()  # Create an independent copy after concatenation


# Retain only relevant columns (Skills/Keywords, Add'l Skills, Languages, Request Name)
columns_to_keep = columns_to_fill + columns_to_concatenate
df_exported_filtered = df[columns_to_keep].drop_duplicates('SP ID').reset_index(drop=True)

# Perform a left join of df_main with df_exported_filtered
# df_joined = df_main.merge(df_exported_filtered, on='SP ID', how='left')

# Write the output to a new Excel file with multiple tabs
output_file = 'D:/Seva-Allocation/old_pipeline/content/output_transformed.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_exported_filtered.to_excel(writer, sheet_name='Concatenated Export Data', index=False)  # Write df_with_concatenate to one tab
    df_filled.to_excel(writer, sheet_name='Filled Vlookup Data', index=False) # Write the filled data to another tab
    # df_joined.to_excel(writer, sheet_name='Joined Data', index=False)  # Write the joined data to another tab

print(f"Transformation complete. Output written to {output_file}")
