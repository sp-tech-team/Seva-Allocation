import pandas as pd

def front_fill_columns(df, columns):
    """Front fills specified columns in the given DataFrame."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].ffill()
    return df

def concatenate_columns(df, group_column, columns_to_concatenate, separator=' '):
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

# Load the data
input_file = 'D:/Seva-Allocation/old_pipeline/content/VRF.xlsx'

# Load the first and third tabs
df_main = pd.read_excel(input_file, sheet_name=0)  # First tab as df_main
df_exported = pd.read_excel(input_file, sheet_name=2)  # Third tab as df_exported

# Replace current df with df_exported
df = df_exported

# Columns to front-fill
columns_to_fill = [
    'Request Name', 'Job Title', 'Job Description', 'Department', 
    'Gender Preference', '# of Volunteers', 'Stage'
]

# Columns to concatenate
columns_to_concatenate = [
    'Skills/Keywords', "Add'l Skills", 'Languages', 
    'Work Experience Needed?', 'Number of Years', 'Educational Qualification'
]

# Apply front-fill function
df = front_fill_columns(df, columns_to_fill)

# Ensure appropriate columns are converted to strings before concatenation
df = convert_columns_to_string(df, columns_to_concatenate)

# Apply concatenation function
df = concatenate_columns(df, 'Request Name', columns_to_concatenate)

df_with_concatenate = df

# Retain only relevant columns (Skills/Keywords, Add'l Skills, Languages, Request Name)
columns_to_keep = ['Request Name', 'Skills/Keywords', "Add'l Skills", 'Languages']
df_exported_filtered = df[columns_to_keep].drop_duplicates('Request Name').reset_index(drop=True)

# Perform a left join of df_main with df_exported_filtered
df_joined = df_main.merge(df_exported_filtered, on='Request Name', how='left')

# Write the output to a new Excel file with multiple tabs
output_file = 'D:/Seva-Allocation/old_pipeline/content/output_transformed.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_with_concatenate.to_excel(writer, sheet_name='Exported Data', index=False)  # Write df_with_concatenate to one tab
    df_joined.to_excel(writer, sheet_name='Joined Data', index=False)  # Write the joined data to another tab

print(f"Transformation complete. Output written to {output_file}")
