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

# Load the data
df = pd.read_excel('/content/seva_data_2024_updated.xlsx')

# Columns to front-fill
columns_to_fill = [
    'SP ID', 'Registration Batch', 'Gender', 'Age', 'Seva Dept', 'City', 'State', 
    'Nationality', 'Country'
]

# Columns to concatenate
columns_to_concatenate = [
    'Work Experience/Company', 'Work Experience/Designation', 'Work Experience/Tasks',
    'Work Experience/Industry', 'Work Experience/From Date', 'Work Experience/To Date',
    'Education/Qualifications', 'Education/Institution\'s Name', 'Education/City', 
    'Education/Specialization', 'Education/Year of Passing/Graduation', 'Marital status', 
    'Program Tags', 'Are you Coming with Laptop?', 'Are you coming as couple?', 'Add\'l Skills', 
    'Add\'l Skills.1', 'Any Additional Skills', 'Computer Skills', 'Skills', 'Skills.1', 
    'Program History', 'Program History.1', 'Local Volunteering', 'Local Volunteering(days)',
    'Are you currently taking any medication? If yes, list medication & its purpose', 
    'Are you currently taking any medication? If yes, list medication & its purpose.1', 
    'Have you taken any medication in the past? If yes, list medication and its purpose here', 
    'Have you taken any medication in the past? If yes, list medication and its purpose here.1', 
    'Highlights to SP Team/Value', 'Interviewer Feedback', 'Interviewer Feedback/Answer', 
    'Input from the interviewer', 'Concerns', 'Please enter any concerns here', 'Languages', 
    'Languages/Can read', 'Languages/Can speak', 'Languages/Can type', 'Languages/Can write', 
    'Volunteering at IYC', 'Volunteering at IYC/Volunteering Duration (No. of Days)', 
    'Volunteering at IYC/Center Activity', 'Volunteering at IYC/Description', 
    'Local Volunteering/Volunteering Duration (No. of Days)', 'Local Volunteering/Local center activity',
    'Any Hobbies/Interests', 'Hobbies/Interests/Type', 'Hobbies/Interests/Name', 'Isha Connect/Name'
]

# Apply front-fill function
df = front_fill_columns(df, columns_to_fill)

# Ensure appropriate columns are converted to strings before concatenation
df = convert_columns_to_string(df, columns_to_concatenate)

# Apply concatenation function
df = concatenate_columns(df, 'SP ID', columns_to_concatenate)

# Drop duplicate rows after the transformation
df = df.drop_duplicates('SP ID').reset_index(drop=True)

# Write the transformed data to a new Excel file
df.to_excel('/content/output_transformed.xlsx', index=False)