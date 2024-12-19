import numpy as np
import pandas as pd
from Gspread_Library import GoogleSheetHandler


class Concatenation_Handler:
    """
    A handler class for various DataFrame concatenation and transformation operations.
    """

    @staticmethod
    def front_fill_columns(df, columns):
        """Front fills specified columns in the given DataFrame."""
        for column in columns:
            if column in df.columns:
                df[column] = df[column].ffill()
        return df

    @staticmethod
    def concatenate_method_for_Gspread(df, group_column, columns_to_concatenate, separator=','):
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
    
    @staticmethod
    def concatenate_method_for_Local_Downloaded_File(df, group_column, columns_to_concatenate, separator=','):
        """Concatenates specified columns within a DataFrame, grouped by another column."""
        for column in columns_to_concatenate:
            if column in df.columns:
                df[column] = df.groupby(group_column)[column].transform(lambda x: separator.join(y for y in x if y != 'nan'))
        return df

    @staticmethod
    def convert_columns_to_string(df, columns):
        """Converts specified columns in a DataFrame to string type."""
        for column in columns:
            if column in df.columns:
                df[column] = df[column].astype(str)
        return df

    @staticmethod
    def process_interviewer_feedback(df):
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

    def Concatenation_Main_Using_Local_Downloaded_File(input_file, output_file):
        """
        Processes a locally downloaded Excel file by:
        - Front-filling specified columns
        - Concatenating work experience, education, skills, and hobby-related columns
        - Exporting transformed data to a new Excel file with multiple tabs

        :param input_file: Path to the input Excel file
        :param output_file: Path to the output Excel file
        """

        # Load the first tab from the input file
        df_main = pd.read_excel(input_file, sheet_name=0)  # First tab as df_main
        df = df_main.copy()

        # Define columns to front-fill
        columns_to_fill = [
            "SP ID",
            "Gender",
            "Age"
        ]

        # Define columns to concatenate
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
        df = Concatenation_Handler.front_fill_columns(df, columns_to_fill)

        # Ensure appropriate columns are converted to strings before concatenation
        df = Concatenation_Handler.convert_columns_to_string(df, columns_to_concatenate)

        # Apply concatenation function
        df = Concatenation_Handler.concatenate_method_for_Local_Downloaded_File(df, 'SP ID', columns_to_concatenate)

        # Retain only relevant columns
        columns_to_keep = columns_to_fill + columns_to_concatenate
        df_exported_filtered = df[columns_to_keep].drop_duplicates('SP ID').reset_index(drop=True)

        # Write the output to a new Excel file with multiple tabs
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_exported_filtered.to_excel(writer, sheet_name='Concatenated Export Data', index=False)

        print(f"Transformation complete. Output written to {output_file}")


    def Concatenation_Main_Using_GSpread(sheet_url, input_tab_name, output_tab_name, credentials_path):
        """
        Processes a Google Sheet by:
        - Fetching data from a specified tab
        - Front-filling specified columns
        - Concatenating work experience, education, skills, and hobby-related columns
        - Processing interviewer feedback columns
        - Writing transformed data to another specified tab in the Google Sheet

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

        # Copy the fetched DataFrame
        df = df_in.copy()

        # Define columns to front-fill
        columns_to_fill = [
            "SP ID",
            "Registration Batch",
            "Gender",
            "Age"
        ]

        # Define columns to concatenate
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

        # Process interviewer feedback columns
        processed_interview_columns = [
            "Interviewer Work Experience Summary",
            "Interviewer Work Experience Feedback",
            "Interviewer Education Summary",
            "Interviewer Education Feedback"
        ]

        # Apply front-fill function
        df = Concatenation_Handler.front_fill_columns(df, columns_to_fill)

        # Ensure appropriate columns are converted to strings before concatenation
        df = Concatenation_Handler.convert_columns_to_string(df, columns_to_concatenate)

        # Apply concatenation function
        df = Concatenation_Handler.concatenate_method_for_Gspread(df, 'SP ID', columns_to_concatenate)

        # Process interviewer feedback
        df = Concatenation_Handler.process_interviewer_feedback(df)

        # Retain only relevant columns
        columns_to_keep = columns_to_fill + columns_to_concatenate + processed_interview_columns
        df_exported_filtered = df[columns_to_keep].drop_duplicates('SP ID').reset_index(drop=True)

        print("Concatenation and processing complete. Writing to output tab...")

        # Write the processed DataFrame to the specified tab in the Google Sheet
        try:
            sheet_handler.write_dataframe_to_sheet(sheet_url, df_exported_filtered, output_tab_name)
            print(f"Transformation complete. Output written to '{output_tab_name}'.")
        except Exception as e:
            print(f"Failed to write data: {e}")

