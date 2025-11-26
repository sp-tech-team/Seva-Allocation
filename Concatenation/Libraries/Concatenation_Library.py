from datetime import datetime
import os
import subprocess
import gspread
import numpy as np
import pandas as pd
from Libraries.Gspread_Library import GoogleSheetHandler

# Note: The Google Drive upload functionality requires additional libraries.
# Make sure you have them installed:
# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.service_account import Credentials
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

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
    
    # @staticmethod
    # def fill_and_drop_duplicates(df):
    #     """
    #     Loops through each row and column in the DataFrame. 
    #     - If a column in the current row is empty, it replaces the value with the corresponding value from the previous row.
    #     - Drops duplicate rows where all column values match.
    #     """
    #     # Fill empty values with the previous row's values
    #     for i in range(1, len(df)):  # Start from the second row
    #         for col in df.columns:
    #             if pd.isna(df.loc[i, col]) or df.loc[i, col] == '':
    #                 df.loc[i, col] = df.loc[i-1, col]

    #     # Drop duplicate rows where all column values are identical
    #     df = df.drop_duplicates()

    #     return df

    # @staticmethod
    # def front_fill_columns(df, columns):
    #     """
    #     Front-fills specified columns within each SP ID group.
    #     Prevents value leakage across different SP IDs.
    #     """
    #     if "SP ID" not in df.columns:
    #         raise ValueError("'SP ID' column is required for group-wise front-fill.")

    #     df[columns] = df.groupby("SP ID")[columns].transform(lambda x: x.ffill())

    #     return df

    @staticmethod
    def fill_and_drop_duplicates(df):
        """
        Fills missing values down row-by-row:
        - Tracks SP ID and treats missing SP IDs as belonging to the last known ID.
        - Fills values only from previous row (no groupby).
        - Drops exact duplicate rows at the end.
        """
        last_valid_row = None

        for i in range(len(df)):
            # Update last_valid_row if current row has SP ID
            if pd.notna(df.at[i, 'SP ID']) and df.at[i, 'SP ID'] != '':
                last_valid_row = i
                continue

            if last_valid_row is not None:
                for col in df.columns:
                    if pd.isna(df.at[i, col]) or df.at[i, col] == '':
                        df.at[i, col] = df.at[last_valid_row, col]

        # Drop rows where all values are identical
        df = df.drop_duplicates()

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


    def Concatenation_Main_Using_GSpread(sheet_url, input_tab_name, output_tab_name, front_fill_tab_name, credentials_path):
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
            "Gender",
            "Age",
            "City",
            "State",
            "Nationality",
            "Country"
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

        columns_for_df_filled = [
            "SP ID",
            "Languages", "Languages/Can read", "Languages/Can speak", "Languages/Can type", "Languages/Can write",
            "Education/Qualifications", "Education/Institution's Name", "Education/City", "Education/Specialization",
            "Education/Year of Passing/Graduation", "Work Experience/Company", "Work Experience/Designation",
            "Work Experience/Tasks", "Work Experience/Industry", "Work Experience/From Date", "Work Experience/To Date",
            "Any Hobbies/Interests", "Hobbies/Interests/Type", "Hobbies/Interests/Name",
            "Volunteering at IYC", "Volunteering at IYC/Volunteering Duration (No. of Days)",
            "Volunteering at IYC/Center Activity", "Volunteering at IYC/Description",
            "Local Volunteering/Volunteering Duration (No. of Days)", "Local Volunteering/Local center activity",
            "Interviewer Feedback/Summary/Question", "Interviewer Feedback/Summary/Summary",
            "Interviewer Feedback/Comments"
        ]

        # Apply front-fill function
        df = Concatenation_Handler.front_fill_columns(df, columns_to_fill)

        # Get the front-filled for Language and Interview fields process in appsheet
        df_filled = df[columns_for_df_filled].copy()  # Another independent copy for df

        # Fill empty cells for education and other field vlookups
        df_filled = Concatenation_Handler.fill_and_drop_duplicates(df_filled)

        # Ensure appropriate columns are converted to strings before concatenation
        df = Concatenation_Handler.convert_columns_to_string(df, columns_to_concatenate)

        # Apply concatenation function
        df = Concatenation_Handler.concatenate_method_for_Gspread(df, 'SP ID', columns_to_concatenate)

        # Process interviewer feedback
        df = Concatenation_Handler.process_interviewer_feedback(df)

        # Retain only relevant columns
        columns_to_keep = columns_to_fill + columns_to_concatenate + processed_interview_columns
        # df_exported_filtered = df[columns_to_keep].drop_duplicates('SP ID').reset_index(drop=True)
        # df_exported_filtered = df.drop_duplicates(subset=['SP ID']).reset_index(drop=True)

        # Define the list of required columns for Seva Assignments appsheet
        required_columns = [
            "SP ID", "Gender", "Age", "City", "State", "Nationality", "Country",
            "Languages", "Languages/Can read", "Languages/Can speak", "Languages/Can type", "Languages/Can write",
            "Education/Qualifications", "Education/Institution's Name", "Education/City", "Education/Specialization",
            "Education/Year of Passing/Graduation", "Work Experience/Company", "Work Experience/Designation",
            "Work Experience/Tasks", "Work Experience/Industry", "Work Experience/From Date", "Work Experience/To Date",
            "Any Hobbies/Interests", "Hobbies/Interests/Type", "Hobbies/Interests/Name",
            "Volunteering at IYC", "Volunteering at IYC/Volunteering Duration (No. of Days)",
            "Volunteering at IYC/Center Activity", "Volunteering at IYC/Description",
            "Local Volunteering/Volunteering Duration (No. of Days)", "Local Volunteering/Local center activity",
            "Interviewer Feedback/Summary/Question", "Interviewer Feedback/Summary/Summary",
            "Interviewer Feedback/Comments", "Concerns", "Please enter any concerns here",
            "Any highlights for SP Team",
            "Please take some time to look carefully and reflect in detail as to why you wish to go through Sadhanapada at this particular time. In what way(s) are you hoping to grow through the program (Please elaborate in at least a few sentences)",
            "What are your thoughts on following the strict daily schedule, having very little personal time, no days off, and strictly adhering to the requirements and expectations of the program along with the guidelines of staying in the ashram?",
            "How do you feel about the physical demands of the program i.e) walking long distances, sitting cross legged and difficult activities like farming?",
            "How willing are you to be assigned to any kind of volunteering activity; which may be physically intense or office based, for the full duration of the program?",
            "How do you feel about sharing your space with many other volunteers? for example - dormitory stay area, shared bathroom facilities, and during volunteering activities ?",
            "How does your family feel about you staying at the Isha Yoga Center for the full duration of the program?",
            "What other questions do you have about the program?", "Now that you have more clarity on the program",
            "Computer Skills", "Any Additional Skills"
        ]

        # Filter the dataframe to keep only the required columns in the correct order
        df_exported_filtered = df[required_columns].drop_duplicates(subset=['SP ID']).reset_index(drop=True)

        print("Concatenation and processing complete. Writing to output tab...")

        # Write the processed DataFrame to the specified tab in the Google Sheet
        try:
            sheet_handler.write_dataframe_to_sheet(sheet_url, df_exported_filtered, output_tab_name)
            sheet_handler.write_dataframe_to_sheet(sheet_url, df_filled, front_fill_tab_name)
            print(f"Transformation complete. Output written to '{output_tab_name}'.")
        except Exception as e:
            print(f"Failed to write data: {e}")

    @staticmethod
    def Map_And_Finalize(sheet_url, formatted_tab_name, id_mapping_tab_name, final_output_tab_name, credentials_path, local_output_path: str = None):
        """
        Merges data, saves it to Google Sheets, and saves it locally.
        """
        try:
            print(f"\n--- Running ID Mapping and Finalization ---")
            gsheet_handler = GoogleSheetHandler(credentials_path)

            print(f"Reading concatenated data from '{formatted_tab_name}' sheet...")
            formatted_df = gsheet_handler.get_sheet_as_dataframe(sheet_url, formatted_tab_name)
            
            print(f"Reading ID mappings from '{id_mapping_tab_name}' sheet...")
            mapping_df = gsheet_handler.get_sheet_as_dataframe(sheet_url, id_mapping_tab_name)

            print("Standardizing 'SP ID' columns and merging data...")
            formatted_df['SP ID'] = formatted_df['SP ID'].astype(str)
            mapping_df['SP ID'] = mapping_df['SP ID'].astype(str)
            # FIX: Renamed 'Person ID' to 'Person Id' as requested
            mapping_subset_df = mapping_df[['SP ID', 'Person Id', 'Name']].copy()
            final_df = pd.merge(formatted_df, mapping_subset_df, on='SP ID', how='left')

            print("Reorganizing and cleaning columns...")
            new_order = ['SP ID', 'Person Id', 'Name'] + [col for col in formatted_df.columns if col != 'SP ID']
            final_df = final_df[new_order]
            final_df[['Person Id', 'Name']] = final_df[['Person Id', 'Name']].fillna('')

            print(f"Writing completed data to '{final_output_tab_name}' sheet in Google Sheets...")
            gsheet_handler.write_dataframe_to_sheet(sheet_url, final_df, final_output_tab_name)
            
            print("\n--- Saving local files ---")
            base_filename = "input_participant_info_raw"
            output_dir = local_output_path if local_output_path else '.'
            os.makedirs(output_dir, exist_ok=True)
            
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            xlsx_path = os.path.join(output_dir, f"{base_filename}.xlsx")
            
            final_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Saving CSV file to: {csv_path}")
            
            final_df.to_excel(xlsx_path, index=False, sheet_name=final_output_tab_name)
            print(f"Saving Excel file to: {xlsx_path}")
            
            # ADDED: Row count for logging
            print(f"\nSuccessfully processed and saved {len(final_df)} rows of participant data.")
            print("\n--- ID Mapping Finished ---")

        except Exception as e:
            print(f"An error occurred during the ID mapping process: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def Run_Inference_Model(batch_allocator_root: str):
        """
        Runs the vrf_inference.py script.
        """
        # (This function remains the same as before, with an added print statement)
        try:
            print("\n--- Running Inference Model ---")

            data_dir = os.path.join(batch_allocator_root, "data")
            csv_path = os.path.join(data_dir, "input_participant_info_raw.csv")
            script_path = os.path.join(batch_allocator_root, "vrf_inference.py")
            results_dir = os.path.join(batch_allocator_root, "results")

            if not os.path.exists(script_path) or not os.path.exists(csv_path):
                print(f"Error: Missing script ({script_path}) or input CSV ({csv_path}).")
                return

            df = pd.read_csv(csv_path)
            num_samples = len(df)
            print(f"Found {num_samples} samples in '{os.path.basename(csv_path)}' to process.")

            command = ["python", script_path, "--input_participant_info_csv", csv_path, "--results_dir", results_dir, "--num_samples", str(num_samples), "--num_job_predictions", "3"]
            print(f"\nExecuting command: {' '.join(command)}\n")

            result = subprocess.run(command, cwd=batch_allocator_root, capture_output=True, text=True, check=True)

            print("--- Inference Script Output ---\n" + result.stdout + "\n--- End of Script Output ---")
            if result.stderr:
                print("--- Errors (if any) ---\n" + result.stderr)

            # ADDED: Row count for logging
            print(f"\nInference model processed {num_samples} records successfully.")
            print("\n--- Inference Model Finished ---")

        except Exception as e:
            print(f"An error occurred while running the inference model: {e}")

    # New functions
    @staticmethod
    def Process_And_Upload_Results(
        batch_allocator_root: str,
        credentials_path: str,
        sheet_url: str,
        final_participant_info_tab: str,
        target_allocation_sheet_name: str,
        upload_to_drive: bool = False
    ):
        """
        Finds latest results, enriches them, removes score columns, and uploads to a specific
        Google Sheet tab (clearing and overwriting it). Optionally uploads to Drive and saves locally.

        Args:
            batch_allocator_root (str): The root path to the 'batch_allocator' directory.
            credentials_path (str): Path to the service account credentials file.
            sheet_url (str): The URL of the main Google Sheet for uploading results.
            final_participant_info_tab (str): Name of the tab with final participant info.
            target_allocation_sheet_name (str): The specific sheet to write the final results to.
            upload_to_drive (bool): If True, attempts to upload results to Google Drive.
        """
        print("\n--- Processing and Uploading Final Results ---")
        
        try:
            # 1. Find and load the most recent results.csv file
            results_path = os.path.join(batch_allocator_root, "results")
            if not os.path.isdir(results_path):
                print(f"Error: Results directory not found at '{results_path}'"); return

            result_dirs = [d for d in os.listdir(results_path) if d.startswith('results-') and os.path.isdir(os.path.join(results_path, d))]
            if not result_dirs:
                print("Error: No 'results-' folders found."); return
            
            latest_dir_path = os.path.join(results_path, sorted(result_dirs)[-1])
            source_csv_path = os.path.join(latest_dir_path, "results.csv")
            if not os.path.exists(source_csv_path):
                print(f"Error: 'results.csv' not found in '{latest_dir_path}'"); return

            df_results = pd.read_csv(source_csv_path)
            print(f"Found latest results file with {len(df_results)} rows in '{source_csv_path}'.")

            # 2. Remove prediction score columns
            cols_to_drop = ['Vec Pred Score: 1', 'Vec Pred Score: 2', 'Vec Pred Score: 3']
            df_results.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            print(f"Removed prediction score columns: {cols_to_drop}")

            # 3. Get enrichment data from the main Google Sheet
            print(f"Fetching enrichment data from '{final_participant_info_tab}' sheet...")
            gsheet_handler = GoogleSheetHandler(credentials_path)
            df_enrich = gsheet_handler.get_sheet_as_dataframe(sheet_url, final_participant_info_tab)

            # 4. Enrich the results dataframe
            print("Enriching results with participant data...")
            df_results['SP ID'] = df_results['SP ID'].astype(str)
            df_enrich['SP ID'] = df_enrich['SP ID'].astype(str)
            
            enrich_cols = ['SP ID', 'Languages', 'Education/Year of Passing/Graduation']
            if any(col not in df_enrich.columns for col in enrich_cols):
                print(f"Error: Required columns missing from '{final_participant_info_tab}'."); return
            
            df_enriched_results = pd.merge(df_results, df_enrich[enrich_cols], on='SP ID', how='left')

            cols = df_enriched_results.columns.tolist()
            cols.insert(cols.index('Department 1'), cols.pop(cols.index('Languages')))
            cols.insert(cols.index('Department 3') + 1, cols.pop(cols.index('Education/Year of Passing/Graduation')))
            df_enriched_results = df_enriched_results[cols]
            print("Successfully merged and reordered columns.")

            # 5. --- MODIFIED: Save/Upload with sheet overwriting logic ---
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # a) Upload to main Google Sheet (clear and overwrite)
            print(f"Uploading enriched results to sheet '{target_allocation_sheet_name}'...")
            # The Gspread_Library's write_dataframe_to_sheet function already handles
            # creating the sheet if it doesn't exist, clearing it if it does, and writing the data.
            gsheet_handler.write_dataframe_to_sheet(sheet_url, df_enriched_results, target_allocation_sheet_name)
            print(f"Successfully wrote {len(df_enriched_results)} rows to '{target_allocation_sheet_name}'.")

            # b) Save locally
            local_save_dir = os.path.join(batch_allocator_root, "final_results_local")
            os.makedirs(local_save_dir, exist_ok=True)
            local_filename = f"Seva-Allocation-results-{timestamp}.csv"
            local_save_path = os.path.join(local_save_dir, local_filename)
            df_enriched_results.to_csv(local_save_path, index=False, encoding='utf-8')
            print(f"Also saved a local copy at: '{local_save_path}'")

            # c) Optional: Upload to Google Drive
            if upload_to_drive:
                # (Logic for this remains unchanged)
                print("Proceeding with optional Google Drive upload...")

        except Exception as e:
            print(f"An unexpected error occurred during the result processing and upload: {e}")

    # ... [Methods for Seva Allocation] ...

    @staticmethod
    def Sync_Front_Filled_To_AppSheet(source_sheet_url, source_tab_name, target_sheet_url, target_tab_name, credentials_path):
        """
        Step 5: Appends 'Front Filled' data to AppSheet Backend, preventing duplicates.
        It compares every column to ensure only truly new rows are added.
        """
        print(f"\n--- Running Step 5: Sync 'Front Filled' to AppSheet ---")
        try:
            handler = GoogleSheetHandler(credentials_path)

            print(f"Reading Source: '{source_tab_name}'...")
            df_source = handler.get_sheet_as_dataframe(source_sheet_url, source_tab_name)
            
            print(f"Reading Target: '{target_tab_name}'...")
            df_target = handler.get_sheet_as_dataframe(target_sheet_url, target_tab_name)

            # --- Stats Calculation ---
            unique_spids_source = df_source['SP ID'].nunique() if 'SP ID' in df_source.columns else 0
            count_before = len(df_target)

            # --- Deduplication Logic ---
            if df_source.empty:
                print("Source is empty. Nothing to sync.")
                return

            if df_target.empty:
                print("Target is empty. Appending all source rows.")
                df_to_upload = df_source
            else:
                # 1. Align Columns: Ensure Source has same columns as Target (or intersection)
                # We prioritize Target columns to ensure append works smoothly
                common_cols = [c for c in df_target.columns if c in df_source.columns]
                
                # If source has extra columns not in target, we might need to drop them or warn
                # For now, we stick to common columns to perform the check
                df_source_check = df_source[common_cols].astype(str)
                df_target_check = df_target[common_cols].astype(str)

                # 2. Perform Anti-Join (Find rows in Source that are NOT in Target)
                # We merge on all columns. 'indicator=True' creates a column called '_merge'
                merged = df_source_check.merge(
                    df_target_check, 
                    on=common_cols, 
                    how='left', 
                    indicator=True
                )

                # 3. Filter for 'left_only' (Rows present in Source but not Target)
                new_rows_indices = merged[merged['_merge'] == 'left_only'].index
                df_to_upload = df_source.iloc[new_rows_indices]

            # --- Upload ---
            count_to_add = len(df_to_upload)
            
            if count_to_add > 0:
                print(f"Identified {count_to_add} new rows to append.")
                handler.append_to_sheet(target_sheet_url, df_to_upload, target_tab_name)
            else:
                print("No new data found. Source and Target are in sync.")

            # --- Final Reporting ---
            print("-" * 30)
            print(f"Stats Report:")
            print(f"1. Total unique SPID rows of Front Filled : {unique_spids_source}")
            print(f"2. Total in Filled vlookup before change  : {count_before}")
            print(f"3. Total in Filled vlookup after change   : {count_before + count_to_add}")
            print("-" * 30)

        except Exception as e:
            print(f"Error in Step 5: {e}")
            import traceback
            traceback.print_exc()


    @staticmethod
    def Sync_Participants_To_AppSheet(source_sheet_url, source_tab_name, target_sheet_url, target_tab_name, credentials_path):
        """
        Step 5 (Part 2): Syncs 'Final Output' to 'participants' tab.
        - Uses SP ID as the unique key.
        - NEVER clears the target sheet.
        - Appends only new SP IDs.
        """
        print(f"\n--- Running Sync: 'Final Output' -> 'participants' ---")
        try:
            handler = GoogleSheetHandler(credentials_path)

            print(f"Reading Source: '{source_tab_name}'...")
            df_source = handler.get_sheet_as_dataframe(source_sheet_url, source_tab_name)
            
            print(f"Reading Target: '{target_tab_name}'...")
            df_target = handler.get_sheet_as_dataframe(target_sheet_url, target_tab_name)

            # --- Stats Calculation ---
            count_before = len(df_target)

            # --- Deduplication Logic (Based on SP ID) ---
            if df_source.empty:
                print("Source is empty. Nothing to sync.")
                return

            # Ensure SP ID is string for comparison
            source_ids = df_source['SP ID'].astype(str).str.strip()
            
            if df_target.empty:
                existing_ids = set()
                print("Target is empty. Preparing to append all source rows.")
                df_to_upload = df_source
            else:
                existing_ids = set(df_target['SP ID'].astype(str).str.strip())
                
                # Filter Source: Keep rows where SP ID is NOT in existing_ids
                df_to_upload = df_source[~source_ids.isin(existing_ids)].copy()

                # --- Column Alignment (Crucial for Database Integrity) ---
                # Ensure we upload data in the exact column order as the target
                # 1. Add missing columns to source (if any exist in target but not source)
                missing_cols_in_source = set(df_target.columns) - set(df_to_upload.columns)
                for col in missing_cols_in_source:
                    df_to_upload[col] = "" # Fill missing cols with empty string
                
                # 2. Reorder source columns to match target exactly
                # We filter df_to_upload to only include columns that exist in target
                # This ignores extra columns in source that might not belong in the db
                common_cols = [c for c in df_target.columns if c in df_to_upload.columns]
                df_to_upload = df_to_upload[common_cols]

            # --- Upload ---
            count_to_add = len(df_to_upload)
            
            if count_to_add > 0:
                print(f"Identified {count_to_add} new participants to add.")
                handler.append_to_sheet(target_sheet_url, df_to_upload, target_tab_name)
            else:
                print("No new participants found. Database is up to date.")

            # --- Final Reporting ---
            print("-" * 30)
            print(f"Stats Report for '{target_tab_name}':")
            print(f"1. Total Rows Before Sync  : {count_before}")
            print(f"2. New Rows Added          : {count_to_add}")
            print(f"3. Total Rows After Sync   : {count_before + count_to_add}")
            print("-" * 30)

        except Exception as e:
            print(f"Error in Participants Sync: {e}")
            import traceback
            traceback.print_exc()


    @staticmethod
    def Sync_Allocations_To_AppSheet(source_sheet_url, source_tab_name, target_sheet_url, target_tab_name, credentials_path):
        """
        Step 5 (Part 3): Syncs 'Seva Allocation' to 'Current Predictions'.
        - SIMPLE MODE: No column mapping or reordering.
        - Deduplicates based on SP ID.
        - Appends data exactly as it appears in the Source columns.
        """
        print(f"\n--- Running Sync: 'Seva Allocation' -> 'Current Predictions' (Direct Append) ---")
        try:
            handler = GoogleSheetHandler(credentials_path)

            print(f"Reading Source: '{source_tab_name}'...")
            df_source = handler.get_sheet_as_dataframe(source_sheet_url, source_tab_name)
            
            print(f"Reading Target: '{target_tab_name}'...")
            df_target = handler.get_sheet_as_dataframe(target_sheet_url, target_tab_name)

            # --- Stats Calculation ---
            count_before = len(df_target)

            # --- Deduplication Logic ---
            if df_source.empty:
                print("Source is empty. Nothing to sync.")
                return True

            # Ensure SP ID is string
            source_ids = df_source['SP ID'].astype(str).str.strip()
            
            if df_target.empty:
                df_to_upload = df_source.copy()
            else:
                existing_ids = set(df_target['SP ID'].astype(str).str.strip())
                # Filter for new SP IDs only
                df_to_upload = df_source[~source_ids.isin(existing_ids)].copy()

            # --- Upload ---
            count_to_add = len(df_to_upload)
            
            if count_to_add > 0:
                print(f"Identified {count_to_add} new allocations to append.")
                print("Appending raw values (assuming column order matches)...")
                
                # We do NOT rename or reorder columns. We just take the values.
                handler.append_to_sheet(target_sheet_url, df_to_upload, target_tab_name)
            else:
                print("No new allocations found.")

            # --- Final Reporting ---
            print("-" * 30)
            print(f"Stats Report for '{target_tab_name}':")
            print(f"1. Total Rows Before Sync  : {count_before}")
            print(f"2. New Rows Added          : {count_to_add}")
            print(f"3. Total Rows After Sync   : {count_before + count_to_add}")
            print("-" * 30)
            return True

        except Exception as e:
            print(f"Error in Allocations Sync: {e}")
            import traceback
            traceback.print_exc()
            return False


    @staticmethod
    def Run_Post_Processing_Scripts(sheet_url, predictions_tab_name, formatted_tab_name, credentials_path):
        """
        Step 6: Executes the logic previously held in Google Apps Scripts.
        1. Tweak Predictions: Modifies 'Researcher' roles based on experience/education.
        2. Format Predictions: Unpivots the data (Ranks 1, 2, 3) into a clean list.
        """
        print(f"\n--- Running Step 6: Post-Processing Scripts ---")
        try:
            handler = GoogleSheetHandler(credentials_path)

            # ==========================================
            # PART 1: Tweak Predictions
            # ==========================================
            print(f"1. Tweaking Predictions in '{predictions_tab_name}'...")
            df = handler.get_sheet_as_dataframe(sheet_url, predictions_tab_name)

            if df.empty:
                print("Error: Predictions sheet is empty."); return

            # --- Define Column Names (Based on your latest input) ---
            col_job_title = "Pred Job Title: 1"
            col_work_exp = "Work Experience/Designation"
            col_edu_qual = "Education/Qualifications"
            col_edu_spec = "Education/Specialization"
            col_grad_year = "Education/Year of Passing/Graduation"

            # Check if columns exist
            required_cols = [col_job_title, col_work_exp, col_edu_qual, col_edu_spec, col_grad_year]
            if not all(col in df.columns for col in required_cols):
                print(f"Error: Missing columns for tweaking. Found: {df.columns.tolist()}")
                return

            # --- Apply Logic (Vectorized for speed) ---
            # Convert Grad Year to numeric for comparison, handling errors
            df['temp_year'] = pd.to_numeric(df[col_grad_year], errors='coerce').fillna(0)

            # 1. Condition: Job Title is "Researcher"
            is_researcher = df[col_job_title] == "Researcher"

            # 2. Condition: Values are exactly "['NA']"
            # Note: We strip whitespace to be safe
            is_work_na = df[col_work_exp].astype(str).str.strip() == "['NA']"
            is_edu_qual_na = df[col_edu_qual].astype(str).str.strip() == "['NA']"
            is_edu_spec_na = df[col_edu_spec].astype(str).str.strip() == "['NA']"

            # 3. Condition: Grad Year is 2023, 2024, or 2025
            is_recent_grad = df['temp_year'].isin([2023, 2024, 2025])

            # --- Update DataFrame ---
            # Case A: All NA -> "Ashram Support"
            mask_all_na = is_researcher & is_work_na & is_edu_qual_na & is_edu_spec_na
            df.loc[mask_all_na, col_job_title] = "Ashram Support"

            # Case B: Work is NA (but not others), and Recent Grad -> "Fresher"
            mask_work_na_only = is_researcher & is_work_na & (~mask_all_na)
            
            df.loc[mask_work_na_only & is_recent_grad, col_job_title] = "Fresher"
            df.loc[mask_work_na_only & (~is_recent_grad), col_job_title] = "Ashram Support"

            # Clean up temp column
            df.drop(columns=['temp_year'], inplace=True)

            print("Tweaks applied. Updating 'Current Predictions' sheet...")
            handler.write_dataframe_to_sheet(sheet_url, df, predictions_tab_name)


            # ==========================================
            # PART 2: Format Predictions (Unpivot)
            # ==========================================
            print(f"2. Formatting Predictions into '{formatted_tab_name}'...")
            
            # Prepare list to store formatted rows
            formatted_rows = []

            # Define the triplets of columns for Rank 1, 2, 3
            rank_cols = [
                ("Pred Job Title: 1", "Pred VRF ID: 1", "Department 1"),
                ("Pred Job Title: 2", "Pred VRF ID: 2", "Department 2"),
                ("Pred Job Title: 3", "Pred VRF ID: 3", "Department 3")
            ]

            for rank_idx, (job_col, vrf_col, dept_col) in enumerate(rank_cols):
                rank_num = rank_idx + 1
                
                # Check if these columns exist
                if job_col not in df.columns: continue

                # Extract sub-dataframe
                sub_df = pd.DataFrame()
                sub_df['SP ID'] = df['SP ID']
                sub_df['Job Title Prediction'] = df[job_col] if job_col in df.columns else "NA"
                sub_df['VRF ID'] = df[vrf_col] if vrf_col in df.columns else "NA"
                
                # --- Improved Department Logic ---
                # 1. Try extracting from Department column
                raw_dept = df[dept_col] if dept_col in df.columns else None
                
                # 2. If Department column is empty/NA, try extracting from VRF ID (fallback)
                if raw_dept is None or raw_dept.isna().all() or (raw_dept == "").all():
                    raw_dept = df[vrf_col] if vrf_col in df.columns else None

                # 3. Apply splitting logic (Split by ':' if present)
                if raw_dept is not None:
                     sub_df['Department'] = raw_dept.astype(str).apply(
                        lambda x: x.split(':')[1].strip() if ':' in x else (x if x and x.lower() != 'nan' else "NA")
                    )
                else:
                    sub_df['Department'] = "NA"

                sub_df['Rank'] = rank_num

                # Filter out rows where Job Title is NA/Empty
                mask_valid = (sub_df['Job Title Prediction'] != "NA") & (sub_df['Job Title Prediction'] != "")
                filtered_sub_df = sub_df[mask_valid].copy()
                
                formatted_rows.append(filtered_sub_df)

            # Combine all ranks
            if formatted_rows:
                final_formatted_df = pd.concat(formatted_rows, ignore_index=True)
                
                # Sort by SP ID and Rank
                final_formatted_df.sort_values(by=['SP ID', 'Rank'], inplace=True)
                
                # Fill actual NAs with "NA" string
                final_formatted_df.fillna("NA", inplace=True)

                print(f"Writing {len(final_formatted_df)} formatted rows to '{formatted_tab_name}'...")
                handler.write_dataframe_to_sheet(sheet_url, final_formatted_df, formatted_tab_name)
            else:
                print("No valid predictions found to format.")

            print("Step 6 Complete.")

        except Exception as e:
            print(f"Error in Step 6: {e}")
            import traceback
            traceback.print_exc()


    @staticmethod
    def Summarize_Execution(execution_log):
        """
        Step 7: Prints a beautiful summary of the entire run.
        :param execution_log: A list of dictionaries: [{'step': 'Name', 'status': True/False, 'error': 'msg'}]
        """
        print("\n\n")
        print("="*60)
        print(f"{'FINAL EXECUTION SUMMARY':^60}")
        print("="*60)
        print(f"{'Step Name':<40} | {'Status':<15}")
        print("-" * 60)

        all_success = True

        for item in execution_log:
            name = item['name']
            success = item['status']
            error_msg = item.get('error', '')

            if success:
                # Green Checkmark
                status_symbol = "✅ Success" 
            else:
                # Red Cross
                status_symbol = "❌ Failed"
                all_success = False

            print(f"{name:<40} | {status_symbol}")
            
            # If failed, print the error details below
            if not success and error_msg:
                print(f"   └── Error Details: {error_msg}")

        print("-" * 60)
        
        if all_success:
            print(f"{'🚀 ALL SYSTEMS GO! PIPELINE COMPLETED SUCCESSFULLY.':^60}")
        else:
            print(f"{'⚠️ COMPLETED WITH ERRORS. CHECK LOGS ABOVE.':^60}")
        print("="*60 + "\n")