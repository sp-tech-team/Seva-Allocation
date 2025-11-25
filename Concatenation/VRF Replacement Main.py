import numpy as np
import pandas as pd
import gspread
import traceback
import os
import sys
import subprocess
from datetime import datetime
from google.oauth2.service_account import Credentials
from Libraries.Gspread_Library import GoogleSheetHandler
from Libraries.Concatenation_Library import Concatenation_Handler

# --- Global Logging List ---
execution_log = []

def run_step(step_name, func, *args, **kwargs):
    """ Helper to run a function, track its status, and log the result. """
    print(f"\n--- Starting {step_name} ---")
    start = datetime.now()
    
    try:
        result = func(*args, **kwargs)
        if result is False:
            status = False
            error_msg = "Function returned 'False' (Check internal logs)"
        else:
            status = True
            error_msg = None
            
    except Exception as e:
        status = False
        error_msg = str(e)
        traceback.print_exc()

    duration = datetime.now() - start
    print(f"--- Finished {step_name} (Time: {duration.total_seconds():.2f}s) ---\n")
    
    execution_log.append({
        "name": step_name,
        "status": status,
        "error": error_msg
    })

# ==========================================
# FUNCTION DEFINITIONS
# ==========================================

def Concatenate_Skills_By_RequestName(sheet_url, input_tab_name, output_tab_name, credentials_path, local_cleaned_paths=[]):
    """ 
    Step 1: Fetches, concatenates skills, updates GSheet, AND saves 'vrf_data_cleaned.csv' to multiple locations.
    """
    try:
        sheet_handler = GoogleSheetHandler(credentials_path)
        print(f"Fetching data from '{input_tab_name}'...")
        df_in = sheet_handler.get_sheet_as_dataframe(sheet_url, input_tab_name)
        
        if df_in.empty:
            print("Input data is empty.")
            return False

        df = df_in.copy()
        columns_to_fill = ["Request Name"]
        columns_to_concatenate = ["Skills/Keywords", "Educational Qualification"]

        df = Concatenation_Handler.front_fill_columns(df, columns_to_fill)
        df = Concatenation_Handler.convert_columns_to_string(df, columns_to_concatenate)
        df = Concatenation_Handler.concatenate_method_for_Gspread(df, 'Request Name', columns_to_concatenate)
        df_result = df.drop_duplicates(subset=["Request Name"]).reset_index(drop=True)

        desired_column_order = [
            "Request Name", "Job Title", "Job Description", "Gender Preference",
            "# of Volunteers", "Work Experience Needed?", "Number of Years",
            "Skills/Keywords", "Educational Qualification", "Comments",
            "Please mention any other comments about Seva timings", "Department"
        ]
        
        existing_cols = [col for col in desired_column_order if col in df_result.columns]
        df_result = df_result[existing_cols]

        print(f"Concatenation complete. Writing {len(df_result)} rows to '{output_tab_name}'...")
        sheet_handler.write_dataframe_to_sheet(sheet_url, df_result, output_tab_name)

        # --- NEW: Save Cleaned Data to Multiple Locations ---
        if local_cleaned_paths:
            for path in local_cleaned_paths:
                output_dir = os.path.dirname(path)
                if not os.path.exists(output_dir): os.makedirs(output_dir)
                
                print(f"Saving Cleaned Data to: {path}")
                df_result.to_csv(path, index=False, encoding='utf-8')

        return True

    except Exception as e:
        print(f"Error in Step 1: {e}")
        traceback.print_exc()
        return False


def Sync_VRF_To_AppSheet(source_sheet_url, source_tab_name, target_sheet_url, target_tab_name, credentials_path):
    """ Step 2: Syncs 'Skills Combined Output' to 'New vrf'. """
    try:
        handler = GoogleSheetHandler(credentials_path)
        print(f"Reading Source: '{source_tab_name}'...")
        df_source = handler.get_sheet_as_dataframe(source_sheet_url, source_tab_name)
        print(f"Reading Target: '{target_tab_name}'...")
        df_target = handler.get_sheet_as_dataframe(target_sheet_url, target_tab_name)

        count_before = len(df_target)
        if df_source.empty: return True

        source_ids = df_source.iloc[:, 0].astype(str).str.strip()
        
        if df_target.empty:
            print("Target is empty. Appending all rows.")
            df_to_upload = df_source.copy()
        else:
            target_ids = set(df_target.iloc[:, 0].astype(str).str.strip())
            df_to_upload = df_source[~source_ids.isin(target_ids)].copy()

        count_to_add = len(df_to_upload)
        if count_to_add > 0:
            print(f"Identified {count_to_add} new VRF requests to append.")
            handler.append_to_sheet(target_sheet_url, df_to_upload, target_tab_name)
        else:
            print("No new VRF requests found.")

        print("-" * 30)
        print(f"Stats Report for '{target_tab_name}':")
        print(f"1. Total Rows Before Sync  : {count_before}")
        print(f"2. New Rows Added          : {count_to_add}")
        print(f"3. Total Rows After Sync   : {count_before + count_to_add}")
        print("-" * 30)
        return True
    except Exception as e:
        print(f"Error in Step 2: {e}")
        traceback.print_exc()
        return False

def Merge_Old_And_New(sheet_url, old_tab, new_tab, merged_tab, credentials_path):
    """ Step 3: Merges 'Old vrf' and 'New vrf' into 'Merged VRF's'. """
    try:
        handler = GoogleSheetHandler(credentials_path)
        print(f"Reading '{old_tab}' and '{new_tab}'...")
        
        df_old = handler.get_sheet_as_dataframe(sheet_url, old_tab)
        df_new = handler.get_sheet_as_dataframe(sheet_url, new_tab)

        df_old['VRF ID Status'] = "Old VRF ID"
        df_new['VRF ID Status'] = "New VRF ID"

        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        id_col = df_combined.columns[0]
        
        print("Merging and handling conflicts...")
        df_merged = df_combined.drop_duplicates(subset=[id_col], keep='last').reset_index(drop=True)
        df_merged = df_merged[df_merged[id_col].astype(str).str.strip() != ""]

        print(f"Writing {len(df_merged)} rows to '{merged_tab}'...")
        handler.write_dataframe_to_sheet(sheet_url, df_merged, merged_tab)
        return True

    except Exception as e:
        print(f"Error in Step 3: {e}")
        traceback.print_exc()
        return False

def Update_Main_VRF_Table(sheet_url, merged_tab, main_vrf_tab, credentials_path):
    """ Step 4: Safe Replace Logic (Merged + Orphans). """
    try:
        handler = GoogleSheetHandler(credentials_path)
        print(f"Reading '{merged_tab}' and '{main_vrf_tab}'...")
        
        df_merged = handler.get_sheet_as_dataframe(sheet_url, merged_tab) 
        df_main = handler.get_sheet_as_dataframe(sheet_url, main_vrf_tab)

        if df_merged.empty:
            print("Merged source is empty. No updates applied.")
            return True

        id_col = df_merged.columns[0]
        print(f"Using Unique Key: '{id_col}'")

        df_merged[id_col] = df_merged[id_col].astype(str).str.strip()
        df_main[id_col] = df_main[id_col].astype(str).str.strip()
        
        merged_ids = set(df_merged[id_col])
        df_orphans = df_main[~df_main[id_col].isin(merged_ids)].copy()
        
        print(f"Updates/New: {len(df_merged)} | Preserved Orphans: {len(df_orphans)}")

        df_final = pd.concat([df_merged, df_orphans], ignore_index=True)
        
        print(f"Final Count: {len(df_final)}. Writing safely to '{main_vrf_tab}' starting at Row 2...")

        sheet = handler.client.open_by_url(sheet_url)
        ws = sheet.worksheet(main_vrf_tab)

        df_final = df_final.fillna("").replace({pd.NA: ""})
        data_values = df_final.values.tolist()

        if data_values:
            ws.update("A2", data_values, value_input_option="RAW")
        
        last_updated_row = 1 + len(data_values) 
        total_rows_in_sheet = ws.row_count
        
        if total_rows_in_sheet > last_updated_row:
            print(f"Clearing residual data from row {last_updated_row + 1} to {total_rows_in_sheet}...")
            ws.batch_clear([f"A{last_updated_row + 1}:{total_rows_in_sheet}"])

        return True

    except Exception as e:
        print(f"Error in Step 4: {e}")
        traceback.print_exc()
        return False

def Run_VRF_Indexer(sheet_url, tab_name, local_raw_paths, batch_allocator_root, credentials_path):
    """
    Step 5: Updates Pinecone Vector DB.
    1. Fetches raw data from the INPUT SHEET ('Input').
    2. Saves it as 'vrf_data_raw.csv' to MULTIPLE LOCATIONS.
    3. Runs 'batch_allocator.vrf_indexer'.
    """
    try:
        # 1. Fetch Data
        handler = GoogleSheetHandler(credentials_path)
        print(f"Fetching RAW data for indexing from '{tab_name}'...")
        
        df = handler.get_sheet_as_dataframe(sheet_url, tab_name)
        
        if df.empty:
            print("Error: Input data for indexing is empty.")
            return False

        # 2. Save Raw CSV to ALL paths
        if local_raw_paths:
            for path in local_raw_paths:
                output_dir = os.path.dirname(path)
                if not os.path.exists(output_dir): os.makedirs(output_dir)
                
                print(f"Saving RAW data to: {path}")
                df.to_csv(path, index=False, encoding='utf-8')

        # 3. Configure Command
        execution_cwd = os.path.dirname(batch_allocator_root)
        
        print(f"Executing Module: batch_allocator.vrf_indexer")
        print(f"Working Directory: {execution_cwd}")
        
        command = [
            sys.executable, 
            "-m", 
            "batch_allocator.vrf_indexer", 
            "--pinecone_index_name", 
            "vrf-vectors"
        ]
        
        result = subprocess.run(
            command, 
            cwd=execution_cwd,  
            capture_output=True, 
            text=True, 
            check=True
        )

        print("--- Indexer Output ---")
        print(result.stdout)
        print("----------------------")
        
        if result.stderr:
            print("--- Indexer Warnings/Errors ---")
            print(result.stderr)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running indexer script: {e}")
        print("Script Output:", e.stdout)
        print("Script Error:", e.stderr)
        return False
    except Exception as e:
        print(f"Error in Step 5: {e}")
        traceback.print_exc()
        return False

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    start_time = datetime.now()
    CREDENTIALS_PATH = "D:/Seva-Allocation/Concatenation/credentials.json"
    
    # --- PATHS (SYNCING TWO FOLDERS) ---
    # 1. The module folder (required for the script to work)
    BATCH_ALLOCATOR_ROOT = r"D:\Seva-Allocation-Checkout\batch_allocator"
    
    # 2. The root folder (extra copy for visibility)
    CHECKOUT_ROOT = r"D:\Seva-Allocation-Checkout"
    
    # Define file names
    FILE_RAW = "vrf_data_raw.csv"
    FILE_CLEANED = "vrf_data_cleaned.csv"

    # Create lists of paths for the functions to use
    PATHS_RAW = [
        os.path.join(BATCH_ALLOCATOR_ROOT, "data", FILE_RAW), # Folder A
        os.path.join(CHECKOUT_ROOT, "data", FILE_RAW)         # Folder B
    ]
    
    PATHS_CLEANED = [
        os.path.join(BATCH_ALLOCATOR_ROOT, "data", FILE_CLEANED), # Folder A
        os.path.join(CHECKOUT_ROOT, "data", FILE_CLEANED)         # Folder B
    ]

    # VRF Input Sheet (Source)
    VRF_SHEET_URL = "https://docs.google.com/spreadsheets/d/1i0ANT5-tamlo6YX9uuMTUwayzL31YEmpgC6qQ-0y7jI/edit?gid=1288139751#gid=1288139751"
    INPUT_TAB = "Input"
    OUTPUT_TAB = "Skills Combined Output"
    
    # AppSheet Backend (Target) - PRODUCTION URL
    APPSHEET_BACKEND_URL = "https://docs.google.com/spreadsheets/d/1UlD-jKkO9HEh_wAwFlcywlrBDYThpL6_s-_GtLOWmAg/edit?gid=1578071586#gid=1578071586"
    
    # AppSheet Backend (Target) - TEST URL
    # APPSHEET_BACKEND_URL = "https://docs.google.com/spreadsheets/d/16VaRQQ6Vu20DlMjC3Uxaj81_RxLQeEEN859FxweLpO8/edit?gid=2124451009#gid=2124451009"

    # Tab Definitions
    TAB_NEW_VRF = "New vrf"
    TAB_OLD_VRF = "Old vrf"
    TAB_MERGED_VRF = "Merged VRF's"
    TAB_MAIN_VRF = "vrf"

    print("\n--- VRF Replacement Pipeline Initialized ---\n")

    # Step 1: Concatenate and Save 'vrf_data_cleaned.csv' to BOTH folders
    run_step(
        "Step 1: Concatenate Skills & Save Cleaned CSVs",
        Concatenate_Skills_By_RequestName,
        sheet_url=VRF_SHEET_URL,
        input_tab_name=INPUT_TAB,
        output_tab_name=OUTPUT_TAB,
        credentials_path=CREDENTIALS_PATH,
        local_cleaned_paths=PATHS_CLEANED # <-- Passing list of paths
    )

    # Step 2: Sync to AppSheet
    run_step(
        "Step 2: Sync to 'New vrf'",
        Sync_VRF_To_AppSheet,
        source_sheet_url=VRF_SHEET_URL,
        source_tab_name=OUTPUT_TAB,
        target_sheet_url=APPSHEET_BACKEND_URL,
        target_tab_name=TAB_NEW_VRF,
        credentials_path=CREDENTIALS_PATH
    )

    # Step 3: Merge Logic (New vrf + Old vrf -> Merged VRF's)
    run_step(
        "Step 3: Merge Old & New to 'Merged VRF's'",
        Merge_Old_And_New,
        sheet_url=APPSHEET_BACKEND_URL,
        old_tab=TAB_OLD_VRF,
        new_tab=TAB_NEW_VRF,
        merged_tab=TAB_MERGED_VRF,
        credentials_path=CREDENTIALS_PATH
    )

    # Step 4: Safe Replace Logic (Merged VRF's + Orphans -> vrf)
    run_step(
        "Step 4: Update Main 'vrf' Table",
        Update_Main_VRF_Table,
        sheet_url=APPSHEET_BACKEND_URL,
        merged_tab=TAB_MERGED_VRF,
        main_vrf_tab=TAB_MAIN_VRF,
        credentials_path=CREDENTIALS_PATH
    )

    # Step 5: Update Pinecone 
    # Logic: Read RAW Input -> Save 'vrf_data_raw.csv' to BOTH folders -> Run module
    run_step(
        "Step 5: Update Pinecone Vectors",
        Run_VRF_Indexer,
        sheet_url=VRF_SHEET_URL, 
        tab_name=INPUT_TAB,
        local_raw_paths=PATHS_RAW, # <-- Passing list of paths
        batch_allocator_root=BATCH_ALLOCATOR_ROOT,
        credentials_path=CREDENTIALS_PATH
    )

    # Final Summary
    Concatenation_Handler.Summarize_Execution(execution_log)

    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02} (HH:MM:SS)")