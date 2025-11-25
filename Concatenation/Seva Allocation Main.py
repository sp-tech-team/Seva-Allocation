import numpy as np
import pandas as pd
from Libraries.Gspread_Library import GoogleSheetHandler
from Libraries.Concatenation_Library import Concatenation_Handler
from datetime import datetime

# --- Timing Start ---
start_time = datetime.now()

# Parameters
credentials_path = "D:/Seva-Allocation/Concatenation/credentials.json"  # Path to service account credentials file
sheet_url = "https://docs.google.com/spreadsheets/d/1ZcaFKwFiu79cyU3q0e7ds8EPfwaBaGICCsXHpX8iOl4/edit?gid=1371297251#gid=1371297251"
input_tab_name = "Input"  # Input tab in the Google Sheet
output_tab_name = "Formatted Output"  # Output tab to write the processed data
front_fill_tab_name = "Front Filled" # Front Fill tab for language and interview columns process in appsheet

# Sheet names for the second step (Mapping)
id_mapping_tab_name = "ID Mappings"
final_output_tab_name = "Final Output"

# AppSheet Backend Config

# Test Appsheet Sheet
# appsheet_backend_url = "https://docs.google.com/spreadsheets/d/16VaRQQ6Vu20DlMjC3Uxaj81_RxLQeEEN859FxweLpO8/edit?gid=2124451009#gid=2124451009"

# PRODUCTION URL - Actual Appsheet sheet [Note: Beware before updating the main appsheet sheet - use test sheet until then]
appsheet_backend_url = "https://docs.google.com/spreadsheets/d/1UlD-jKkO9HEh_wAwFlcywlrBDYThpL6_s-_GtLOWmAg/edit?gid=1578071586#gid=1578071586"#


# Location for Seva Allocation Input data
local_output_path = r"D:\Seva-Allocation-Checkout\batch_allocator\data"

# Seva Allocation checkout path to run seva allocation automatically
batch_allocator_path = r"D:\Seva-Allocation-Checkout\batch_allocator"

# Derived path for saving intermediate files
# local_save_path = os.path.join(batch_allocator_path, "data")

# --- NEW: Configuration for final step ---
upload_final_results_to_drive = False # Set to True to enable optional GDrive upload

# --- NEW: Define the target sheet name for the final results ---
seva_allocation_sheet_name = "Seva Allocation"

print("\n You can download credentials.json from https://console.cloud.google.com/")
print("Google cloud console -> API & Services -> Credentials -> Add Service Account - Add Key and download as Json", end="\n \n")
print("Note: Make sure to give editor access in the sheet for the client_email mentioned in credentials.json", end="\n \n")

# --- Step 1: Run the original concatenation function ---
print("--- Starting Step 1: Concatenation ---")
Concatenation_Handler.Concatenation_Main_Using_GSpread(
    sheet_url=sheet_url,
    input_tab_name=input_tab_name,
    output_tab_name=output_tab_name,
    front_fill_tab_name=front_fill_tab_name,
    credentials_path=credentials_path
)
print("--- Finished Step 1 ---\n")


# --- Step 2: Run the new ID mapping function ---
print("--- Starting Step 2: ID Mapping ---")
Concatenation_Handler.Map_And_Finalize(
    sheet_url=sheet_url,
    formatted_tab_name=output_tab_name,  # The output of step 1 is the input for step 2
    id_mapping_tab_name=id_mapping_tab_name,
    final_output_tab_name=final_output_tab_name,
    credentials_path=credentials_path,
    local_output_path=local_output_path
)
print("--- Finished Step 2 ---")


# --- Step 3: Run the inference model ---
print("--- Starting Step 3: Run Inference Model ---")
Concatenation_Handler.Run_Inference_Model(
    batch_allocator_root=batch_allocator_path
)
print("--- Finished Step 3 ---")


# --- Step 4: Process and Upload the final results ---
print("--- Starting Step 4: Process and Upload Results ---")
Concatenation_Handler.Process_And_Upload_Results(
    batch_allocator_root=batch_allocator_path,
    credentials_path=credentials_path,
    sheet_url=sheet_url,
    final_participant_info_tab=final_output_tab_name,
    target_allocation_sheet_name=seva_allocation_sheet_name, # Pass the new parameter
    upload_to_drive=upload_final_results_to_drive
)
print("--- Finished Step 4 ---")

# --- NEW: Step 5: Sync All Data to AppSheet Backend ---
print("--- Starting Step 5: Sync Data to AppSheet ---")

# ---------------------------------------------------------
# Part 1: Sync Front Filled -> Filled Vlookup Data
# ---------------------------------------------------------
Concatenation_Handler.Sync_Front_Filled_To_AppSheet(
    source_sheet_url=sheet_url,
    source_tab_name=front_fill_tab_name,
    target_sheet_url=appsheet_backend_url,
    target_tab_name="Filled Vlookup Data",
    credentials_path=credentials_path
)

# ---------------------------------------------------------
# Part 2: Sync Final Output -> participants
# ---------------------------------------------------------
Concatenation_Handler.Sync_Participants_To_AppSheet(
    source_sheet_url=sheet_url,
    source_tab_name=final_output_tab_name, 
    target_sheet_url=appsheet_backend_url,
    target_tab_name="participants",
    credentials_path=credentials_path
)

# ---------------------------------------------------------
# Part 3: Sync Seva Allocation -> Current Predictions
# ---------------------------------------------------------


# Step 5c: Sync Allocations (Direct Append)
# Note: Ensure the columns in 'Seva Allocation' are in the same visual order 
# as 'Current Predictions' for this to work perfectly.
Concatenation_Handler.Sync_Allocations_To_AppSheet(
    source_sheet_url=sheet_url,
    source_tab_name=seva_allocation_sheet_name,
    target_sheet_url=appsheet_backend_url,
    target_tab_name="Current Predictions", 
    credentials_path=credentials_path)

# --- NEW: Step 6: Run AppScript Logic (Tweak & Format) ---
print("--- Starting Step 6: Post-Processing (Tweak & Format) ---")

Concatenation_Handler.Run_Post_Processing_Scripts(
    sheet_url=appsheet_backend_url,
    predictions_tab_name="Current Predictions", # The sheet to be tweaked
    formatted_tab_name="Formatted Predictions", # The output sheet to be created/overwritten
    credentials_path=credentials_path
)

print("--- Finished Step 6 ---")

print("\n--- All tasks completed! ---")

# --- Timing End ---
end_time = datetime.now()
duration = end_time - start_time
total_seconds = duration.total_seconds()
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02} (HH:MM:SS)")