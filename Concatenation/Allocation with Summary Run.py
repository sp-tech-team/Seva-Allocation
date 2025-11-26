import numpy as np
import pandas as pd
from Libraries.Gspread_Library import GoogleSheetHandler
from Libraries.Concatenation_Library import Concatenation_Handler
from datetime import datetime
import traceback

# --- Global Logging List ---
execution_log = []

def run_step(step_name, func, *args, **kwargs):
    """
    Helper to run a function, track its status, and log the result.
    """
    print(f"\n--- Starting {step_name} ---")
    start = datetime.now()
    
    try:
        # Run the function
        result = func(*args, **kwargs)
        
        # Check if the function explicitly returned False (failure)
        if result is False:
            status = False
            error_msg = "Function returned 'False' (Check internal logs)"
        else:
            status = True
            error_msg = None
            
    except Exception as e:
        status = False
        error_msg = str(e)
        # Optional: Print the full traceback so you can debug
        traceback.print_exc()

    duration = datetime.now() - start
    print(f"--- Finished {step_name} (Time: {duration.total_seconds():.2f}s) ---\n")
    
    # Append to log for Step 7
    execution_log.append({
        "name": step_name,
        "status": status,
        "error": error_msg
    })

# --- Configuration ---
start_time = datetime.now()
credentials_path = "D:/Seva-Allocation/Concatenation/credentials.json"
sheet_url = "https://docs.google.com/spreadsheets/d/1ZcaFKwFiu79cyU3q0e7ds8EPfwaBaGICCsXHpX8iOl4/edit?gid=1371297251#gid=1371297251"
input_tab_name = "Input"
output_tab_name = "Formatted Output"
front_fill_tab_name = "Front Filled"
id_mapping_tab_name = "ID Mappings"
final_output_tab_name = "Final Output"
local_output_path = r"D:\Seva-Allocation-Checkout\batch_allocator\data"
batch_allocator_path = r"D:\Seva-Allocation-Checkout\batch_allocator"
upload_final_results_to_drive = False
seva_allocation_sheet_name = "Seva Allocation"

# AppSheet Backend Config
appsheet_backend_url = "https://docs.google.com/spreadsheets/d/16VaRQQ6Vu20DlMjC3Uxaj81_RxLQeEEN859FxweLpO8/edit?gid=2124451009#gid=2124451009"

print("\n--- Seva Allocation Pipeline Initialized ---\n")

# ==========================================================
# EXECUTION PIPELINE
# ==========================================================

# Step 1: Concatenation
run_step(
    "Step 1: Concatenation & Formatting",
    Concatenation_Handler.Concatenation_Main_Using_GSpread,
    sheet_url=sheet_url,
    input_tab_name=input_tab_name,
    output_tab_name=output_tab_name,
    front_fill_tab_name=front_fill_tab_name,
    credentials_path=credentials_path
)

# Step 2: ID Mapping
run_step(
    "Step 2: ID Mapping & Export",
    Concatenation_Handler.Map_And_Finalize,
    sheet_url=sheet_url,
    formatted_tab_name=output_tab_name,
    id_mapping_tab_name=id_mapping_tab_name,
    final_output_tab_name=final_output_tab_name,
    credentials_path=credentials_path,
    local_output_path=local_output_path
)

# Step 3: Inference Model
run_step(
    "Step 3: AI Inference Model",
    Concatenation_Handler.Run_Inference_Model,
    batch_allocator_root=batch_allocator_path
)

# Step 4: Upload Results
run_step(
    "Step 4: Process & Upload Results",
    Concatenation_Handler.Process_And_Upload_Results,
    batch_allocator_root=batch_allocator_path,
    credentials_path=credentials_path,
    sheet_url=sheet_url,
    final_participant_info_tab=final_output_tab_name,
    target_allocation_sheet_name=seva_allocation_sheet_name,
    upload_to_drive=upload_final_results_to_drive
)

# Step 5a: Sync Front Filled
run_step(
    "Step 5a: Sync 'Front Filled' Data",
    Concatenation_Handler.Sync_Front_Filled_To_AppSheet,
    source_sheet_url=sheet_url,
    source_tab_name=front_fill_tab_name,
    target_sheet_url=appsheet_backend_url,
    target_tab_name="Filled Vlookup Data",
    credentials_path=credentials_path
)

# Step 5b: Sync Participants
run_step(
    "Step 5b: Sync 'Participants' Data",
    Concatenation_Handler.Sync_Participants_To_AppSheet,
    source_sheet_url=sheet_url,
    source_tab_name=final_output_tab_name, 
    target_sheet_url=appsheet_backend_url,
    target_tab_name="participants",
    credentials_path=credentials_path
)

# Step 5c: Sync Allocations (Direct Append)
# Note: Ensure the columns in 'Seva Allocation' are in the same visual order 
# as 'Current Predictions' for this to work perfectly.
run_step(
    "Step 5c: Sync 'Allocations' Data",
    Concatenation_Handler.Sync_Allocations_To_AppSheet,
    source_sheet_url=sheet_url,
    source_tab_name=seva_allocation_sheet_name,
    target_sheet_url=appsheet_backend_url,
    target_tab_name="Current Predictions", 
    credentials_path=credentials_path
    # Removed column_mapping argument
)

# Step 6: Post Processing
run_step(
    "Step 6: Post-Processing (Tweak & Format)",
    Concatenation_Handler.Run_Post_Processing_Scripts,
    sheet_url=appsheet_backend_url,
    predictions_tab_name="Current Predictions",
    formatted_tab_name="Formatted Predictions",
    credentials_path=credentials_path
)

# ==========================================================
# STEP 7: FINAL SUMMARY
# ==========================================================
Concatenation_Handler.Summarize_Execution(execution_log)

# --- Timing End ---
end_time = datetime.now()
duration = end_time - start_time
print(f"Total script execution time: {duration}")