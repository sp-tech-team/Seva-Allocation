import pandas as pd
import argparse
import os

def process_file(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Sort the DataFrame by SP_ID if it's not already sorted
    df = df.sort_values(by='SP_ID')
    
    # Add a 'Row Number' column, which is the row number within each SP_ID group
    df['Row Number'] = df.groupby('SP_ID').cumcount() + 1


    ### Only extra added code is below ###
    ######################################
    # Create rank
    # Apply ranking within each customer_id based on order_score in descending order
    df['rank'] = df.groupby('SP_ID')['Skill Score'].rank(method='first', ascending=False)
    # Sort the DataFrame by customer_id and rank in ascending order
    df = df.sort_values(by=['SP_ID', 'rank'])
    ######################################


    # Split 'Predicted Cluster' into 'General Cluster' and 'Specific Cluster'
    df[['General Cluster', 'Specific Cluster']] = df['Predicted Cluster'].str.split(' - ', expand=True)

    # Generate the output file path by adding _PROCESSED before the file extension
    file_name, file_extension = os.path.splitext(file_path)
    output_file_path = f"{file_name}_PROCESSED{file_extension}"

    # Save the updated DataFrame back to a new Excel file with _PROCESSED suffix
    df.to_excel(output_file_path, index=False)
    print(f'The Excel file has been updated and saved as "{output_file_path}".')

if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Process an Excel file to add row numbers within each SP_ID group and split Predicted Cluster, then save with a _PROCESSED suffix.')
    parser.add_argument('file_path', type=str, help='Path to the Excel file to process')
    
    args = parser.parse_args()
    
    # Process the file
    process_file(args.file_path)

