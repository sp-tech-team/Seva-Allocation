import numpy as np
import pandas as pd


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
    def concatenate_method_for_Local_Downloaded_File(df, group_column, columns_to_concatenate, separator='/-/'):
        """Concatenates specified columns within a DataFrame, grouped by another column."""
        grouped_df = df.groupby(group_column)
        for column in columns_to_concatenate:
            if column in df.columns:
                df[column] = grouped_df[column].transform(lambda x: separator.join(y for y in x if y != 'nan'))
                df[column] = df[column].str.split(separator)
        return df

    @staticmethod
    def convert_columns_to_string(df, columns):
        """Converts specified columns in a DataFrame to string type."""
        for column in columns:
            if column in df.columns:
                df[column] = df[column].astype(str)
        return df

    def Concatenation_Main_Using_Local_Downloaded_File(df, fill_cols, concat_cols, key_col):
        """
        Processes a locally downloaded Excel file by:
        - Front-filling specified columns
        - Concatenating work experience, education, skills, and hobby-related columns
        - Exporting transformed data to a new Excel file with multiple tabs

        :param input_file: Path to the input Excel file
        :param output_file: Path to the output Excel file
        """
        if key_col not in fill_cols:
            raise ValueError(f"Key column '{key_col}' must be included in the fill columns.")
        df = df.copy()
        df = df[fill_cols + concat_cols]
        fill_str = 'NA'
        # Fill empty cells for rows with non-empty IDs
        df = df.apply(
            lambda row: row.fillna(fill_str) if not pd.isna(row[key_col]) and row.isnull().any() else row,
            axis=1
        )
        # Replace empty strings with 'NA' in rows with valid IDs
        df.loc[~df[key_col].isna()] = df.loc[~df[key_col].isna()].replace('', fill_str)
        
        # Apply front-fill function
        df = Concatenation_Handler.front_fill_columns(df, fill_cols)

        # Ensure appropriate columns are converted to strings before concatenation
        df = Concatenation_Handler.convert_columns_to_string(df, concat_cols)

        # Apply concatenation function
        df = Concatenation_Handler.concatenate_method_for_Local_Downloaded_File(df, key_col, concat_cols)

        # Retain only relevant columns
        columns_to_keep = fill_cols + concat_cols
        df_exported_filtered = df[columns_to_keep].drop_duplicates(key_col).reset_index(drop=True)
        return df_exported_filtered