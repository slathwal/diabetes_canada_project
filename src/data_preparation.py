
import pandas as pd
import numpy as np


# Keep given codes in a column in a dataframe
def keep_known_codes_in_column(df, col_name, codes_list):
    """
    This function is used to filter a dataframe to keep only the rows
    containing the codes in the codes_list variable for the given column
    specified by col_name.
    """
    return df[df[col_name].isin(codes_list)]

# Remove given codes from a column in a dataframe
def remove_known_codes_in_column(df, col_name, codes_list):
    """
    This function is used to filter a dataframe to remove the rows
    containing the codes in the codes_list variable for the given column
    specified by col_name.
    """
    return df[~df[col_name].isin(codes_list)]


# Identify codes indicating missing data in all columns of a dataframe and convert them to na
def convert_missing_codes_to_na(df):
    """
    This function is used to identify the codes in each column of dataframe
    that correspond to missing data and convert them to na values. The function
    returns a dataframe with codes corresponding to missing data replaced
    with na.
    """
    df_cleaned = df.copy()
    missing_summary = {}
    #max_val_list = []
    for col in df.columns:
        #print(col)
        if df[col].dtype in [np.float64, np.int64]:
            unique_vals = df[col].dropna().unique()
            #print(len(unique_vals), unique_vals)
            if len(unique_vals) == 0:
                continue
            max_val = max(unique_vals)
            #max_val_list.append(max_val)
            #print(max_val)
            missing_codes = []
            # Identify the missing value codes based on magnitude
            if max_val < 10:  # single digits (1–9)
                missing_codes = [6, 7, 8, 9]
            elif max_val < 100:  # double digits (01–99)
                missing_codes = [96, 97, 98, 99]
            elif max_val < 1000:  # triple digits (001–999)
                missing_codes = [996, 997, 998, 999, 999.6, 999.7, 999.8, 999.9]
            elif max_val < 10000:
                missing_codes = [9996, 9997, 9998, 9999, 9999.90, 9999.80, 9999.70, 9999.60]
            elif max_val < 100000:
                missing_codes = [99996, 99997, 99998, 99999]
            else:
                continue
            
            # Count and replace
            if missing_codes:
                counts = {code: (df[col] == code).sum() for code in missing_codes}
                if any(counts.values()):
                    missing_summary[col] = counts
                df_cleaned[col] = df_cleaned[col].replace(missing_codes, np.nan)

    # Return cleaned data
    return df_cleaned


def separate_df_into_train_and_test(df, target_col):
    """Separate a dataframe containing a target column into
    a training dataframe and a target pandas series"""
    X = df.copy().drop(columns = target_col, inplace = False)
    y = df[target_col].copy()
    return (X, y)