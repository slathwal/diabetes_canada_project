
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import chi2_contingency


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

# If a dataframe contains the target column, separate it from the dataframe
def separate_df_into_train_and_test(df, target_col):
    """Separate a dataframe containing a target column into
    a training dataframe and a target pandas series"""
    X = df.copy().drop(columns = target_col, inplace = False)
    y = df[target_col].copy()
    return (X, y)

def return_columns_with_missing_data(df, pct_threshold):
    """
    Given a dataframe, this function returns a list of columns
    with missing data percentange greater than or equal to 
    pct_threshold
    """
    df_nan = convert_missing_codes_to_na(df)
    cols_missing_data = pd.Series(round(df_nan.isna().sum()/df_nan.shape[0]*100,2))
    return df_nan.columns[cols_missing_data >= pct_threshold]

def cols_with_zero_variance(df, variance_threshold = 0):
    """
    Given a dataframe, this function returns a list of columns
    with variance less than variance_threshold across all rows.
    By default, it returns columns that have same value across
    all rows.
    """
    #df = convert_missing_codes_to_na(df)
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(df)
    #selector.variances_
    zero_variance_cols = df.columns[~selector.get_support()]
    print(zero_variance_cols)
    return zero_variance_cols

def additional_cols_to_exclude(df, col_list=['ADM_RNO1', 'VERDATE', 'REFPER', 'ADM_PRX','GEOGPRV', 'GEODGHR4', 'ADM_045','WTS_M'] ):
    col_list_return = []
    for col in col_list:
        if col in df.columns:
            col_list_return.append(col)
    for col in df.columns[df.columns.str.startswith("DO")]:
        col_list_return.append(col)
    # print(col_list_return)
    return col_list_return


def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k - 1, r - 1))

def find_high_correlations(corr_matrix, threshold=0.8):
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                correlated_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
    return correlated_pairs


def cols_with_high_correlation(df, ord_cols, cat_cols, corr_threshold = 0.7):
    
    df_nan = convert_missing_codes_to_na(df)
    
    # For ordinal variables
    spearman_corr = df_nan[ord_cols].corr(method = "spearman").abs()

    df_encoded = df_nan.copy()
    for col in cat_cols:
        df_encoded[col] = df_nan[col].astype('category').cat.codes
    #print(df_encoded)

    cramers_results = pd.DataFrame(index=cat_cols, columns=cat_cols)

    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 == col2:
                cramers_results.loc[col1, col2] = 1.0
            else:
                val = cramers_v(df_encoded[col1], df_encoded[col2])
                cramers_results.loc[col1, col2] = round(val, 3)
    
    correlated_ord_cols = find_high_correlations(spearman_corr.astype(float), threshold = corr_threshold)
    correlated_cat_cols = find_high_correlations(cramers_results.astype(float), threshold = corr_threshold)

    # Collect correlated columns to drop

    corr_ord_cols_to_drop = set()
    for col1, col2, _ in correlated_ord_cols:
        corr_ord_cols_to_drop.add(col2)
    corr_ord_cols_to_drop = list(corr_ord_cols_to_drop)

    corr_cat_cols_to_drop = set()
    for col1, col2, corr in correlated_cat_cols:
        corr_cat_cols_to_drop.add(col2)
    corr_cat_cols_to_drop = list(corr_cat_cols_to_drop)

    corr_cols_to_drop = corr_ord_cols_to_drop+corr_cat_cols_to_drop

    return corr_ord_cols_to_drop, corr_cat_cols_to_drop
