import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


def column_reformatting(df):
    """Reformats columns names from CamelCase to Pythonic naming

    Args:
        df ([DataFrame]): DataFrame you want modified

    Returns:
        [DataFrame]: Reformatted DataFrame with Pythonic columns 
    """    
    cols = df.columns.tolist()
    new_cols = []
    for col in cols:
        new_cols.append(re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower())
    df.columns = new_cols
    return df

def drop_columns(df, col_list_to_drop):
    """Drops unwanted columns from DataFrame

    Args:
        df ([DataFrame]): DataFrame you want modified
        col_list_to_drop ([list]): Columns you want dropped

    Returns:
        [DataFrame]: Modified DataFrame with desired columns dropped
    """    
    df.drop(col_list_to_drop, axis=1, inplace=True)
    return df

def change_to_binary(df, change_to_bin_cols):
    """Changes Yes/No Values to binary values 

    Args:
        df ([DataFrame]): DataFrame you want modified
        change_to_bin_cols ([list]): List of columns that need changes

    Returns:
        [DataFrame]: Modified DataFrame with binary values inserted
    """    
    for col in change_to_bin_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

def one_hot_encoding(df, categorical_features):
    """One-Hot Encodes a list of categorical columns and drops the first 
    column to avoid collinearity

    Args:
        df ([DataFrame]): DataFrame you want modified
        categorical_features ([list]): List of categorical columns 

    Returns:
        [DataFrame]: Modified DataFrame with columns one-hot encoded
    """        
        dummies = pd.get_dummies(df[[categorical_features]], drop_first=True)
        new_df = pd.concat([df, dummies], axis=1)
    return new_df


if __name__ == '__main__':
    
    column_reformatting(df)
    
    col_list_to_drop = ['over18', 'standard_hours', 'employee_count', 'employee_number']
    drop_columns(df, col_list_to_drop)
    
    change_to_bin_cols = ['attrition', over_time']
    change_to_binary(df, change_to_bin_cols)

    df = one_hot_encoding(df, categorical_features_list)
