def column_reformatting(df):
    cols = df.columns.tolist()
    new_cols = []
    for col in cols:
        new_cols.append(re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower())
    df.columns = new_cols
    return df.head

def drop_columns(df, col_list_to_drop):
    df.drop(col_list_to_drop, axis=1, inplace=True)
    return df.head

def change_to_binary(df, change_to_bin_cols):
    for col in change_to_bin_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df.head()

def one_hot_encoding(df, categorical_feature):
        dummies = pd.get_dummies(df[[categorical_feature]], drop_first=True)
        new_df = pd.concat([df, dummies], axis=1)
    return new_df


if __name__ == '__main__':
    
    # column_reformatting(df)
    
    # col_list_to_drop = ['over18', 'standard_hours', 'employee_count', 'employee_number']
    # drop_columns(df, col_list_to_drop)
    
    # change_to_bin_cols = ['attrition', over_time']
    # change_to_binary(df, change_to_bin_cols)

    # df = one_hot_encoding(df, 'marital_status')
