"""
data_cleaning.py
--------
This is to visualise the data cleaning process.
"""

import pandas as pd

# ----------------------------
# 1. Replace Missing Values
# ----------------------------
def replace_missing_value(df):
    """
    Function used in -> load_and_clean_data().
    Imputes missing values (NaN) in the DataFrame based on column data type.
    - String columns (object type) are filled with 'Unknown'.
    - All other column types (numeric, datetime, etc.) are filled with "NA".

    Args:
        df (pd.DataFrame): The DataFrame to modify in place.
    Returns:
        pd.DataFrame: The DataFrame with imputed values.
    """
    for col in df.columns:
        if df[col].dtype == 'O':  # Object type (string)
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna("NA")
    return df


# ----------------------------
# 2. Remove Special Characters
# ----------------------------
def remove_special_characters(df, removechar, char=''):
    """
    Function used in -> load_and_clean_data().
    Removes a predefined list of noise characters from specific text columns.

    - Characters are replaced with a single space in the 'Airlines' column 
      to prevent word concatenation.
    - Characters are completely removed (replaced with an empty string) in 
      the 'Text Content' column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        removechar (list): A list of strings/characters to search for and remove.
        char (str): An unused parameter (retained from original signature).
        
    Returns:
        pd.DataFrame: The DataFrame with characters removed from specified columns.
    """
    # Ensure target columns exist
    if 'Airlines' not in df.columns or 'Text Content' not in df.columns:
        raise KeyError("Columns 'Airlines' or 'Text Content' not found in DataFrame.")
    
    for c in removechar:
        df['Airlines'] = df['Airlines'].str.replace(c, ' ', regex=False)
        df['Text Content'] = df['Text Content'].str.replace(c, '', regex=False)
    return df


# ----------------------------
# 3. Main Data Cleaning Function
# ----------------------------
def load_and_clean_data(PATH):
    """
    Data Loading & Cleaning
    Cleans the input CSV by dropping duplicates and NaNs, removing noise characters, 
    and standardizing text casing.
    
    Args:
        PATH (str): The path of the CSV file containing the data.
    
    Returns:
        pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    # Load data
    df = pd.read_csv(PATH)
    
    # Remove duplicates and NaNs
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Characters to remove
    removechar = [
        '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+',
        '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', '?',
        '/', '~', '`', '✅ Trip Verified', 'Not Verified', 'Â Â', '✅ Verified Review'
    ]
    
    # Apply cleaning functions
    df = replace_missing_value(df)
    df = remove_special_characters(df, removechar, char='')

    # Standardize text case
    if 'Airlines' in df.columns:
        df['Airlines'] = df['Airlines'].str.title().str.lstrip()
    if 'Name' in df.columns:
        df['Name'] = df['Name'].str.title().str.lstrip()
    if 'Text Content' in df.columns:
        df['Text Content'] = df['Text Content'].str.lstrip()
        df['Text Content Lower Case'] = df['Text Content'].str.lower()
    if 'Date Published' in df.columns:
        df['Date Published'] = df['Date Published'].astype(str).str.lstrip()
    
    # Export cleaned data
    output_path = 'data/airlines_review_cleaned.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaning complete! Saved cleaned file to: {output_path}")
    print(f"📊 Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


# ----------------------------
# 4. Run when executed directly
# ----------------------------
if __name__ == "__main__":
    PATH = 'data/airlines_review.csv'  # adjust if needed
    cleaned_df = load_and_clean_data(PATH)
    print(cleaned_df.head())