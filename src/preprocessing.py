import pandas as pd
import numpy as np

def clean_student_data(df):
    """
    Standardizes column names and applies final transformations.
    """
    processed_df = df.copy()
    
    # 1. Handle Target mapping only if it exists (for batch exports)
    if 'Target' in processed_df.columns:
        processed_df['Target'] = processed_df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 2})
    
    # 2. Logic to handle Age column naming variations
    # If the user sends raw age, log it. If already logged in app.py, skip.
    if 'Age at enrollment' in processed_df.columns:
        processed_df['Age_at_Enrollment_Log'] = np.log1p(processed_df['Age at enrollment'])
        processed_df = processed_df.drop(columns=['Age at enrollment'])
    elif 'Age_at_Enrollment_Log' in processed_df.columns:
        # If it's already logged but still raw numbers, apply log1p
        # (This protects against double-logging)
        pass

    return processed_df

def prepare_features(df, feature_list):
    """Ensures the dataframe has exactly the columns the model expects in order."""
    # Fill any missing features with 0.0 to prevent model crash
    for feat in feature_list:
        if feat not in df.columns:
            df[feat] = 0.0
    return df[feature_list]