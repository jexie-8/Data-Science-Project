import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Helper: Caps outliers using the IQR method.
def cap_outliers(df, column):
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# Main function to clean the student dataset.
def clean_data(df, target_column):
   
   # 1. Make a copy of the original dataframe to avoid modifying it directly.
    clean_df = df.copy()
    
    # 2. Handle missing values by filling numerical columns with their median & drop duplicates.
    numerical_cols = clean_df.select_dtypes(include=np.number).columns
    clean_df[numerical_cols] = clean_df[numerical_cols].fillna(clean_df[numerical_cols].median())
    clean_df = clean_df.drop_duplicates()

    # 3. Drop rows where the target variable is missing, as we cannot train on those.
    if target_column in clean_df.columns:
        if clean_df[target_column].isnull().any():
            clean_df = clean_df.dropna(subset=[target_column])
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    # Also remove these students that were flagge dduirng data integrity checking in data cleaning,
    # as they are graduates with 0 units approved or credited in the final year 
    grad_conflict = clean_df[(clean_df['Target'] == 'Graduate') & (clean_df['Curricular units 2nd sem (approved)'] == 0) 
                             & (clean_df['Curricular units 1st sem (approved)'] == 0) 
                             & (clean_df['Curricular units 1st sem (credited)'] == 0)]
    clean_df = clean_df.drop(grad_conflict.index)

    # 4. Cap outliers in the three key academic features to reduce their influence on the model.
    academic_cols = [
        'Curricular units 1st sem (enrolled)', 
        'Curricular units 1st sem (evaluations)', 
        'Curricular units 1st sem (approved)'
    ]
    for col in academic_cols:
        clean_df = cap_outliers(clean_df, col)


    # 5. Derived Attributes - These are based on EDA insights and domain knowledge to capture complex patterns.
    clean_df['Academic_Efficiency_Ratio'] = clean_df['Curricular units 1st sem (approved)'] / (clean_df['Curricular units 1st sem (evaluations)'] + 1e-9) 
    clean_df['Financial_Risk_Score'] = clean_df['Debtor'] + (1 - clean_df['Tuition fees up to date'])
    clean_df['is_mature_entry'] = ((clean_df['Age at enrollment'] > 25) & (clean_df['Application mode'].isin([8, 39]))).astype(int)
    clean_df['is_zero_sem1'] = (clean_df['Curricular units 1st sem (grade)'] == 0).astype(int)
    clean_df['is_single'] = (clean_df['Marital status'] == 1).astype(int)
    clean_df['is_standard_secondary'] = (clean_df['Previous qualification'] == 1).astype(int)
    clean_df['has_credits'] = (clean_df['Curricular units 1st sem (credited)'] > 0).astype(int)
    clean_df['Total_Academic_Momentum'] = clean_df['Curricular units 1st sem (approved)'] + clean_df['Curricular units 2nd sem (approved)']
    clean_df['late_switcher'] = ((clean_df['Age at enrollment'] > 22) & (clean_df['Application mode'] == 15)).astype(int)
    clean_df['Grade_Velocity_Trend'] = clean_df['Curricular units 2nd sem (grade)'] - clean_df['Curricular units 1st sem (grade)']
    clean_df['Recovery_Signal'] = clean_df['Grade_Velocity_Trend'] * clean_df['is_zero_sem1']
    clean_df['Age_at_Enrollment_Log'] = np.log1p(clean_df['Age at enrollment'])

    # 6. Risk Mapping
    risk_mapping = {'Dropout': 1.0, 'Enrolled': 0.5, 'Graduate': 0.0}
    temp_target_for_risk = clean_df['Target'].map(risk_mapping)
    high_card_features = ["Mother's occupation", "Father's occupation", "Application mode"]
    for col in high_card_features:
        if col in clean_df.columns:
            # Map risk scores and fill rare categories with the global mean risk
            encoding_map = temp_target_for_risk.groupby(clean_df[col]).mean()
            new_col_name = f"{col.replace(' ', '_')}_risk"
            clean_df[new_col_name] = clean_df[col].map(encoding_map).fillna(temp_target_for_risk.mean())

    # 7. Drop rows that are not needed for modeling. This is also based on EDA insights.
    drop_col = [
    'Unemployment rate', 'Inflation rate', 'GDP', 'Debtor', 
    'Nacionality', 'International', 'Educational special needs',
    'Curricular units 1st sem (without evaluations)', 'Application mode',
    'Curricular units 2nd sem (without evaluations)', 'Tuition fees up to date',
    'Curricular units 1st sem (credited)', 'Previous qualification',
    'Curricular units 2nd sem (credited)', 'Marital status', 'Age at enrollment',
    'Curricular units 2nd sem (enrolled)', "Father's occupation", 
    'Curricular units 2nd sem (evaluations)', "Mother's occupation",
    'Curricular units 2nd sem (grade)', "Father's qualification", 
    'Curricular units 2nd sem (approved)', "Mother's qualification", ]
    clean_df = clean_df.drop(columns=[c for c in drop_col if c in clean_df.columns])



    # 8. Mapping to convert the 'Target' strings into integers for the model to predict.
    target_mapping = {'Dropout': 1, 'Graduate': 0, 'Enrolled': 2}
    clean_df['Target'] = clean_df['Target'].map(target_mapping)


    # 9. Data Type Conversions 
    binary_cols = [col for col in clean_df.columns if col.startswith('is_') or col == 'has_credits']
    clean_df[binary_cols] = clean_df[binary_cols].astype(int)
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    clean_df[numeric_cols] = clean_df[numeric_cols].astype(float)


    # 10. Column Reordering (Target Last) 
    feature_cols = [col for col in clean_df.columns if col != target_column]
    clean_df = clean_df[feature_cols + [target_column]].copy()


    print("Data cleaning complete. Final dataset ready for modeling.")

    return clean_df

