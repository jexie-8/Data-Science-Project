import pandas as pd
import numpy as np
import joblib
from mappings import *

class StudentPredictor:
    def __init__(self, model_path='models/XGBoost_v1_20260409_2342.joblib'):
        self.model = joblib.load(model_path)

    def _get_risk_value(self, val, mapping, field_name, warnings):
        try:
            val_int = int(val)
            return mapping.get(val_int, GLOBAL_MEAN_RISK)
        except:
            return GLOBAL_MEAN_RISK

    def process_and_predict(self, raw):
        """Logic for a single prediction from the UI form."""
        warnings = []
        
        def get_v(key, default=0.0): 
            return float(raw.get(key, default))

        # --- Feature Engineering ---
        s1_enrolled = get_v('Curricular units 1st sem (enrolled)')
        s1_evals = get_v('Curricular units 1st sem (evaluations)')
        s1_appr = get_v('Curricular units 1st sem (approved)')
        s1_grade = get_v('Curricular units 1st sem (grade)')
        s2_appr = get_v('Curricular units 2nd sem (approved)')
        s2_grade = get_v('Curricular units 2nd sem (grade)')
        age = get_v('Age at enrollment')
        app_mode = int(get_v('Application mode', 1))

        f = {}
        f["Marital status"] = get_v("Marital status")
        f["Application order"] = get_v("Application order")
        f["Course"] = get_v("Course")
        f["Daytime/evening attendance"] = get_v("Daytime/evening attendance")
        f["Previous qualification"] = get_v("Previous qualification")
        f["Displaced"] = get_v("Displaced")
        f["Gender"] = get_v("Gender")
        f["Scholarship holder"] = get_v("Scholarship holder")
        f["Curricular units 1st sem (enrolled)"] = s1_enrolled
        f["Curricular units 1st sem (evaluations)"] = s1_evals
        f["Curricular units 1st sem (approved)"] = s1_appr
        f["Curricular units 1st sem (grade)"] = s1_grade
        f["Curricular units 2nd sem (grade)"] = s2_grade
        f["Curricular units 2nd sem (approved)"] = s2_appr

        f['Academic_Efficiency_Ratio'] = s1_appr / (s1_evals + 1e-9)
        f['Financial_Risk_Score'] = get_v('Debtor') + (1 - get_v('Tuition fees up to date'))
        f['is_mature_entry'] = 1.0 if (age > 25 and app_mode in [8, 39]) else 0.0
        f['is_zero_sem1'] = 1.0 if s1_grade == 0 else 0.0
        f['is_single'] = 1.0 if f["Marital status"] == 1 else 0.0
        f['is_standard_secondary'] = 1.0 if f["Previous qualification"] == 1 else 0.0
        f['has_credits'] = 1.0 if get_v('Curricular units 1st sem (credited)') > 0 else 0.0
        f['Total_Academic_Momentum'] = s1_appr + s2_appr
        f['late_switcher'] = 1.0 if (age > 22 and app_mode == 15) else 0.0
        f['Grade_Velocity_Trend'] = s2_grade - s1_grade
        f['Recovery_Signal'] = f['Grade_Velocity_Trend'] * f['is_zero_sem1']
        
        f["Mother's_occupation_risk"] = self._get_risk_value(get_v("Mother's occupation"), MOTHER_OCC_MAP, "Mother", warnings)
        f["Father's_occupation_risk"] = self._get_risk_value(get_v("Father's occupation"), FATHER_OCC_MAP, "Father", warnings)
        f['Application_mode_risk'] = self._get_risk_value(app_mode, APP_MODE_MAP, "App Mode", warnings)
        f['Age_at_Enrollment_Log'] = np.log1p(age)

        input_df = pd.DataFrame([f])[MODEL_FEATURES]
        prob = self.model.predict_proba(input_df)[0][1]
        label = "Graduate" if prob > 0.5 else "Dropout"
        
        return label, f"{prob*100:.1f}%", warnings

    def process_batch(self, file_storage):
        """Logic for batch processing a CSV file containing RAW data."""
        # 1. Load the raw data
        df_raw = pd.read_csv(file_storage)
        df_raw.columns = df_raw.columns.str.strip()
        
        # 2. Create a working copy for engineering so we don't mess up the original CSV structure
        df = df_raw.copy()

        # 3. Calculate Engineered Features on the copy
        df['Academic_Efficiency_Ratio'] = df['Curricular units 1st sem (approved)'] / (df['Curricular units 1st sem (evaluations)'] + 1e-9)
        df['Financial_Risk_Score'] = df['Debtor'] + (1 - df['Tuition fees up to date'])
        df['is_mature_entry'] = ((df['Age at enrollment'] > 25) & (df['Application mode'].isin([8, 39]))).astype(float)
        df['is_zero_sem1'] = (df['Curricular units 1st sem (grade)'] == 0).astype(float)
        df['is_single'] = (df['Marital status'] == 1).astype(float)
        df['is_standard_secondary'] = (df['Previous qualification'] == 1).astype(float)
        df['has_credits'] = (df['Curricular units 1st sem (credited)'] > 0).astype(float)
        df['Total_Academic_Momentum'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
        df['late_switcher'] = ((df['Age at enrollment'] > 22) & (df['Application mode'] == 15)).astype(float)
        df['Grade_Velocity_Trend'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
        df['Recovery_Signal'] = df['Grade_Velocity_Trend'] * df['is_zero_sem1']
        
        df["Mother's_occupation_risk"] = df["Mother's occupation"].map(MOTHER_OCC_MAP).fillna(GLOBAL_MEAN_RISK)
        df["Father's_occupation_risk"] = df["Father's occupation"].map(FATHER_OCC_MAP).fillna(GLOBAL_MEAN_RISK)
        df['Application_mode_risk'] = df['Application mode'].map(APP_MODE_MAP).fillna(GLOBAL_MEAN_RISK)
        df['Age_at_Enrollment_Log'] = np.log1p(df['Age at enrollment'])

        # 4. Predict using the engineered features
        input_data = df[MODEL_FEATURES]
        probs = self.model.predict_proba(input_data)[:, 1]
        
        # 5. Attach results only to the ORIGINAL RAW DATAFRAME
        # This ensures the downloaded CSV looks like the input CSV + 2 columns
        df_raw['Prediction'] = ["Graduate" if p > 0.5 else "Dropout" for p in probs]
        df_raw['Probability'] = [f"{p*100:.1f}%" for p in probs]
        
        # Format the summary results for the UI table
        summary_results = []
        for index, row in df_raw.iterrows():
            summary_results.append({
                "Prediction": row['Prediction'],
                "Probability": row['Probability']
            })
            
        return summary_results, df_raw.to_dict(orient='records')