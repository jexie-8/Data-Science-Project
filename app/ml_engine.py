import pandas as pd
import numpy as np
import joblib
import sys
import os
from mappings import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.processing import clean_data

class StudentPredictor:
    def __init__(self, model_path='models/XGBoost_Pipeline_v1.joblib'):
        self.pipeline = joblib.load(model_path)

    def _prepare_data(self, df):
        # pass 'Target' as a dummy if it doesn't exist in raw UI data
        if 'Target' not in df.columns:
            df['Target'] = 0 
            
        cleaned_df = clean_data(df, 'Target')
        
        # Drop the dummy target and keep only the features the model needs
        return cleaned_df.drop(columns=['Target'])

    def process_and_predict(self, raw):

        df_raw = pd.DataFrame([raw]) # Convert dict to DataFrame
        df_raw = df_raw.apply(pd.to_numeric, errors='coerce') # Convert types
        
        # Clean data (from func above that uses processing.py)
        X_processed = self._prepare_data(df_raw)
        
        # Predict and return label + probability
        prob = self.pipeline.predict_proba(X_processed)[0][1]
        if prob >= 0.60:
            label = "Dropout"
        elif prob <= 0.40:
            label = "Graduate"
        else:
            label = "Uncertain"
        
        return label, f"{prob*100:.1f}%", []

    def process_batch(self, file_storage):
        df_raw = pd.read_csv(file_storage)
        df_raw.columns = df_raw.columns.str.strip()
        X_processed = self._prepare_data(df_raw.copy())
        
        # Batch Predict
        probs = self.pipeline.predict_proba(X_processed)[:, 1]
        
        # Attach to original raw data
        df_raw['Prediction'] = [
            "Dropout" if p >= 0.60 else
            "Graduate" if p <= 0.40 else
            "Uncertain"
            for p in probs
        ]
        df_raw['Probability'] = [f"{p*100:.1f}%" for p in probs]
        
        summary_results = [
            {"Prediction": p, "Probability": prob} 
            for p, prob in zip(df_raw['Prediction'], df_raw['Probability'])
        ]
            
        return summary_results, df_raw.to_dict(orient='records')