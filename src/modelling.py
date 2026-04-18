import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


def get_student_pipeline(numeric_cols):
    
    # 1. Define the column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[('num', RobustScaler(), numeric_cols)],
        remainder='passthrough' 
    )

    # 2. Create the full pipeline with XGBoost as the final estimator
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            colsample_bytree=0.8,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    
    print("Modeling pipeline created successfully.")

    return pipeline