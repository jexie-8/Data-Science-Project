import joblib
import os

def load_production_model(model_path):
    """Loads the serialized XGBoost model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def get_prediction(model, input_data):
    """Returns the risk label and probability."""
    prob = model.predict_proba(input_data)[:, 1][0]
    label = "At-Risk" if prob > 0.5 else "Healthy"
    return label, round(float(prob), 4)