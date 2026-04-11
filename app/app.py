# app.py
from flask import Flask, render_template, request, jsonify
from ml_engine import StudentPredictor
import os

app = Flask(__name__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Build the path to the model file inside root/model/
# Adjust the filename extension (.pkl or .joblib) to match your actual file!
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'XGBoost_v1_20260409_2342.joblib')

if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
    # You might want to exit here so you don't hunt ghosts later

# 3. Initialize with the absolute path
predictor = StudentPredictor(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        label, prob, warnings = predictor.process_and_predict(data)
        
        return jsonify({
            "prediction": label,
            "probability": prob,
            "warnings": warnings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        results, csv_data = predictor.process_batch(request.files['file'])
        return jsonify({
            "results": results,
            "csv_data": csv_data
        })
    except Exception as e:
        print(f"BATCH ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

