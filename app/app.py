from flask import Flask, render_template, request, jsonify
from ml_engine import StudentPredictor
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILENAME = 'XGBoost_Pipeline_v2_20260418_2041.joblib'
MODEL_PATH = os.path.join(ROOT_DIR, 'models', MODEL_FILENAME)

if not os.path.exists(MODEL_PATH):
    logger.error(f"FATAL: Model file not found at {MODEL_PATH}")
    predictor = None
else:
    predictor = StudentPredictor(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        label, prob, warnings = predictor.process_and_predict(data)
        
        return jsonify({
            "prediction": label,
            "probability": prob,
            "warnings": warnings,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({"error": "An internal error occurred during prediction"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not predictor:
        return jsonify({"error": "Model not initialized"}), 500

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Selected file is empty"}), 400

        results, csv_data = predictor.process_batch(file)
        
        return jsonify({
            "results": results,
            "csv_data": csv_data,
            "count": len(results)
        })
    except Exception as e:
        logger.error(f"Batch Processing Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)