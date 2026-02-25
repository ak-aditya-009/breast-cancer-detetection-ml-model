from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load your saved model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
feature_names = ['concave points_mean', 'area_worst', 'radius_worst', 'perimeter_worst']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 2. Get data from the HTML form
    input_features = [float(x) for x in request.form.values()]
    
    # 3. Convert to DataFrame and Scale
    input_df = pd.DataFrame([input_features], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    
    # 4. Predict
    prediction = model.predict(input_scaled)
    probability = np.max(model.predict_proba(input_scaled)) * 100

    # Standard mapping: 0 = Malignant, 1 = Benign
    result = "Malignant" if prediction[0] == 0 else "Benign"

    return render_template('index.html', 
                           prediction_text=f'Diagnosis: {result}',
                           confidence_text=f'Confidence: {probability:.2f}%')

if __name__ == "__main__":
    app.run(debug=True)