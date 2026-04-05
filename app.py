from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os

application = Flask(__name__)
app = application

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')

#Cargar el modelo
mejor_modelo = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    #Obtener los 6 predictores desde el frontend
    age = float(data['age'])
    bmi = float(data['bmi'])
    children = int(data['children'])
    sex = data['sex']
    smoker = data['smoker']
    region = data['region']
    
    #Calcular las columnas de feature engineering
    is_parent = 1 if children > 0 else 0
    smoker_binary = 1 if smoker == "yes" else 0
    age_bmi = age * bmi
    bmi_smoker = bmi * smoker_binary
    age_smoker = age * smoker_binary
    
    
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region,
        "is_parent": is_parent,
        "smoker_binary": smoker_binary,
        "age_bmi": age_bmi,
        "bmi_smoker": bmi_smoker,
        "age_smoker": age_smoker
    }])
    
    #Hacer la predicción
    prediction_log = mejor_modelo.predict(input_data)[0]
    
    #Convertir de escala logarítmica a escala original
    prediction_original = np.expm1(prediction_log)
    
    return jsonify({"charges": round(prediction_original, 2)})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    application.run(host='0.0.0.0', port=port)