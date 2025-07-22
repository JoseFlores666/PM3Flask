from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import json

app = Flask(__name__)

# Orígenes permitidos (ajusta según tu entorno)
allowed_origins = [
    'http://localhost:5173',
    'https://localhost:5173',
    'https://farma-medic.vercel.app',
    'https://back-farmam.onrender.com'
]

# Configura CORS para esos orígenes
CORS(app, origins=allowed_origins)

# Carga el modelo entrenado
with open('modelo_clasificacion.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        print("Error leyendo metrics.json:", e)  # Esto aparece en la consola de Flask

        return jsonify({"error": "No se pudo leer metrics.json: " + str(e)}), 500
    

@app.route('/rules', methods=['GET'])
def get_rules():
    try:
        with open('rules.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        columnas_esperadas = [
            'hora_num', 'nombre', 'apellidoPaterno',
            'apellidoMaterno', 'especialidad', 'turno', 'dia_semana'
        ]
        for col in columnas_esperadas:
            if col not in df.columns:
                df[col] = 0  # valor por defecto, ajusta si necesitas

        df_model = df[columnas_esperadas]

        pred = model.predict(df_model)
        proba = model.predict_proba(df_model)[:, 1]

        return jsonify({
            "predicciones": pred.tolist(),
            "probabilidades": proba.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
