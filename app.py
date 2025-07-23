from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json
import joblib

app = Flask(__name__)

# Habilitar CORS
CORS(app, origins=[
    'http://localhost:5173',
    'https://localhost:5173',
    'https://farma-medic.vercel.app',
    'https://back-farmam.onrender.com'
])

# Cargar modelo y preprocesador
model = joblib.load('modelo_kmeans_riesgo.pkl')
preprocessor = joblib.load('preprocessor_kmeans.pkl')

# Dataset base para otros endpoints
df = pd.read_csv('dataset_pacientes_riesgo_es.csv')
@app.route('/predecir', methods=['GET'])
def predecir_todos():
    try:
        columnas_esperadas = ['edad', 'hipertension', 'diabetes', 'alcoholismo', 'discapacidad', 'no_asistio', 'sexo']
        df_model = df[columnas_esperadas].copy()

        # Mapear sexo
        map_sexo = {'0': 'M', '1': 'F'}
        df_model['sexo'] = df_model['sexo'].astype(str).map(map_sexo).fillna('M')

        X_pre = preprocessor.transform(df_model)
        pred = model.predict(X_pre)
        riesgos = {0: 'Medio Riesgo', 1: 'Alto Riesgo', 2: 'Bajo Riesgo'}
        resultados = [riesgos.get(p, 'Desconocido') for p in pred]

        # Añadir columnas con resultados al dataframe original para enviar todo
        df_respuesta = df_model.copy()
        df_respuesta['cluster'] = pred
        df_respuesta['riesgo'] = resultados

        # Convertir a lista de dicts para JSON
        data_list = df_respuesta.to_dict(orient='records')

        return jsonify(data_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route('/reporte_clasificacion', methods=['GET'])
def reporte_clasificacion():
    try:
        with open('classification_report.json', 'r') as f:
            reporte = json.load(f)
        return jsonify(reporte)
    except FileNotFoundError:
        return jsonify({"error": "Archivo classification_report.json no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/datos_cantidad', methods=['GET'])
def datos_cantidad():
    try:
        with open('datos_cantidad.json', 'r') as f:
            datos = json.load(f)
        return jsonify(datos)
    except FileNotFoundError:
        return jsonify({"error": "Archivo datos_cantidad.json no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/datos_riesgo', methods=['GET'])
def datos_riesgo():
    try:
        with open('datos_riesgo.json', 'r') as f:
            datos = json.load(f)
        return jsonify(datos)
    except FileNotFoundError:
        return jsonify({"error": "Archivo datos_riesgo.json no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/datos_heatmap', methods=['GET'])
def datos_heatmap():
    try:
        with open('datos_heatmap.json', 'r') as f:
            datos = json.load(f)
        return jsonify(datos)
    except FileNotFoundError:
        return jsonify({"error": "Archivo datos_heatmap.json no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5001, debug=True)
