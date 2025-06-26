from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Cargar el modelo y el scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
app.logger.debug('Modelo y scaler cargados correctamente.')

# Lista de las variables que el modelo espera (en orden)
final_features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Glazing_Area', 'Roof_Area']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados desde el formulario
        compactness = float(request.form['compactness'])
        surface_area = float(request.form['surface_area'])
        wall_area = float(request.form['wall_area'])
        glazing_area = float(request.form['glazing_area'])
        roof_area = float(request.form['roof_area'])

        # Crear DataFrame con los datos nuevos
        data_df = pd.DataFrame([[compactness, surface_area, wall_area, glazing_area, roof_area]],
                               columns=final_features)
        app.logger.debug(f'DataFrame recibido: {data_df}')

        # Escalar los datos nuevos
        data_scaled = scaler.transform(data_df)

        # Realizar la predicción
        prediction = model.predict(data_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')

        return jsonify({'prediccion': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
