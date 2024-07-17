from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Función para cargar los modelos
def cargar_modelo(modelo_nombre):
    with open(modelo_nombre, 'rb') as f:
        modelo_cargado = pickle.load(f)
    return modelo_cargado

# Ruta para la página principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para la predicción de MatchKills (regresión)
@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    # Cargar el modelo de regresión
    svr_modelado = cargar_modelo('svr_modelado.pkl')

    if request.method == 'POST':
        # Obtener valores del formulario
        match_headshots = float(request.form['MatchHeadshots'])
        match_flank_kills = float(request.form['MatchFlankKills'])
        match_assists = float(request.form['MatchAssists'])
        round_kills = float(request.form['RoundKills'])
        primary_pistol = float(request.form['PrimaryPistol'])
        match_winner_true = int(request.form['MatchWinner_True'])
        
        # Crear un array con los valores de entrada para la regresión
        features_regression = np.array([[match_headshots, match_flank_kills, match_assists, round_kills, primary_pistol, match_winner_true]])

        # Predecir MatchKills utilizando el modelo de regresión
        match_kills_prediction = svr_modelado.predict(features_regression)[0]

        # Descargar el modelo de regresión para evitar mantenerlo en memoria
        del svr_modelado

        return render_template('result_regression.html', match_kills=match_kills_prediction)

# Ruta para la predicción de MatchWinner (clasificación)
@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    # Cargar el modelo de clasificación
    gb_modelado = cargar_modelo('gb_modelado.pkl')

    if request.method == 'POST':
        # Obtener valores del formulario
        round_winner_true = int(request.form['RoundWinner_True'])
        survived_false = int(request.form['Survived_False'])
        primary_pistol = float(request.form['PrimaryPistol'])
        match_assists = float(request.form['MatchAssists'])
        round_starting_equipment_value = float(request.form['RoundStartingEquipmentValue'])
        match_kills = float(request.form['MatchKills'])
        team_starting_equipment_value = float(request.form['TeamStartingEquipmentValue'])
        
        # Crear un array con los valores de entrada para la clasificación
        features_classification = np.array([[round_winner_true, survived_false, primary_pistol, match_assists, round_starting_equipment_value, match_kills, team_starting_equipment_value]])

        # Predecir MatchWinner utilizando el modelo de clasificación
        match_winner_prediction = gb_modelado.predict(features_classification)[0]
        match_winner = 'Ganará' if match_winner_prediction == 1 else 'Perderá'

        # Descargar el modelo de clasificación para evitar mantenerlo en memoria
        del gb_modelado

        return render_template('result_classification.html', match_winner=match_winner)

if __name__ == '__main__':
    app.run(debug=True)
