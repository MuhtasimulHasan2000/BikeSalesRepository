import pandas as pd
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# Load the KNN model
with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        brand = request.form['Brand']
        manufactured_year = float(request.form['Manufactured_year'])
        speed_km = float(request.form['Speed(km)'])
        engine_cc = float(request.form['Engine(cc)'])

        label_enc = LabelEncoder()
        brand_cat = label_enc.fit_transform([brand])[0]  # Transform and get the first (and only) element

        input_data = np.array([brand_cat, manufactured_year, speed_km, engine_cc]).reshape(1, -1)
        predicted_price = knn_model.predict(input_data)

        response = {'predicted_price': int(np.round(predicted_price[0], 2))}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
