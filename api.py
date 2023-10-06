#!/usr/bin/env python
# coding: utf-8

import flask
from flask import Flask, jsonify, request, render_template
import joblib
import pickle
import pandas as pd
import shap
import json
import warnings

# Suppression warnings
warnings.filterwarnings('ignore')

# Initialiser l'application Flask
app = Flask(__name__)
app.config["DEBUG"] = False

# Charger les données et le modèle
df = pd.read_csv("data/X_sample.csv")
df.set_index('SK_ID_CURR', inplace=True)
print(df.head())
LGBMClassifier = pickle.load(open("modele/LGBMClassifier.pkl", "rb"))
df.drop(columns='TARGET', inplace=True)
num_client = df.index.unique()
print(num_client[:10])
@app.route('/')
def home():
    return "api"

@app.route('/predict')
def predict():
    """
    Returns
    liste des clients dans le fichier
    """
    return jsonify({
        "model": "LGBMClassifier",
        "list_client_id" : list(num_client.astype(str))
    })

@app.route('/predict/<int:sk_id>',methods=['GET'])
def predict_get(sk_id):
    """
    Parameters
    ----------
    sk_id : numero de client

    Returns
    -------
    prediction  0 pour paiement OK
                1 pour defaut de paiement
    """
    if sk_id in num_client:
        predict = LGBMClassifier.predict(df.loc[sk_id].values.reshape(1,-1))
        predict_proba = LGBMClassifier.predict_proba(df.loc[sk_id].values.reshape(1,-1))
        predict_proba_0 = str(predict_proba[0][0])
        predict_proba_1 = str(predict_proba[0][1])
    else:
        predict = predict_proba_0 = predict_proba_1 = "client inconnu"
    return jsonify({
        'retour_prediction' : str(predict),
        'predict_proba_0': predict_proba_0,
        'predict_proba_1': predict_proba_1
    })
# ... (reste du code)

print(num_client[:10])  # printez les 10 premiers ID pour vérifier

@app.route('/details/<int:sk_id>')
def client_details(sk_id):
    print(f"Received sk_id: {sk_id}")  # Affichez le sk_id reçu
    print(f"Type of sk_id: {type(sk_id)}")  # Affichez le type de sk_id
    print(f"First 10 clients in num_client: {num_client[:10]}")  # Affichez les 10 premiers clients
    print(f"Type of first client in num_client: {type(num_client[0])}")  # Affichez le type du premier client

    if sk_id in num_client:
        client_data = df.loc[sk_id].to_dict()
        mean_values = df.mean().to_dict()
        return jsonify({
            "client_data": client_data,
            "mean_values": mean_values
        })
    else:
        print(f"sk_id {sk_id} not found in num_client")
        return jsonify({"error": "client inconnu"}), 404


if __name__ == '__main__':
    app.run()



