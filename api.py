#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flask
from flask import Flask, jsonify, request, render_template
import joblib
import pickle
import pandas as pd
import shap
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = False


# In[ ]:


# Suppression warnings
import warnings
warnings.filterwarnings('ignore')
#Chargement du tableau et du modèle

df = pd.read_csv("data/X_sample.csv")
print(df.head())
app = Flask(__name__)
LGBMClassifier = pickle.load(open("modele/LGBMClassifier.pkl", "rb"))


df.drop(columns='TARGET', inplace=True)
num_client = df.index.unique()



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/')
def predict():
    """

    Returns
    liste des clients dans le fichier

    """
    return jsonify({"model": "'LGBMClassifier",
                    "list_client_id" : list(num_client.astype(str))})


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
    return jsonify({ 'retour_prediction' : str(predict), 'predict_proba_0': predict_proba_0,
                     'predict_proba_1': predict_proba_1 })

@app.route('/details/<int:sk_id>')
def client_details(sk_id):
    """
    Parameters
    ----------
    sk_id : numero de client

    Returns
    -------
    Détails des valeurs pour le client spécifié et moyennes de toutes les variables pour tous les clients.
    """

    if sk_id in num_client:
        # Récupérer les détails du client
        client_data = df.loc[sk_id].to_dict()

        # Récupérer les moyennes pour toutes les variables
        mean_values = df.mean().to_dict()

        return jsonify({
            "client_data": client_data,
            "mean_values": mean_values
        })

    else:
        return jsonify({"error": "client inconnu"}), 404


if __name__ == '__main__':
    app.run()


# In[ ]:




