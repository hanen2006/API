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
import numpy as np


# In[ ]:


# Suppression warnings
import warnings
warnings.filterwarnings('ignore')
# Initialiser l'application Flask

app.config["DEBUG"] = False

#Chargement du tableau et du modèle

df = pd.read_csv("./data/X_sample.csv")
print(df.head())
app = Flask(__name__)
LGBMClassifier = pickle.load(open("./modele/LGBMClassifier.pkl", "rb"))


df.drop(columns='TARGET', inplace=True)
num_client = df.index.unique()
print(num_client[:10])

@app.route('/')
def home():
    return "api"

@app.route('/predict/')
def predict():
    return jsonify({"model": "'LGBMClassifier",
                    "list_client_id": list(num_client.astype(str))})

@app.route('/all_scores')
def all_scores():
    scores = LGBMClassifier.predict_proba(df)[:, 1] * 100
    return jsonify(scores.tolist())

@app.route('/predict/<int:sk_id>',methods=['GET'])
def predict_get(sk_id):

    if sk_id in num_client:
        predict = LGBMClassifier.predict(df.loc[sk_id].values.reshape(1,-1))
        predict_proba = LGBMClassifier.predict_proba(df.loc[sk_id].values.reshape(1,-1))
        predict_proba_0 = str(predict_proba[0][0])
        predict_proba_1 = str(predict_proba[0][1])
    else:
        predict = predict_proba_0 = predict_proba_1 = "client inconnu"
    
    # Ici, nous obtenons le premier élément de 'predict' et le convertissons en chaîne.
    return jsonify({ 'retour_prediction': str(predict[0]), 'predict_proba_0': predict_proba_0,
                     'predict_proba_1': predict_proba_1 })


@app.route('/details/')
def all_clients_details():
    selected_columns = ['CREDIT_TERM', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'PREV_APPL_MEAN_CNT_PAYMENT', 
                        'ANNUITY_INCOME_PERCENT', 'AMT_CREDIT', 'DAYS_EMPLOYED']
    client_details = df[selected_columns].to_dict(orient='index')
    return jsonify(client_details)

@app.route('/details/<int:sk_id>')
def specific_client_details(sk_id):
  selected_columns = ['CREDIT_TERM', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'PREV_APPL_MEAN_CNT_PAYMENT', 
                        'ANNUITY_INCOME_PERCENT', 'AMT_CREDIT', 'DAYS_EMPLOYED']
  if sk_id in num_client:
        client_data = df[selected_columns].loc[sk_id].to_dict()
        mean_values = df[selected_columns].mean().to_dict()
        return jsonify({
            "client_data": client_data,
            "mean_values": mean_values
        })
 else:
        return jsonify({"error": "client inconnu"}), 404

@app.route('/shap_values/<int:sk_id>')
def get_shap_values(sk_id):
   if sk_id in num_client:
        subsampled_test_data = df.loc[sk_id].values.reshape(1, -1)
        explainer = shap.TreeExplainer(LGBMClassifier)
        shap_values = explainer.shap_values(subsampled_test_data)
        shap_values_list = shap_values[1][0].tolist()
        return jsonify({
            "shap_values": shap_values_list
        })
    else:
        return jsonify({"error": "client inconnu"}), 404

if __name__ == '__main__':
    app.run()


# In[ ]:
