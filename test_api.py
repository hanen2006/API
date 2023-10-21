#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pytest

from api import app  
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data.decode("utf-8") == "api"

def test_predict(client):
    response = client.get('/predict')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'model' in json_data
    assert json_data['model'] == 'LGBMClassifier'
    assert 'list_client_id' in json_data

def test_predict_valid_sk_id(client):
    response = client.get('/predict/146052')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'retour_prediction' in json_data

def test_predict_invalid_sk_id(client):
    response = client.get('/predict/999999999')
    assert response.status_code == 200  # À ajuster si vous changez le code pour retourner une autre réponse.
    json_data = response.get_json()
    assert 'retour_prediction' in json_data
    assert json_data['retour_prediction'] == 'client inconnu'

def test_details_valid_sk_id(client):
    response = client.get('/details/146052')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'client_data' in json_data
    assert 'mean_values' in json_data

def test_details_invalid_sk_id(client):
    response = client.get('/details/999999999')
    assert response.status_code == 404



# In[ ]:



