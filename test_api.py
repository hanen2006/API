#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pytest-flask')


# In[1]:


import pytest
from api import app as flask_app
import json

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_home(client):
    rv = client.get('/')
    assert rv.status_code == 200

def test_predict(client):
    rv = client.get('/predict/')
    data = json.loads(rv.data)
    assert 'list_client_id' in data
    assert 'model' in data
    assert data['model'] == 'LGBMClassifier'

def test_predict_get(client):
    rv = client.get('/predict/100003')
    data = json.loads(rv.data)
    assert 'retour_prediction' in data
    assert 'predict_proba_0' in data
    assert 'predict_proba_1' in data


# In[ ]:




