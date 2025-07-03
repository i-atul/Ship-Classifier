import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_index_get(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"prediction" in response.data or b"Predict" in response.data

def test_index_post_no_file(client):
    response = client.post('/', data={})
    assert response.status_code == 200
    assert b"No file part" in response.data or b"No selected file" in response.data

# More tests can be added for file upload and prediction if PredictionPipeline.
