"""FastAPI endpoint tests."""
import pytest, numpy as np
from fastapi.testclient import TestClient

FEATS = [200.0,95.0,135.0,3.0,4.0,3.5,0.32,0.44,0.21,0.30,
         0.47,0.67,0.37,25.0,0.72,0.37,62.0,18.0,12.0]

@pytest.fixture(scope="module")
def client():
    from api.app import app
    return TestClient(app)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid(client):
    r = client.post("/predict/features", json={"features":FEATS,"sex":"female","pregnant":True,"trimester":2})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        d = r.json()
        assert d["severity"] in ("normal","mild","moderate","severe","life_threatening")

def test_predict_bad_sex(client):
    assert client.post("/predict/features", json={"features":FEATS,"sex":"alien","pregnant":False}).status_code == 422

def test_predict_bad_length(client):
    assert client.post("/predict/features", json={"features":[1.0]*18,"sex":"female","pregnant":False}).status_code == 422
