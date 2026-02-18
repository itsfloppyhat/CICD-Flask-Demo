"""Tests for the Flask prediction API."""
import json
import pytest
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Health check should return 200 with status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    """Valid features should return a prediction with confidence."""
    response = client.post(
        "/predict",
        data=json.dumps({"features": [5.1, 3.5, 1.4, 0.2]}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["prediction"] in ["setosa", "versicolor", "virginica"]
    assert 0 <= data["confidence"] <= 1
    assert "probabilities" in data


def test_predict_missing_features(client):
    """Request without features should return 400."""
    response = client.post(
        "/predict",
        data=json.dumps({"wrong_key": [1, 2, 3, 4]}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_predict_wrong_feature_count(client):
    """Request with wrong number of features should return 400."""
    response = client.post(
        "/predict",
        data=json.dumps({"features": [1.0, 2.0]}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "Expected 4 features" in data["error"]


def test_predict_setosa(client):
    """Classic setosa measurements should predict setosa."""
    response = client.post(
        "/predict",
        data=json.dumps({"features": [5.0, 3.4, 1.5, 0.2]}),
        content_type="application/json",
    )
    data = json.loads(response.data)
    assert data["prediction"] == "setosa"


def test_predict_virginica(client):
    """Classic virginica measurements should predict virginica."""
    response = client.post(
        "/predict",
        data=json.dumps({"features": [6.7, 3.0, 5.2, 2.3]}),
        content_type="application/json",
    )
    data = json.loads(response.data)
    assert data["prediction"] == "virginica"