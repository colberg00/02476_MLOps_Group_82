"""Tests for the FastAPI application."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from mlops_course_project.api import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_missing_input(client):
    """Test prediction fails when no slug or url provided."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "Provide either" in response.json()["detail"]


def test_predict_invalid_url(client):
    """Test prediction fails with invalid URL that can't be parsed."""
    response = client.post("/predict", json={"url": "https://example.com/"})
    assert response.status_code == 400
    assert "Could not extract" in response.json()["detail"]


@patch("mlops_course_project.api.get_model")
def test_predict_with_slug(mock_get_model, client):
    """Test prediction with a slug."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    mock_get_model.return_value = mock_model

    response = client.post("/predict", json={"slug": "trump announces new policy"})
    assert response.status_code == 200
    data = response.json()
    assert data["slug"] == "trump announces new policy"
    assert data["prediction"] == "fox"
    assert data["proba_nbc"] == pytest.approx(0.3)
    assert data["proba_fox"] == pytest.approx(0.7)


@patch("mlops_course_project.api.get_model")
def test_predict_with_url(mock_get_model, client):
    """Test prediction with a valid URL."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.8, 0.2]]
    mock_get_model.return_value = mock_model

    response = client.post("/predict", json={"url": "https://www.nbcnews.com/politics/trump-announces-new-policy"})
    assert response.status_code == 200
    data = response.json()
    assert "trump" in data["slug"]
    assert data["prediction"] == "nbc"


@patch("mlops_course_project.api.get_model")
def test_predict_model_not_found(mock_get_model, client):
    """Test prediction fails gracefully when model is not found."""
    mock_get_model.side_effect = FileNotFoundError("Model not found")

    response = client.post("/predict", json={"slug": "some headline"})
    assert response.status_code == 503
    assert "Model not found" in response.json()["detail"]
