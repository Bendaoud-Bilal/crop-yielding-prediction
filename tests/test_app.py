import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app


def _get_sample_payload(client: TestClient) -> dict:
    artifacts = getattr(client.app.state, "artifacts", None)
    if not artifacts:
        pytest.skip("Model artifacts are unavailable; cannot run integration tests.")

    unique_values = artifacts["unique_values"]
    crop = unique_values["crops"][0]
    state = unique_values["states"][0]
    season = unique_values["seasons"][0]

    return {
        "crop": crop,
        "state": state,
        "season": season,
        # Optional inputs intentionally omitted to rely on median fallback
    }


def test_home_page_renders():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Crop Yield Prediction" in response.text


def test_predict_route_returns_result():
    client = TestClient(app)
    payload = _get_sample_payload(client)

    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert "Predicted Yield" in response.text


def test_crop_recommendation_form_renders():
    client = TestClient(app)
    assets = getattr(client.app.state, "crop_recommendation", None)
    if not assets:
        pytest.skip("Crop recommendation assets not available; skipping test.")

    response = client.get("/crop-recommendation")
    assert response.status_code == 200
    assert "Crop Recommendation" in response.text


def test_crop_recommendation_prediction():
    client = TestClient(app)
    assets = getattr(client.app.state, "crop_recommendation", None)
    if not assets:
        pytest.skip("Crop recommendation assets not available; skipping test.")

    payload = {
        "nitrogen": "90",
        "phosphorus": "42",
        "potassium": "43",
        "temperature": "23",
        "humidity": "80",
        "ph": "6.5",
        "rainfall": "200",
    }
    response = client.post("/crop-recommendation/predict", data=payload)
    assert response.status_code == 200
    assert "Recommended Crop" in response.text
