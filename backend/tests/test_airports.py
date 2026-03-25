"""
Tests des endpoints REST /airports.

On mocke entièrement data_loader pour ne pas dépendre du dataset réel.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

MOCK_ALERTS = [
    {
        "alert_id": "42",
        "airport": "Bastia",
        "start_date": "2023-06-01T10:00:00+00:00",
        "end_date": "2023-06-01T10:30:00+00:00",
        "n_flashes": 15,
        "duration_s": 1800.0,
        "prediction": {
            "triggered_at_rank": 12,
            "triggered_at_date": "2023-06-01T10:25:00+00:00",
            "confidence": 0.91,
        },
        "flashes": [
            {
                "rank": i,
                "date": f"2023-06-01T10:{i:02d}:00+00:00",
                "lat": 42.55,
                "lon": 9.48,
                "flash_type": "CG",
                "dist_km": 2.0,
                "amplitude": -30.0,
                "score": 0.6 + i * 0.02,
                "prediction_triggered": i == 12,
            }
            for i in range(1, 16)
        ],
    }
]

MOCK_STATS = {
    "airport": "bastia",
    "total_alerts": 1,
    "covered_alerts": 1,
    "coverage_rate": 1.0,
    "total_gain_h": 0.08,
    "risk": 0.0,
}


@pytest.fixture
def client():
    with (
        patch("backend.services.data_loader.initialize", return_value=None),
        patch("backend.services.data_loader._alerts_cache", {"bastia": MOCK_ALERTS}),
        patch("backend.services.data_loader.get_airports_list", return_value=[
            {"id": "bastia", "name": "Bastia", "lat": 42.5527, "lon": 9.4837}
        ]),
        patch("backend.services.data_loader.get_alerts", return_value=[
            {k: v for k, v in MOCK_ALERTS[0].items() if k != "flashes"}
        ]),
        patch("backend.services.data_loader.get_alert", side_effect=lambda ap, aid: (
            MOCK_ALERTS[0] if aid == "42" else None
        )),
        patch("backend.services.data_loader.get_stats", return_value=MOCK_STATS),
    ):
        from backend.main import app
        yield TestClient(app, raise_server_exceptions=True)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_airports(client):
    r = client.get("/airports")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert data[0]["id"] == "bastia"


def test_list_alerts(client):
    r = client.get("/airports/bastia/alerts")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert data[0]["alert_id"] == "42"
    assert "flashes" not in data[0]


def test_list_alerts_unknown_airport(client):
    r = client.get("/airports/unknown/alerts")
    assert r.status_code == 404


def test_get_alert(client):
    r = client.get("/airports/bastia/alerts/42")
    assert r.status_code == 200
    data = r.json()
    assert data["alert_id"] == "42"
    assert "flashes" in data
    assert len(data["flashes"]) == 15


def test_get_alert_not_found(client):
    r = client.get("/airports/bastia/alerts/999")
    assert r.status_code == 404


def test_get_alert_unknown_airport(client):
    r = client.get("/airports/unknown/alerts/42")
    assert r.status_code == 404


def test_get_stats(client):
    r = client.get("/airports/bastia/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_gain_h" in data
    assert "risk" in data
    assert data["covered_alerts"] == 1


def test_get_stats_unknown_airport(client):
    r = client.get("/airports/unknown/stats")
    assert r.status_code == 404
