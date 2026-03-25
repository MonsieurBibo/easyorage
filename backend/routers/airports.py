from fastapi import APIRouter, HTTPException
from backend.config import AIRPORTS
from backend.services import data_loader

router = APIRouter(prefix="/airports", tags=["airports"])


@router.get("")
def list_airports():
    return data_loader.get_airports_list()


@router.get("/{airport}/alerts")
def list_alerts(airport: str):
    if airport.lower() not in AIRPORTS:
        raise HTTPException(status_code=404, detail=f"Aéroport '{airport}' inconnu")
    return data_loader.get_alerts(airport)


@router.get("/{airport}/alerts/{alert_id}")
def get_alert(airport: str, alert_id: str):
    if airport.lower() not in AIRPORTS:
        raise HTTPException(status_code=404, detail=f"Aéroport '{airport}' inconnu")
    alert = data_loader.get_alert(airport, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alerte '{alert_id}' introuvable pour '{airport}'")
    return alert


@router.get("/{airport}/stats")
def get_stats(airport: str):
    if airport.lower() not in AIRPORTS:
        raise HTTPException(status_code=404, detail=f"Aéroport '{airport}' inconnu")
    return data_loader.get_stats(airport)
