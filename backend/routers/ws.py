"""
ws.py
-----
WebSocket global /ws.

Protocol client → serveur :
  { "action": "subscribe", "airport": "bastia" }
  { "action": "set_speed", "speed": 10.0 }
  { "action": "unsubscribe" }

Protocol serveur → client :
  { "type": "subscribed",          ...alert metadata }
  { "type": "flash",               "data": {...flash + score} }
  { "type": "prediction_triggered","data": {...prediction} }
  { "type": "alert_end",           "data": {...} }
  { "type": "error",               "message": "..." }
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.data_loader import get_alert, get_default_alert
from backend.services.replay_engine import ReplaySession

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session: ReplaySession | None = None

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "subscribe":
                airport = str(data.get("airport", "")).lower()
                alert_id = data.get("alert_id")

                if session:
                    session.cancel()
                    session = None

                if alert_id:
                    alert = get_alert(airport, str(alert_id))
                    if not alert:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Alerte '{alert_id}' introuvable pour '{airport}'",
                        })
                        continue
                else:
                    alert = get_default_alert(airport)
                    if not alert:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Aucune alerte disponible pour '{airport}'",
                        })
                        continue

                session = ReplaySession(websocket, alert)
                session.start()

            elif action == "set_speed":
                speed = data.get("speed", 1.0)
                if session:
                    session.set_speed(speed)

            elif action == "unsubscribe":
                if session:
                    session.cancel()
                    session = None

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Action inconnue : '{action}'",
                })

    except WebSocketDisconnect:
        pass
    finally:
        if session:
            session.cancel()
