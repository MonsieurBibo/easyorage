"""
replay_engine.py
----------------
Gère une session de replay par connexion WebSocket.

Le client contrôle la vitesse via { action: "set_speed", speed: X }.
X est un multiplicateur : 1.0 = temps réel, 10.0 = 10× plus rapide.

L'interruptible_sleep recalcule le délai restant à chaque tick (50ms),
ce qui rend le changement de vitesse quasi-instantané.
"""

import asyncio
from datetime import datetime

from fastapi import WebSocket


class ReplaySession:
    def __init__(self, websocket: WebSocket, alert: dict):
        self.websocket = websocket
        self.alert = alert
        self.speed: float = 1.0
        self._task: asyncio.Task | None = None

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, min(float(speed), 200.0))

    async def _sleep(self, sim_seconds: float) -> None:
        """Dort sim_seconds de temps simulé. Réagit aux changements de vitesse."""
        if sim_seconds <= 0:
            return
        elapsed = 0.0
        tick = 0.05  # 50 ms
        while elapsed < sim_seconds / self.speed:
            chunk = min(tick, sim_seconds / self.speed - elapsed)
            await asyncio.sleep(chunk)
            elapsed += chunk
            # Recalcule la durée cible si speed a changé
            # (la boucle while se réévalue automatiquement)

    async def _run(self) -> None:
        flashes = self.alert["flashes"]

        await self.websocket.send_json({
            "type": "subscribed",
            "airport": self.alert["airport"].lower(),
            "alert_id": self.alert["alert_id"],
            "n_flashes": self.alert["n_flashes"],
            "duration_s": self.alert["duration_s"],
            "start_date": self.alert["start_date"],
            "end_date": self.alert["end_date"],
        })

        prev_date: datetime | None = None

        for flash in flashes:
            curr_date = datetime.fromisoformat(flash["date"])

            if prev_date is not None:
                gap_s = (curr_date - prev_date).total_seconds()
                await self._sleep(gap_s)

            prev_date = curr_date

            await self.websocket.send_json({"type": "flash", "data": flash})

            if flash["prediction_triggered"] and self.alert["prediction"]:
                await self.websocket.send_json({
                    "type": "prediction_triggered",
                    "data": self.alert["prediction"],
                })

        await self.websocket.send_json({
            "type": "alert_end",
            "data": {
                "n_flashes": self.alert["n_flashes"],
                "end_date": self.alert["end_date"],
                "prediction": self.alert["prediction"],
            },
        })

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run())
        return self._task

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
