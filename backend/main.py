from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services import data_loader
from backend.routers import airports, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    await data_loader.initialize()
    yield


app = FastAPI(
    title="EasyOrage API",
    version="0.1.0",
    description="Prédiction de fin d'alerte orage — DataBattle 2026",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(airports.router)
app.include_router(ws.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
