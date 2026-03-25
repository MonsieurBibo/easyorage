# EasyOrage — DataBattle 2026 Météorage

Prédiction de fin d'alerte orage pour 5 aéroports, à partir de données foudre uniquement.

## Lancer le projet

**Backend**

```bash
uv sync --project backend
uv run --project backend uvicorn backend.main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

L'interface est disponible sur [http://localhost:5173](http://localhost:5173).
