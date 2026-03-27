# 🏆 Data Battle IA PAU 2026 – EasyOrage

## 👥 Équipe
- Nom de l'équipe : **EasyOrage**
- Membres :
  - Selim MOHAMED
  - Lilian PICHARD
  - Louis BESLE

## 🎯 Problématique

Les aéroports interrompent leurs opérations au sol dès qu'un orage est détecté à moins de 3 km. La fin d'alerte est déclenchée **30 minutes après le dernier éclair** — une règle conservatrice qui immobilise inutilement les pistes.

L'objectif est de **prédire en temps réel la fin d'une alerte orage** (dernier éclair CG) autour de 5 aéroports (Ajaccio, Bastia, Biarritz, Nantes, Pise), en utilisant uniquement les données foudre Météorage, sans radar ni satellite.

Métrique du challenge :
- **Gain G** : secondes gagnées par rapport à la baseline 30 min, sommées sur toutes les alertes
- **Risk R** : fraction d'éclairs dangereux (dist < 3 km) manqués après la fin prédite — doit rester **< 2 %**

## 💡 Solution proposée

### Modèle

XGBoost entraîné sur **106 features** (foudre, terrain SRTM/DEM, météo ERA5), optimisé directement sur le Gain G via **Optuna** (200 trials, ~30 min). Cible binaire : `is_last_lightning_cloud_ground` — ce flash est-il le dernier de l'alerte ?

### Stratégie de déclenchement temps réel — AorJ + fallback silence

À chaque nouveau flash, deux règles déclenchent une prédiction "fin d'alerte" :

- **Règle A (K-consécutifs)** : K flashs consécutifs avec score ≥ threshold → déclenche tôt, haute confiance
- **Règle J (ILI-conditionné)** : score ≥ 0.60 ET inter-flash interval ≥ P75 causal de l'alerte en cours → détecte les orages qui s'affaiblissent
- **Fallback silence** : si aucune règle ne s'est déclenchée, on émet une prédiction après 15 min de silence (≥ P99 ILI sur train)

### Résultats officiels

| Métrique | Eval local (2016–2022) | Dataset test officiel (2023–2025) |
|---|---|---|
| **Gain G** | 177 h | **212 h** |
| **Risk R** | 0.016 ✓ | 0.008 ✓ |
| Couverture alertes | 49 % | 35 % |
| AUC | 0.925 | 0.898 |

### Features clés (par importance)

1. `fr_log_slope` — pente log du flash rate (indicateur de déclin, Schultz 2009)
2. `rolling_ili_5` / `rolling_ili_3` — inter-éclair interval glissant récent
3. `flash_rate_3` — fréquence des éclairs très récente
4. `centroid_dist_5` — distance aéroport → centroïde des 5 derniers éclairs

### Dashboard temps réel

Application web (React + FastAPI) simulant une session d'alerte en direct : replay d'alertes historiques avec visualisation des scores de prédiction éclair par éclair, carte interactive, et graphique d'évolution de la probabilité de fin d'alerte.

## ⚙️ Stack technique

- **Langages** : Python 3.12, TypeScript
- **ML** : XGBoost, Optuna, scikit-learn
- **Data** : Polars, Pandas, NumPy, PyArrow
- **Terrain / Météo** : SRTM (élévation), ERA5 via Open-Meteo
- **Backend** : FastAPI + WebSocket (replay temps réel)
- **Frontend** : React + Vite, TanStack Router, Recharts, Tailwind CSS + shadcn/ui
- **Outils** : uv (gestion dépendances Python), joblib (sérialisation modèle)

## 🚀 Installation & exécution

### Prérequis

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) : `pip install uv`
- Node.js 18+

### Installation

```bash
# Cloner le repo
git clone https://github.com/MonsieurBibo/easyorage.git
cd easyorage

# Backend — installer les dépendances
uv sync --project backend

# Frontend — installer les dépendances
cd frontend && npm install && cd ..
```

### Données

Placer les fichiers CSV bruts Météorage dans `data/raw/` :

```
data/raw/Ajaccio.csv
data/raw/Bastia.csv
data/raw/Biarritz.csv
data/raw/Nantes.csv
data/raw/Pise.csv
```

Pour le feature engineering terrain (SRTM) et météo (ERA5), les scripts de téléchargement sont fournis :

```bash
uv run --script scripts/fetch_terrain.py   # SRTM par aéroport
uv run --script scripts/fetch_weather.py   # ERA5 Open-Meteo
```

### Entraînement du modèle

```bash
# 1. Feature engineering (génère data/processed/*.parquet)
uv run --script scripts/compute_features.py

# 2. Entraînement XGBoost + Optuna (~30 min, 200 trials)
#    Génère models/xgb_best.joblib et models/predict_params.joblib
uv run --script scripts/train_optuna_gain.py

# 3. Générer les prédictions sur le dataset test
uv run --script scripts/generate_predictions.py
```

### Lancement du dashboard temps réel

```bash
# Terminal 1 — Backend API (depuis la racine du projet)
uv run --project backend uvicorn backend.main:app --reload

# Terminal 2 — Frontend
cp frontend/.env.example frontend/.env.local
cd frontend && npm run dev
```

Dashboard accessible sur **[http://localhost:5173](http://localhost:5173)**

> **Note** : le backend charge automatiquement `models/xgb_best.joblib` au démarrage. Les modèles pré-entraînés ne sont pas inclus dans le repo (fichiers binaires) — il faut exécuter l'étape d'entraînement au préalable.

## 🔗 Liens

- **Support de présentation** : *(à ajouter)*
