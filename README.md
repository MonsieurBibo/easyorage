# 🏆 Data Battle IA PAU 2026 – EasyOrage

## 👥 Équipe
- Nom de l'équipe : **EasyOrage**
- Membres :
  - Selim MOHAMED
  - Lilian PICHARD
  - Louis BESLE

## 🎯 Problématique

Les aéroports doivent interrompre leurs opérations au sol dès qu'un orage menace (éclair à moins de 3 km). La fin d'alerte est actuellement déclenchée 30 minutes après le dernier éclair détecté — une règle conservatrice qui immobilise inutilement les pistes.

L'objectif est de **prédire en temps réel la fin d'une alerte orage** (dernier éclair CG) autour de 5 aéroports (Ajaccio, Bastia, Biarritz, Nantes, Pise), en n'utilisant que les données foudre Météorage, sans radar ni satellite.

La métrique du challenge :
- **Gain G** : secondes gagnées vs la baseline des 30 min, sommées sur toutes les alertes
- **Risk R** : fraction d'éclairs dangereux (dist < 3 km) manqués après la fin d'alerte prédite — doit rester **< 2 %**

## 💡 Solution proposée

### Modèle
XGBoost entraîné sur **106 features** (foudre, terrain SRTM/DEM, météo ERA5), optimisé directement sur le Gain G via **Optuna** (pas sur l'AUC). Cible binaire : `is_last_lightning_cloud_ground` (ce flash est-il le dernier de l'alerte ?).

### Stratégie de déclenchement temps réel — AorJ + silence fallback
À chaque nouveau flash, deux règles déclenchent une prédiction "fin d'alerte" :

- **Règle A (K-consécutifs)** : K=2 flashs consécutifs avec score ≥ threshold
- **Règle J (ILI-conditionné)** : score ≥ 0.60 ET inter-flash interval ≥ P75 causal de l'alerte en cours (grand silence = storm qui s'affaiblit)
- **Fallback silence** : si aucune règle ne s'est déclenchée, on émet une prédiction après 15 min de silence (≥ P99 ILI sur le train)

### Résultats
| Métrique | Eval local (2016–2022) | Dataset test officiel (2023–2025) |
|---|---|---|
| Gain G | 177 h | **212 h** |
| Risk R | 0.016 ✓ | 0.008 ✓ |
| Couverture alertes | 49 % | 35 % |
| AUC | 0.9246 | 0.8981 |

### Features clés (top par importance)
1. `fr_log_slope` — pente log du flash rate (indicateur de déclin, Schultz 2009)
2. `rolling_ili_5` / `rolling_ili_3` — inter-éclair interval glissant
3. `flash_rate_3` — fréquence très récente
4. `centroid_dist_5` — distance aéroport → centroïde des 5 derniers éclairs (orage qui s'éloigne)

## ⚙️ Stack technique
- **Langages** : Python 3.12
- **ML** : XGBoost, Optuna, scikit-learn
- **Data** : Polars, Pandas, NumPy, PyArrow
- **Terrain/Météo** : SRTM (elevation), ERA5 via Open-Meteo
- **Backend** : FastAPI
- **Frontend** : React + TypeScript + Vite, Recharts, Tailwind CSS
- **Outils** : uv (gestion deps), joblib (sérialisation modèle)
- **IA** : Claude Sonnet 4.6 (assistance développement)

## 🚀 Installation & exécution

### Prérequis
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- Node.js 18+ (pour le frontend)

### Installation

```bash
# Cloner le repo
git clone <repo-url>
cd easyorage

# Backend — installer les dépendances
uv sync --project backend

# Frontend — installer les dépendances
cd frontend && npm install && cd ..
```

### Données

Placer les fichiers de données dans :
```
data/raw/                    # fichiers CSV bruts Météorage
data/processed/              # générés automatiquement par les scripts
dataset_test/dataset_set.csv # dataset de test officiel
```

### Entraînement du modèle

```bash
# 1. Feature engineering
uv run --script scripts/compute_features.py

# 2. Entraînement XGBoost + Optuna (100 trials, ~30 min)
uv run --script scripts/train_optuna_gain.py

# 3. Générer les prédictions (dataset test)
uv run --script scripts/generate_predictions.py

# 4. Vérifier l'overfitting (optionnel)
uv run --script scripts/check_overfit.py
```

### Exécution (dashboard temps réel)

```bash
# Terminal 1 — Backend API
uv run --project backend uvicorn backend.main:app --reload

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Le dashboard est accessible sur [http://localhost:5173](http://localhost:5173).
