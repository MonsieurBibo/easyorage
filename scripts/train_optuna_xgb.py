# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "numpy<2",
#     "scikit-learn",
#     "xgboost",
#     "optuna",
#     "joblib",
#     "pyarrow==23.0.1",
# ]
# ///
"""
Optuna XGBoost unifié — script standalone avec persistence SQLite.

Résumable : relancer le script reprend l'étude depuis le dernier trial.
Résultat : models/xgb_best.joblib + models/optuna_xgb.db
"""

import json
import pathlib
import numpy as np
import polars as pl
import xgboost as xgb
import optuna
import joblib
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT  = pathlib.Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
MODEL = ROOT / "models"
MODEL.mkdir(exist_ok=True)

N_TRIALS = 40
AIRPORTS = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]

# ── Chargement données ────────────────────────────────────────────────────────
meta = json.loads((PROC / "feature_cols.json").read_text())
FEATURE_COLS = meta["feature_cols"]
TARGET_COL   = meta["target_col"]

splits = {}
for ap in AIRPORTS:
    splits[ap] = {
        "train": pl.read_parquet(str(PROC / f"{ap}_train.parquet")),
        "eval":  pl.read_parquet(str(PROC / f"{ap}_eval.parquet")),
    }

X_tr = np.vstack([splits[ap]["train"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
                  for ap in AIRPORTS])
y_tr = np.concatenate([splits[ap]["train"][TARGET_COL].cast(pl.Int8).to_numpy()
                       for ap in AIRPORTS])

n_pos = y_tr.sum()
n_neg = len(y_tr) - n_pos
pos_weight = n_neg / max(n_pos, 1)
print(f"Dataset: {len(X_tr):,} éclairs · {len(FEATURE_COLS)} features · pos_weight={pos_weight:.1f}")

# ── Objective Optuna ──────────────────────────────────────────────────────────
def objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 600),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        gamma            = trial.suggest_float("gamma", 0, 3),
        reg_alpha        = trial.suggest_float("reg_alpha", 0, 1),
        scale_pos_weight = pos_weight,
        verbosity=0, random_state=42,
    )
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr)

    all_true, all_proba = [], []
    for ap in AIRPORTS:
        Xev = splits[ap]["eval"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
        yev = splits[ap]["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        all_true.extend(yev.tolist())
        all_proba.extend(m.predict_proba(Xev)[:, 1].tolist())

    return roc_auc_score(np.array(all_true), np.array(all_proba))

# ── Étude Optuna avec persistence SQLite ─────────────────────────────────────
storage = f"sqlite:///{MODEL}/optuna_xgb.db"
study = optuna.create_study(
    study_name="xgb_unified",
    storage=storage,
    direction="maximize",
    load_if_exists=True,
)

n_done = len(study.trials)
n_remaining = max(0, N_TRIALS - n_done)
print(f"Trials déjà complétés : {n_done} / {N_TRIALS}")

if n_remaining == 0:
    print("Étude déjà complète.")
else:
    print(f"Lancement de {n_remaining} trial(s)...")
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)

# ── Résultats ─────────────────────────────────────────────────────────────────
print(f"\n=== Meilleur trial ===")
print(f"AUC = {study.best_value:.4f}")
print(f"Params : {study.best_params}")

# ── Entraîne modèle final avec les meilleurs params ───────────────────────────
best_model = xgb.XGBClassifier(
    **study.best_params,
    scale_pos_weight=pos_weight,
    verbosity=0, random_state=42,
)
best_model.fit(X_tr, y_tr)

print("\n=== AUC par aéroport (best model) ===")
all_true, all_proba = [], []
for ap in AIRPORTS:
    Xev = splits[ap]["eval"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    yev = splits[ap]["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
    auc = roc_auc_score(yev, best_model.predict_proba(Xev)[:, 1])
    all_true.extend(yev.tolist()); all_proba.extend(best_model.predict_proba(Xev)[:, 1].tolist())
    print(f"  {ap:<12} AUC={auc:.4f}")

mean_auc = roc_auc_score(np.array(all_true), np.array(all_proba))
print(f"  {'MEAN':<12} AUC={mean_auc:.4f}")

joblib.dump(best_model, MODEL / "xgb_best.joblib")
print(f"\nModèle sauvegardé → models/xgb_best.joblib")
print(f"Étude SQLite       → models/optuna_xgb.db (résumable)")
