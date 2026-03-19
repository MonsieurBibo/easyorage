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
#     "pandas",
# ]
# ///
"""
Optuna XGBoost — optimisation directe sur Gain G (métrique officielle).

Objectif : maximiser le Gain G sur l'eval split, sous contrainte Risk R < 0.02.
Contrairement à train_optuna_xgb.py (optimise AUC), ici on optimise directement
la métrique du concours : secondes gagnées vs baseline 30 min.

Tune aussi K et base_threshold (stratégie K-consécutifs).

Résultat : models/xgb_best.joblib + models/optuna_gain.db
"""

import json
import pathlib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import optuna
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT  = pathlib.Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
MODEL = ROOT / "models"
MODEL.mkdir(exist_ok=True)

N_TRIALS    = 60
R_ACCEPT    = 0.02
MAX_GAP_MIN = 30
MIN_DIST_KM = 3
AIRPORTS    = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]

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

# ── Eval raw pour les métriques officielles ───────────────────────────────────
eval_feat_all = pl.concat([splits[ap]["eval"] for ap in AIRPORTS])
eval_raw = eval_feat_all.select(["airport", "airport_alert_id", "date", "dist"]).to_pandas()
eval_raw = eval_raw[eval_raw["airport_alert_id"].notna()]
eval_raw["date"] = pd.to_datetime(eval_raw["date"], utc=True)

tot_dangerous = len(eval_raw[eval_raw["dist"] < MIN_DIST_KM])
alerts_grouped = eval_raw.groupby(["airport", "airport_alert_id"])

THETAS = np.round(np.linspace(0.05, 0.95, 19), 2).tolist()


def make_predictions_eval(model, df_feat: pl.DataFrame, k: int, base_threshold: float) -> pd.DataFrame:
    """Stratégie K-consécutifs sur l'eval split."""
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    X = df_alerts.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = model.predict_proba(X)[:, 1]

    df_pd = df_alerts.select(["airport", "airport_alert_id", "date"]).to_pandas()
    df_pd["score"] = scores
    df_pd["date"] = pd.to_datetime(df_pd["date"], utc=True)

    rows = []
    for (airport, alert_id), grp in df_pd.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s = grp["score"].to_numpy()
        alert_has_pred = False
        for i in range(len(s)):
            if i < k - 1:
                continue
            window = s[i - k + 1 : i + 1]
            if (window >= base_threshold).all():
                rows.append({
                    "airport": airport,
                    "airport_alert_id": alert_id,
                    "prediction_date": grp["date"].iloc[i],
                    "predicted_date_end_alert": grp["date"].iloc[i],
                    "confidence": float(window.min()),
                })
                alert_has_pred = True
        if not alert_has_pred:
            last = grp.iloc[-1]
            rows.append({
                "airport": airport,
                "airport_alert_id": alert_id,
                "prediction_date": last["date"],
                "predicted_date_end_alert": last["date"],
                "confidence": base_threshold,
            })

    return pd.DataFrame(rows, columns=[
        "airport", "airport_alert_id", "prediction_date",
        "predicted_date_end_alert", "confidence"
    ])


def best_gain_under_risk(preds_df: pd.DataFrame, base_threshold: float = 0.0) -> float:
    """Retourne le meilleur Gain (en heures) parmi les θ qui respectent R < R_ACCEPT.
    Évalue uniquement θ > base_threshold pour exclure les fallbacks (confidence=thr)
    et forcer le modèle à trouver de vrais K-consécutifs utiles.
    """
    preds_df = preds_df.copy()
    preds_df["predicted_date_end_alert"] = pd.to_datetime(
        preds_df["predicted_date_end_alert"], utc=True
    )
    best = 0.0
    for theta in [t for t in THETAS if t > base_threshold]:
        over = preds_df[preds_df["confidence"] >= theta]
        if over.empty:
            continue
        pred_min = over.groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"].min()
        gain, missed = 0, 0
        for (airport, alert_id), end_pred in pred_min.items():
            try:
                lightnings = alerts_grouped.get_group((airport, alert_id))
            except KeyError:
                continue
            end_baseline = pd.to_datetime(lightnings["date"], utc=True).max() + pd.Timedelta(minutes=MAX_GAP_MIN)
            gain += (end_baseline - end_pred).total_seconds()
            close = pd.to_datetime(lightnings[lightnings["dist"] < MIN_DIST_KM]["date"], utc=True)
            missed += int((close > end_pred).sum())
        r = missed / max(tot_dangerous, 1)
        if r < R_ACCEPT:
            best = max(best, gain / 3600)
    return best


# ── Objective Optuna ──────────────────────────────────────────────────────────
def objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 700),
        max_depth        = trial.suggest_int("max_depth", 3, 7),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        gamma            = trial.suggest_float("gamma", 0, 3),
        reg_alpha        = trial.suggest_float("reg_alpha", 0, 1),
        scale_pos_weight = pos_weight,
        verbosity=0, random_state=42,
    )
    k               = trial.suggest_int("k", 2, 5)
    base_threshold  = trial.suggest_float("base_threshold", 0.15, 0.5)

    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr)

    preds = make_predictions_eval(m, eval_feat_all, k, base_threshold)
    return best_gain_under_risk(preds, base_threshold)


# ── Étude Optuna ──────────────────────────────────────────────────────────────
storage = f"sqlite:///{MODEL}/optuna_gain.db"
study = optuna.create_study(
    study_name="xgb_gain",
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
print(f"Gain = {study.best_value:.1f}h")
print(f"Params : {study.best_params}")

best_p = study.best_params
k_best             = best_p.pop("k")
base_threshold_best = best_p.pop("base_threshold")
print(f"K={k_best}, base_threshold={base_threshold_best:.2f}")

# ── Entraîne modèle final ─────────────────────────────────────────────────────
best_model = xgb.XGBClassifier(
    **best_p,
    scale_pos_weight=pos_weight,
    verbosity=0, random_state=42,
)
best_model.fit(X_tr, y_tr)

joblib.dump(best_model, MODEL / "xgb_best.joblib")
joblib.dump({"k": k_best, "base_threshold": base_threshold_best},
            MODEL / "predict_params.joblib")

print(f"\nModèle sauvegardé → models/xgb_best.joblib")
print(f"Params prédiction → models/predict_params.joblib")
print(f"Étude SQLite       → models/optuna_gain.db (résumable)")
