# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "numpy<2",
#     "scikit-learn",
#     "xgboost",
#     "joblib",
#     "pyarrow==23.0.1",
#     "pandas",
# ]
# ///
"""
generate_predictions.py
-----------------------
Applique le modèle XGBoost sur le dataset test officiel et génère predictions.csv
au format attendu par le notebook d'évaluation.

Évalue aussi les métriques officielles (Risk R, Gain G) sur les données locales
(eval split) pour avoir un point de comparaison.

Format predictions.csv :
  airport, airport_alert_id, prediction_date, predicted_date_end_alert, confidence
"""

import sys
import json
import pathlib
import importlib.util

import numpy as np
import pandas as pd
import polars as pl
import joblib

ROOT  = pathlib.Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
MODEL = ROOT / "models"

# ── Import des fonctions de compute_features ────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "compute_features", ROOT / "scripts" / "compute_features.py"
)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

# ── Chargement modèle et features ────────────────────────────────────────────
model = joblib.load(MODEL / "xgb_best.joblib")
meta  = json.loads((PROC / "feature_cols.json").read_text())
FEATURE_COLS = meta["feature_cols"]
TARGET_COL   = meta["target_col"]

_pp = joblib.load(MODEL / "predict_params.joblib")
K_OPT   = _pp["k"]
THR_OPT = _pp["base_threshold"]
print(f"Paramètres prédiction : K={K_OPT}, threshold={THR_OPT:.3f}")

# ── Métriques officielles ─────────────────────────────────────────────────────
MAX_GAP_MIN = 30
MIN_DIST_KM = 3
R_ACCEPT    = 0.02

def compute_official_metrics(df_raw: pd.DataFrame, preds_df: pd.DataFrame):
    """
    Calcule Risk R et Gain G pour différents θ.
    df_raw   : données brutes avec colonnes date, dist, airport, airport_alert_id
    preds_df : predictions avec prediction_date, predicted_date_end_alert, confidence, airport, airport_alert_id
    """
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    preds_df = preds_df.copy()
    preds_df["predicted_date_end_alert"] = pd.to_datetime(
        preds_df["predicted_date_end_alert"], utc=True
    )

    tot_dangerous = len(df_raw[df_raw["dist"] < MIN_DIST_KM])
    alerts = df_raw.groupby(["airport", "airport_alert_id"])

    thetas = np.round(np.linspace(0.05, 0.95, 19), 2).tolist()
    results = {}

    for theta in thetas:
        over_theta = preds_df[preds_df["confidence"] >= theta]
        if over_theta.empty:
            results[theta] = (0, tot_dangerous)
            continue

        pred_min = (
            over_theta
            .groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"]
            .min()
        )

        gain, missed = 0, 0
        for (airport, alert_id), end_pred in pred_min.items():
            try:
                lightnings = alerts.get_group((airport, alert_id))
            except KeyError:
                continue
            end_baseline = pd.to_datetime(lightnings["date"], utc=True).max() + pd.Timedelta(minutes=MAX_GAP_MIN)
            gain += (end_baseline - end_pred).total_seconds()
            close = pd.to_datetime(lightnings[lightnings["dist"] < MIN_DIST_KM]["date"], utc=True)
            missed += int((close > end_pred).sum())

        results[theta] = (gain, missed)

    return results, tot_dangerous


def print_metrics(results: dict, tot_dangerous: int, label: str):
    print(f"\n=== {label} ===")
    print(f"{'θ':>6} {'Risk R':>8} {'Gain (h)':>10} {'OK?':>5}")
    print("─" * 35)

    best_gain, best_theta, best_risk = 0, None, None
    for theta, (gain, missed) in sorted(results.items()):
        r = missed / max(tot_dangerous, 1)
        ok = "✓" if r < R_ACCEPT else " "
        print(f"{theta:>6.2f} {r:>8.4f} {gain/3600:>10.1f}h {ok}")
        if r < R_ACCEPT and gain > best_gain:
            best_gain, best_theta, best_risk = gain, theta, r

    if best_theta is not None:
        print(f"\n→ θ optimal = {best_theta}  |  Gain = {best_gain/3600:.1f}h  |  Risk = {best_risk:.4f}")
    else:
        print("\n→ Aucun θ ne respecte R < 0.02")

    return best_theta, best_gain


def recompute_ili_percentiles(df_feat: pl.DataFrame) -> pl.DataFrame:
    """
    Recalcule ili_vs_p75 et ili_vs_p95 avec les percentiles issus du dataset
    courant (pas les valeurs hardcodées du train 2016-2022).

    Utile sur le dataset test (2023-2025) où la distribution ILI peut différer.
    """
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    airports = df_alerts["airport"].unique().to_list()

    p75_map, p95_map = {}, {}
    for ap in airports:
        ili_vals = (
            df_alerts
            .filter(pl.col("airport") == ap)
            ["ili_s"]
            .drop_nulls()
            .to_numpy()
        )
        if len(ili_vals) > 10:
            p75_map[ap] = float(np.percentile(ili_vals, 75))
            p95_map[ap] = float(np.percentile(ili_vals, 95))
        # Si trop peu de données, on garde les valeurs existantes

    print("  Percentiles ILI recalculés depuis le dataset courant :")
    for ap in sorted(p75_map):
        print(f"    {ap:<12} P75={p75_map[ap]:.0f}s  P95={p95_map[ap]:.0f}s")

    # Patch les colonnes sur tout le dataframe (alertes + hors-alerte)
    p75_expr = pl.lit(None, dtype=pl.Float64)
    p95_expr = pl.lit(None, dtype=pl.Float64)
    for ap, v75 in p75_map.items():
        p75_expr = pl.when(pl.col("airport") == ap).then(pl.lit(v75)).otherwise(p75_expr)
    for ap, v95 in p95_map.items():
        p95_expr = pl.when(pl.col("airport") == ap).then(pl.lit(v95)).otherwise(p95_expr)

    return df_feat.with_columns([
        (pl.col("rolling_ili_max_5") / (p95_expr + 1e-6)).alias("ili_vs_p95"),
        (pl.col("rolling_ili_max_5") / (p75_expr + 1e-6)).alias("ili_vs_p75"),
    ])


def make_predictions(df_feat: pl.DataFrame, k: int = 3, base_threshold: float = 0.3) -> pd.DataFrame:
    """
    Génère le fichier predictions au format officiel.

    Stratégie K consécutifs : on n'émet une prédiction que lorsque K éclairs
    consécutifs ont tous un score >= base_threshold. Cela évite les faux positifs
    isolés tôt dans l'alerte qui ruineraient la métrique Risk R (min selection).

    confidence = min(scores des K derniers éclairs) → confiance conservative.

    Fallback : pour les alertes sans aucune prédiction K-consécutive,
    on émet une prédiction au dernier éclair avec confidence=base_threshold.
    Cela donne une couverture à 100% des alertes à bas θ sans nuire à R à θ élevé.
    """
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())

    # Ajoute les colonnes manquantes (ex: météo ERA5 absente du test) avec 0
    missing_cols = [c for c in FEATURE_COLS if c not in df_alerts.columns]
    if missing_cols:
        df_alerts = df_alerts.with_columns(
            [pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing_cols]
        )
    X = df_alerts.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = model.predict_proba(X)[:, 1]

    df_pd = df_alerts.select(["airport", "airport_alert_id", "date"]).to_pandas()
    df_pd["score"] = scores
    df_pd["date"] = pd.to_datetime(df_pd["date"], utc=True)

    rows = []
    covered_alerts = set()

    for (airport, alert_id), grp in df_pd.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s = grp["score"].to_numpy()
        alert_has_pred = False
        for i in range(len(s)):
            if i < k - 1:
                continue
            window = s[i - k + 1 : i + 1]
            if (window >= base_threshold).all():
                confidence = float(window.min())
                rows.append({
                    "airport": airport,
                    "airport_alert_id": alert_id,
                    "prediction_date": grp["date"].iloc[i],
                    "predicted_date_end_alert": grp["date"].iloc[i],
                    "confidence": confidence,
                })
                alert_has_pred = True

        if alert_has_pred:
            covered_alerts.add((airport, alert_id))
        else:
            # Fallback : dernier éclair de l'alerte, confidence = base_threshold
            last = grp.iloc[-1]
            rows.append({
                "airport": airport,
                "airport_alert_id": alert_id,
                "prediction_date": last["date"],
                "predicted_date_end_alert": last["date"],
                "confidence": base_threshold,
            })

    all_alerts = set(df_pd.groupby(["airport", "airport_alert_id"]).groups.keys())
    n_fallback = len(all_alerts) - len(covered_alerts)
    print(f"  Alertes couvertes (K-consécutif) : {len(covered_alerts)} / {len(all_alerts)}")
    print(f"  Alertes avec fallback : {n_fallback}")

    return pd.DataFrame(rows, columns=[
        "airport", "airport_alert_id", "prediction_date",
        "predicted_date_end_alert", "confidence"
    ])


# ── 1. Évaluation sur données locales (eval split) ───────────────────────────
print("=== Évaluation sur eval local (données connues) ===\n")

airports = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]
eval_dfs, eval_feat_dfs = [], []
for ap in airports:
    ev_feat = pl.read_parquet(str(PROC / f"{ap}_eval.parquet"))
    eval_feat_dfs.append(ev_feat)

eval_feat_all = pl.concat(eval_feat_dfs)
preds_eval = make_predictions(eval_feat_all, k=K_OPT, base_threshold=THR_OPT)

# Données brutes eval pour les métriques
eval_raw = pd.concat([
    pl.read_parquet(str(PROC / f"{ap}_eval.parquet"))
      .select(["airport", "airport_alert_id", "date", "dist"])
      .to_pandas()
    for ap in airports
])
eval_raw = eval_raw[eval_raw["airport_alert_id"].notna()]

results_eval, tot_eval = compute_official_metrics(eval_raw, preds_eval)
best_theta, _ = print_metrics(results_eval, tot_eval, "Eval local — Risk R & Gain G")

# ── 2. Génération predictions sur dataset test officiel ──────────────────────
print("\n\n=== Génération prédictions sur dataset test officiel ===\n")

print("Chargement et feature engineering du dataset test...")
test_raw = pl.read_csv(
    str(ROOT / "dataset_test" / "dataset_set.csv"),
    schema_overrides={"airport_alert_id": pl.Float64, "is_last_lightning_cloud_ground": pl.Boolean},
).with_columns(
    pl.col("date").str.to_datetime(time_unit="us", time_zone="UTC")
)

# Feature engineering (mêmes fonctions que compute_features.py)
test_feats_raw = cf.compute_features(test_raw)
test_feats_raw = cf.add_terrain_features(test_feats_raw)

# Ajout météo par aéroport, puis harmonisation des schémas
ap_dfs = []
for ap in ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]:
    subset = test_feats_raw.filter(pl.col("airport") == ap)
    subset = cf.add_weather_features(subset, ap)
    ap_dfs.append(subset)

# Harmonise les colonnes (ajoute nulls si météo manquante pour un aéroport)
all_cols = set()
for df in ap_dfs:
    all_cols.update(df.columns)
ap_dfs_aligned = []
for df in ap_dfs:
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None).cast(pl.Float32).alias(c) for c in missing])
    ap_dfs_aligned.append(df)
test_feats = pl.concat(ap_dfs_aligned)

n_alerts = test_feats.filter(pl.col("airport_alert_id").is_not_null())["airport_alert_id"].n_unique()
print(f"Dataset test : {len(test_feats):,} éclairs · {n_alerts} alertes")

print("  Recalcul des percentiles ILI...")
test_feats = recompute_ili_percentiles(test_feats)

preds_test = make_predictions(test_feats, k=K_OPT, base_threshold=THR_OPT)

out_path = ROOT / "dataset_test" / "predictions.csv"
preds_test.to_csv(out_path, index=False)
print(f"predictions.csv sauvegardé → {out_path}")
print(f"  {len(preds_test):,} prédictions · {preds_test['airport_alert_id'].nunique()} alertes")

# ── 3. Évaluation sur test si labels disponibles ─────────────────────────────
has_labels = test_feats.filter(
    pl.col("airport_alert_id").is_not_null()
)["is_last_lightning_cloud_ground"].is_not_null().any()

if has_labels:
    test_raw_pd = test_raw.select(
        ["airport", "airport_alert_id", "date", "dist"]
    ).to_pandas()
    test_raw_pd = test_raw_pd[test_raw_pd["airport_alert_id"].notna()]
    results_test, tot_test = compute_official_metrics(test_raw_pd, preds_test)
    print_metrics(results_test, tot_test, "Dataset test officiel — Risk R & Gain G")
else:
    print("\n(Dataset test sans labels — évaluation impossible localement)")
    if best_theta is not None:
        print(f"→ Utiliser θ={best_theta} (optimal sur eval local)")
