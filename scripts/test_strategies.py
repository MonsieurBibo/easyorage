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
test_strategies.py
------------------
Teste différentes stratégies de génération de predictions.csv et compare
Risk R / Gain G sur eval local et dataset test officiel.

Stratégies testées :
  A (current)  : K=2, fallback confidence = base_threshold
  B            : K=2, fallback confidence = max_score de l'alerte
  C            : K=2, fallback AU flash max_score (pas dernier)
  D            : K=1 (pas de consécutif), fallback = max_score
  E            : Tous les éclairs émis avec leur score brut (no K-filter)
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

# ── Import compute_features ──────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "compute_features", ROOT / "scripts" / "compute_features.py"
)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

# ── Modèle et features ───────────────────────────────────────────────────────
model = joblib.load(MODEL / "xgb_best.joblib")
meta  = json.loads((PROC / "feature_cols.json").read_text())
FEATURE_COLS = meta["feature_cols"]

_pp = joblib.load(MODEL / "predict_params.joblib")
K_OPT   = _pp["k"]
THR_OPT = _pp["base_threshold"]
print(f"Modèle chargé : K={K_OPT}, base_threshold={THR_OPT:.3f}")

MAX_GAP_MIN = 30
MIN_DIST_KM = 3
R_ACCEPT    = 0.02


# ── Score helpers ─────────────────────────────────────────────────────────────

def score_alerts(df_feat: pl.DataFrame) -> pd.DataFrame:
    """Retourne un DataFrame avec (airport, airport_alert_id, date, score)."""
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    missing = [c for c in FEATURE_COLS if c not in df_alerts.columns]
    if missing:
        df_alerts = df_alerts.with_columns(
            [pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing]
        )
    X = df_alerts.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    df_pd = df_alerts.select(["airport", "airport_alert_id", "date"]).to_pandas()
    df_pd["score"] = scores
    df_pd["date"]  = pd.to_datetime(df_pd["date"], utc=True)
    return df_pd


# ── Stratégies ────────────────────────────────────────────────────────────────

def strategy_A(df_scored: pd.DataFrame, k: int = 2, base_thr: float = THR_OPT) -> pd.DataFrame:
    """Current : K-consécutif + fallback confidence=base_thr au dernier éclair."""
    rows = []
    for (airport, alert_id), grp in df_scored.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s = grp["score"].to_numpy()
        triggered = False
        for i in range(len(s)):
            if i < k - 1:
                continue
            window = s[i - k + 1 : i + 1]
            if (window >= base_thr).all():
                rows.append(dict(airport=airport, airport_alert_id=alert_id,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(window.min())))
                triggered = True
        if not triggered:
            last = grp.iloc[-1]
            rows.append(dict(airport=airport, airport_alert_id=alert_id,
                             prediction_date=last["date"],
                             predicted_date_end_alert=last["date"],
                             confidence=base_thr))
    return pd.DataFrame(rows)


def strategy_B(df_scored: pd.DataFrame, k: int = 2, base_thr: float = THR_OPT) -> pd.DataFrame:
    """K-consécutif + fallback confidence=max_score au dernier éclair."""
    rows = []
    for (airport, alert_id), grp in df_scored.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s = grp["score"].to_numpy()
        triggered = False
        for i in range(len(s)):
            if i < k - 1:
                continue
            window = s[i - k + 1 : i + 1]
            if (window >= base_thr).all():
                rows.append(dict(airport=airport, airport_alert_id=alert_id,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(window.min())))
                triggered = True
        if not triggered:
            last = grp.iloc[-1]
            rows.append(dict(airport=airport, airport_alert_id=alert_id,
                             prediction_date=last["date"],
                             predicted_date_end_alert=last["date"],
                             confidence=float(s.max())))
    return pd.DataFrame(rows)


def strategy_C(df_scored: pd.DataFrame, k: int = 2, base_thr: float = THR_OPT) -> pd.DataFrame:
    """K-consécutif + fallback AU flash avec le score max (pas dernier)."""
    rows = []
    for (airport, alert_id), grp in df_scored.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s = grp["score"].to_numpy()
        triggered = False
        for i in range(len(s)):
            if i < k - 1:
                continue
            window = s[i - k + 1 : i + 1]
            if (window >= base_thr).all():
                rows.append(dict(airport=airport, airport_alert_id=alert_id,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(window.min())))
                triggered = True
        if not triggered:
            best_idx = int(np.argmax(s))
            best_flash = grp.iloc[best_idx]
            rows.append(dict(airport=airport, airport_alert_id=alert_id,
                             prediction_date=best_flash["date"],
                             predicted_date_end_alert=best_flash["date"],
                             confidence=float(s.max())))
    return pd.DataFrame(rows)


def strategy_D(df_scored: pd.DataFrame, base_thr: float = THR_OPT) -> pd.DataFrame:
    """K=1 (chaque éclair au-dessus du seuil = prédiction) + fallback=max_score."""
    rows = []
    for (airport, alert_id), grp in df_scored.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s = grp["score"].to_numpy()
        triggered = False
        for i, si in enumerate(s):
            if si >= base_thr:
                rows.append(dict(airport=airport, airport_alert_id=alert_id,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(si)))
                triggered = True
        if not triggered:
            last = grp.iloc[-1]
            rows.append(dict(airport=airport, airport_alert_id=alert_id,
                             prediction_date=last["date"],
                             predicted_date_end_alert=last["date"],
                             confidence=float(s.max())))
    return pd.DataFrame(rows)


def strategy_E(df_scored: pd.DataFrame) -> pd.DataFrame:
    """Tous les éclairs émis avec leur score brut — pas de filtre K."""
    rows = []
    for (airport, alert_id), grp in df_scored.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        for _, row in grp.iterrows():
            rows.append(dict(airport=airport, airport_alert_id=alert_id,
                             prediction_date=row["date"],
                             predicted_date_end_alert=row["date"],
                             confidence=float(row["score"])))
    return pd.DataFrame(rows)


# ── Évaluation ────────────────────────────────────────────────────────────────

def best_metrics(df_raw: pd.DataFrame, preds_df: pd.DataFrame):
    """Retourne (best_gain_h, best_theta, best_risk, n_covered) sous R<0.02."""
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    preds_df = preds_df.copy()
    preds_df["predicted_date_end_alert"] = pd.to_datetime(
        preds_df["predicted_date_end_alert"], utc=True
    )

    tot_dangerous = len(df_raw[df_raw["dist"] < MIN_DIST_KM])
    alerts = df_raw.groupby(["airport", "airport_alert_id"])
    total_alerts = len(df_raw.groupby(["airport", "airport_alert_id"]))

    thetas = np.round(np.linspace(0.05, 0.95, 37), 3).tolist()

    best = (0, None, None, 0)
    for theta in thetas:
        over = preds_df[preds_df["confidence"] >= theta]
        if over.empty:
            continue
        pred_min = over.groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"].min()
        gain, missed = 0, 0
        for (airport, alert_id), end_pred in pred_min.items():
            try:
                lts = alerts.get_group((airport, alert_id))
            except KeyError:
                continue
            end_baseline = pd.to_datetime(lts["date"], utc=True).max() + pd.Timedelta(minutes=MAX_GAP_MIN)
            gain += (end_baseline - end_pred).total_seconds()
            close = pd.to_datetime(lts[lts["dist"] < MIN_DIST_KM]["date"], utc=True)
            missed += int((close > end_pred).sum())

        r = missed / max(tot_dangerous, 1)
        n_cov = len(pred_min)
        if r < R_ACCEPT and gain > best[0]:
            best = (gain, theta, r, n_cov)

    gain_h = best[0] / 3600 if best[0] else 0
    return gain_h, best[1], best[2], best[3], total_alerts


def recompute_ili_percentiles(df_feat: pl.DataFrame) -> pl.DataFrame:
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    airports = df_alerts["airport"].unique().to_list()
    p75_map, p95_map = {}, {}
    for ap in airports:
        ili_vals = (df_alerts.filter(pl.col("airport") == ap)["ili_s"]
                    .drop_nulls().to_numpy())
        if len(ili_vals) > 10:
            p75_map[ap] = float(np.percentile(ili_vals, 75))
            p95_map[ap] = float(np.percentile(ili_vals, 95))

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


# ── Chargement données ────────────────────────────────────────────────────────

print("\n=== Chargement eval local ===")
airports = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]
eval_feat_dfs = [pl.read_parquet(str(PROC / f"{ap}_eval.parquet")) for ap in airports]
eval_feat_all = pl.concat(eval_feat_dfs)
eval_scored   = score_alerts(eval_feat_all)

eval_raw = pd.concat([
    pl.read_parquet(str(PROC / f"{ap}_eval.parquet"))
      .select(["airport", "airport_alert_id", "date", "dist"]).to_pandas()
    for ap in airports
])
eval_raw = eval_raw[eval_raw["airport_alert_id"].notna()]

print("\n=== Chargement dataset test officiel ===")
test_raw_pl = pl.read_csv(
    str(ROOT / "dataset_test" / "dataset_set.csv"),
    schema_overrides={"airport_alert_id": pl.Float64,
                      "is_last_lightning_cloud_ground": pl.Boolean},
).with_columns(pl.col("date").str.to_datetime(time_unit="us", time_zone="UTC"))

test_feats_raw = cf.compute_features(test_raw_pl)
test_feats_raw = cf.add_terrain_features(test_feats_raw)

ap_dfs = []
for ap in ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]:
    subset = test_feats_raw.filter(pl.col("airport") == ap)
    subset = cf.add_weather_features(subset, ap)
    ap_dfs.append(subset)

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
test_feats = recompute_ili_percentiles(test_feats)
test_scored = score_alerts(test_feats)

test_raw_pd = test_raw_pl.select(["airport", "airport_alert_id", "date", "dist"]).to_pandas()
test_raw_pd = test_raw_pd[test_raw_pd["airport_alert_id"].notna()]

has_test_labels = (
    test_feats.filter(pl.col("airport_alert_id").is_not_null())
    ["is_last_lightning_cloud_ground"].is_not_null().any()
)

# ── Run stratégies ────────────────────────────────────────────────────────────

strategies = {
    "A (current K=2 fallback=thr)": lambda s: strategy_A(s),
    "B (K=2 fallback=max_score)":   lambda s: strategy_B(s),
    "C (K=2 fallback=best_flash)":  lambda s: strategy_C(s),
    "D (K=1 fallback=max_score)":   lambda s: strategy_D(s),
    "E (tous éclairs, no K-filter)":lambda s: strategy_E(s),
}

print("\n\n" + "="*80)
print("COMPARAISON DES STRATÉGIES")
print("="*80)

header = f"{'Stratégie':<35} {'Gain eval':>10} {'θ eval':>7} {'R eval':>7} {'Cov%':>6}"
if has_test_labels:
    header += f"  {'Gain test':>10} {'θ test':>7} {'R test':>7} {'Cov%':>6}"
print(header)
print("─" * len(header))

best_strategy_name = None
best_strategy_preds = None
best_test_gain = 0

for name, fn in strategies.items():
    preds_eval = fn(eval_scored)
    g_eval, t_eval, r_eval, cov_eval, tot_eval = best_metrics(eval_raw, preds_eval)
    cov_eval_pct = 100 * cov_eval / tot_eval if tot_eval else 0

    line = f"{name:<35} {g_eval:>9.1f}h {str(t_eval):>7} {str(round(r_eval,4)) if r_eval else 'N/A':>7} {cov_eval_pct:>5.1f}%"

    if has_test_labels:
        preds_test = fn(test_scored)
        g_test, t_test, r_test, cov_test, tot_test = best_metrics(test_raw_pd, preds_test)
        cov_test_pct = 100 * cov_test / tot_test if tot_test else 0
        line += f"  {g_test:>9.1f}h {str(t_test):>7} {str(round(r_test,4)) if r_test else 'N/A':>7} {cov_test_pct:>5.1f}%"

        if g_test > best_test_gain:
            best_test_gain = g_test
            best_strategy_name = name
            best_strategy_preds = preds_test

    print(line)

print("─" * len(header))

# ── Sauvegarde de la meilleure stratégie ─────────────────────────────────────
if has_test_labels and best_strategy_preds is not None:
    print(f"\n→ Meilleure stratégie sur test : {best_strategy_name} ({best_test_gain:.1f}h)")
    out_path = ROOT / "dataset_test" / "predictions.csv"
    best_strategy_preds.to_csv(out_path, index=False)
    print(f"  predictions.csv mis à jour → {out_path}")
else:
    print("\n(Test sans labels — predictions.csv non mis à jour)")
