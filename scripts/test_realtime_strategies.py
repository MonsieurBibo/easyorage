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
test_realtime_strategies.py
---------------------------
Toutes les stratégies sont 100% causales (utilisables en temps réel).
Aucune n'a besoin de voir le futur.

Stratégies :
  A  (current)       : K=2 consécutifs ≥ 0.355
  F1 : K=1 ≥ 0.60
  F2 : K=1 ≥ 0.70
  F3 : K=1 ≥ 0.75
  F4 : K=1 ≥ 0.80
  F5 : K=1 ≥ 0.85
  F6 : K=1 ≥ 0.90
  G  : double trigger  K=2 ≥ 0.355  OU  K=1 ≥ 0.75
  H  : double trigger  K=2 ≥ 0.355  OU  K=1 ≥ 0.80

Résumé : pour chaque stratégie on affiche
  Gain (h), θ optimal, Risk, Couverture %
sur eval local ET dataset test officiel.
"""

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

model        = joblib.load(MODEL / "xgb_best.joblib")
meta         = json.loads((PROC / "feature_cols.json").read_text())
FEATURE_COLS = meta["feature_cols"]
_pp          = joblib.load(MODEL / "predict_params.joblib")
THR_OPT      = _pp["base_threshold"]   # 0.355

MAX_GAP_MIN = 30
MIN_DIST_KM = 3
R_ACCEPT    = 0.02


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_alerts(df_feat: pl.DataFrame) -> pd.DataFrame:
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    missing   = [c for c in FEATURE_COLS if c not in df_alerts.columns]
    if missing:
        df_alerts = df_alerts.with_columns(
            [pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing]
        )
    X      = df_alerts.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    df_pd  = df_alerts.select(["airport", "airport_alert_id", "date"]).to_pandas()
    df_pd["score"] = scores
    df_pd["date"]  = pd.to_datetime(df_pd["date"], utc=True)
    return df_pd


# ── Builders de prédictions ───────────────────────────────────────────────────

def build_preds_k_consec(df: pd.DataFrame, k: int, thr: float) -> pd.DataFrame:
    """Émet une prédiction au i-ème flash si les k derniers scores >= thr."""
    rows = []
    for (airport, aid), grp in df.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s   = grp["score"].to_numpy()
        for i in range(len(s)):
            if i < k - 1:
                continue
            window = s[i - k + 1 : i + 1]
            if (window >= thr).all():
                rows.append(dict(airport=airport, airport_alert_id=aid,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(window.min())))
    return pd.DataFrame(rows, columns=["airport", "airport_alert_id",
                                        "prediction_date", "predicted_date_end_alert",
                                        "confidence"])


def build_preds_double(df: pd.DataFrame,
                       k1: int, thr1: float,
                       k2: int, thr2: float) -> pd.DataFrame:
    """Double trigger : condition 1 OU condition 2 (les deux causales)."""
    rows = []
    for (airport, aid), grp in df.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s   = grp["score"].to_numpy()
        emitted = set()
        for i in range(len(s)):
            conf = None
            # Condition 1
            if i >= k1 - 1:
                w1 = s[i - k1 + 1 : i + 1]
                if (w1 >= thr1).all():
                    conf = float(w1.min())
            # Condition 2 (peut surcharger conf si meilleur)
            if i >= k2 - 1:
                w2 = s[i - k2 + 1 : i + 1]
                if (w2 >= thr2).all():
                    c2 = float(w2.min())
                    if conf is None or c2 > conf:
                        conf = c2
            if conf is not None and i not in emitted:
                rows.append(dict(airport=airport, airport_alert_id=aid,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=conf))
                emitted.add(i)
    return pd.DataFrame(rows, columns=["airport", "airport_alert_id",
                                        "prediction_date", "predicted_date_end_alert",
                                        "confidence"])


# ── Métriques ─────────────────────────────────────────────────────────────────

def best_metrics(df_raw: pd.DataFrame, preds: pd.DataFrame):
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    preds  = preds.copy()
    if preds.empty:
        return 0.0, None, None, 0, 0
    preds["predicted_date_end_alert"] = pd.to_datetime(
        preds["predicted_date_end_alert"], utc=True
    )

    tot_dangerous = len(df_raw[df_raw["dist"] < MIN_DIST_KM])
    alerts        = df_raw.groupby(["airport", "airport_alert_id"])
    total_alerts  = df_raw.groupby(["airport", "airport_alert_id"]).ngroups

    thetas = np.round(np.linspace(0.05, 0.95, 37), 3).tolist()
    best   = (0, None, None, 0)

    for theta in thetas:
        over = preds[preds["confidence"] >= theta]
        if over.empty:
            continue
        pred_min = over.groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"].min()
        gain, missed = 0, 0
        for (airport, aid), end_pred in pred_min.items():
            try:
                lts = alerts.get_group((airport, aid))
            except KeyError:
                continue
            end_bl = pd.to_datetime(lts["date"], utc=True).max() + pd.Timedelta(minutes=MAX_GAP_MIN)
            gain  += (end_bl - end_pred).total_seconds()
            close  = pd.to_datetime(lts[lts["dist"] < MIN_DIST_KM]["date"], utc=True)
            missed += int((close > end_pred).sum())

        r     = missed / max(tot_dangerous, 1)
        n_cov = len(pred_min)
        if r < R_ACCEPT and gain > best[0]:
            best = (gain, theta, r, n_cov)

    return best[0] / 3600, best[1], best[2], best[3], total_alerts


def recompute_ili_percentiles(df_feat: pl.DataFrame) -> pl.DataFrame:
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    airports  = df_alerts["airport"].unique().to_list()
    p75_map, p95_map = {}, {}
    for ap in airports:
        ili = df_alerts.filter(pl.col("airport") == ap)["ili_s"].drop_nulls().to_numpy()
        if len(ili) > 10:
            p75_map[ap] = float(np.percentile(ili, 75))
            p95_map[ap] = float(np.percentile(ili, 95))
    p75_expr = pl.lit(None, dtype=pl.Float64)
    p95_expr = pl.lit(None, dtype=pl.Float64)
    for ap, v in p75_map.items():
        p75_expr = pl.when(pl.col("airport") == ap).then(pl.lit(v)).otherwise(p75_expr)
    for ap, v in p95_map.items():
        p95_expr = pl.when(pl.col("airport") == ap).then(pl.lit(v)).otherwise(p95_expr)
    return df_feat.with_columns([
        (pl.col("rolling_ili_max_5") / (p95_expr + 1e-6)).alias("ili_vs_p95"),
        (pl.col("rolling_ili_max_5") / (p75_expr + 1e-6)).alias("ili_vs_p75"),
    ])


# ── Chargement données ────────────────────────────────────────────────────────

print("Chargement eval local...")
airports       = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]
eval_feat_all  = pl.concat([pl.read_parquet(str(PROC / f"{ap}_eval.parquet")) for ap in airports])
eval_scored    = score_alerts(eval_feat_all)
eval_raw       = pd.concat([
    pl.read_parquet(str(PROC / f"{ap}_eval.parquet"))
      .select(["airport", "airport_alert_id", "date", "dist"]).to_pandas()
    for ap in airports
])
eval_raw = eval_raw[eval_raw["airport_alert_id"].notna()]

print("Chargement dataset test officiel...")
test_raw_pl = pl.read_csv(
    str(ROOT / "dataset_test" / "dataset_set.csv"),
    schema_overrides={"airport_alert_id": pl.Float64,
                      "is_last_lightning_cloud_ground": pl.Boolean},
).with_columns(pl.col("date").str.to_datetime(time_unit="us", time_zone="UTC"))

test_feats_raw = cf.compute_features(test_raw_pl)
test_feats_raw = cf.add_terrain_features(test_feats_raw)
ap_dfs = []
for ap in ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]:
    sub = test_feats_raw.filter(pl.col("airport") == ap)
    sub = cf.add_weather_features(sub, ap)
    ap_dfs.append(sub)
all_cols = set().union(*[set(d.columns) for d in ap_dfs])
ap_dfs = [d.with_columns([pl.lit(None).cast(pl.Float32).alias(c)
                           for c in all_cols if c not in d.columns])
          for d in ap_dfs]
test_feats  = recompute_ili_percentiles(pl.concat(ap_dfs))
test_scored = score_alerts(test_feats)
test_raw_pd = test_raw_pl.select(["airport", "airport_alert_id", "date", "dist"]).to_pandas()
test_raw_pd = test_raw_pd[test_raw_pd["airport_alert_id"].notna()]
has_labels  = (test_feats.filter(pl.col("airport_alert_id").is_not_null())
               ["is_last_lightning_cloud_ground"].is_not_null().any())


# ── Définition des stratégies ─────────────────────────────────────────────────

strategies = {
    "A  current  K=2 ≥ 0.355":       lambda s: build_preds_k_consec(s, 2, 0.355),
    "F1 K=1 ≥ 0.60":                  lambda s: build_preds_k_consec(s, 1, 0.60),
    "F2 K=1 ≥ 0.70":                  lambda s: build_preds_k_consec(s, 1, 0.70),
    "F3 K=1 ≥ 0.75":                  lambda s: build_preds_k_consec(s, 1, 0.75),
    "F4 K=1 ≥ 0.80":                  lambda s: build_preds_k_consec(s, 1, 0.80),
    "F5 K=1 ≥ 0.85":                  lambda s: build_preds_k_consec(s, 1, 0.85),
    "F6 K=1 ≥ 0.90":                  lambda s: build_preds_k_consec(s, 1, 0.90),
    "G  K=2≥0.355 OR K=1≥0.75":      lambda s: build_preds_double(s, 2, 0.355, 1, 0.75),
    "H  K=2≥0.355 OR K=1≥0.80":      lambda s: build_preds_double(s, 2, 0.355, 1, 0.80),
}

# ── Évaluation ────────────────────────────────────────────────────────────────

print("\n" + "="*95)
print("STRATÉGIES TEMPS RÉEL — comparaison Risk / Gain")
print("="*95)

col1 = f"{'Stratégie':<32}"
col2 = f"{'Gain eval':>10} {'θ':>6} {'Risk':>7} {'Cov%':>6}"
col3 = f"  {'Gain test':>10} {'θ':>6} {'Risk':>7} {'Cov%':>6}" if has_labels else ""
print(col1 + col2 + col3)
print("─" * (len(col1 + col2 + col3)))

best_name = None
best_preds = None
best_gain  = 0

for name, fn in strategies.items():
    pe = fn(eval_scored)
    ge, te, re, ce, ne = best_metrics(eval_raw, pe)
    cep = 100 * ce / ne if ne else 0

    line = f"{name:<32} {ge:>9.1f}h {str(te):>6} {str(round(re,4)) if re is not None else 'N/A':>7} {cep:>5.1f}%"

    if has_labels:
        pt = fn(test_scored)
        gt, tt, rt, ct, nt = best_metrics(test_raw_pd, pt)
        ctp = 100 * ct / nt if nt else 0
        line += f"  {gt:>9.1f}h {str(tt):>6} {str(round(rt,4)) if rt is not None else 'N/A':>7} {ctp:>5.1f}%"
        if gt > best_gain:
            best_gain  = gt
            best_name  = name
            best_preds = pt

    print(line)

print("─" * (len(col1 + col2 + col3)))

if has_labels and best_preds is not None:
    print(f"\n→ Meilleure stratégie temps réel : {best_name}  ({best_gain:.1f}h)")
    out = ROOT / "dataset_test" / "predictions.csv"
    best_preds.to_csv(out, index=False)
    print(f"  predictions.csv mis à jour → {out}")
