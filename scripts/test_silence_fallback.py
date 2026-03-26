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
test_silence_fallback.py
------------------------
Stratégie complète temps réel :

  1. Déclenchement modèle (précoce) : K=2≥0.355 OU (K=1≥0.60 AND ILI≥P75)
     → gain > 30min sur les alertes couvertes
     → confidence élevée (>0.355)

  2. Fallback silence (tardif) : si aucun éclair depuis T minutes
     → émet prediction_date = flash_i + T, predicted_date_end_alert = flash_i
     → confidence = CONF_SILENCE (très bas, ex: 0.05)
     → gain = ~30min (équivalent baseline mais déclenché proactivement)
     → couvre quasiment toutes les alertes restantes

En real-time : timer T minutes après chaque éclair.
En batch    : pour chaque flash i, si next_flash - current > T min → émet.

On teste T = 5, 10, 15, 20, 25 min et on compare au meilleur actuel.
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

spec = importlib.util.spec_from_file_location(
    "compute_features", ROOT / "scripts" / "compute_features.py"
)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

model        = joblib.load(MODEL / "xgb_best.joblib")
meta         = json.loads((PROC / "feature_cols.json").read_text())
FEATURE_COLS = meta["feature_cols"]
_pp          = joblib.load(MODEL / "predict_params.joblib")
THR_OPT      = _pp["base_threshold"]

MAX_GAP_MIN  = 30
MIN_DIST_KM  = 3
R_ACCEPT     = 0.02
CONF_SILENCE = 0.85   # confidence du fallback silence (même niveau que θ optimal)


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_alerts(df_feat: pl.DataFrame) -> pd.DataFrame:
    df_a     = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    missing  = [c for c in FEATURE_COLS if c not in df_a.columns]
    if missing:
        df_a = df_a.with_columns([pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing])
    X      = df_a.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    cols   = ["airport", "airport_alert_id", "date"]
    if "ili_s" in df_a.columns:
        cols.append("ili_s")
    df_pd  = df_a.select(cols).to_pandas()
    df_pd["score"] = scores
    df_pd["date"]  = pd.to_datetime(df_pd["date"], utc=True)
    if "ili_s" not in df_pd.columns:
        df_pd["ili_s"] = np.nan
    return df_pd


# ── Stratégie complète ────────────────────────────────────────────────────────

def causal_pct(history: list, p: float) -> float:
    return float(np.percentile(history, p)) if len(history) >= 3 else 0.0


def build_full_realtime(df: pd.DataFrame,
                        silence_minutes: float,
                        k2_thr: float = 0.355,
                        k1_score_thr: float = 0.60,
                        ili_pct: float = 75) -> pd.DataFrame:
    """
    Combine :
      - Trigger modèle : K=2≥k2_thr OU (K=1≥k1_score_thr AND ILI≥percentile(ili_pct))
      - Fallback silence : si prochain éclair > silence_minutes min de distance
        → prédit "ce flash est le dernier" avec confidence=CONF_SILENCE

    Toutes les prédictions sont au format (prediction_date, predicted_date_end_alert).
    Batch-simulation : on connaît le next_flash pour calculer le silence.
    """
    silence_td = pd.Timedelta(minutes=silence_minutes)
    rows = []

    for (airport, aid), grp in df.groupby(["airport", "airport_alert_id"], sort=False):
        grp      = grp.sort_values("date").reset_index(drop=True)
        s        = grp["score"].to_numpy()
        dates    = grp["date"].to_numpy()
        ili_vals = grp["ili_s"].fillna(0).to_numpy()
        n        = len(grp)
        ili_hist = []
        model_triggered = False

        for i in range(n):
            ili_now = float(ili_vals[i])
            if ili_now > 0:
                ili_hist.append(ili_now)
            conf = None

            # Trigger A : K=2 consécutifs
            if i >= 1:
                w = s[i-1:i+1]
                if (w >= k2_thr).all():
                    conf = float(w.min())

            # Trigger J : K=1 + ILI ≥ P75 causal
            pct_thr = causal_pct(ili_hist[:-1], ili_pct)
            if s[i] >= k1_score_thr and ili_now >= pct_thr and ili_now > 0:
                c_j = float(s[i])
                if conf is None or c_j > conf:
                    conf = c_j

            if conf is not None:
                rows.append(dict(
                    airport=airport, airport_alert_id=aid,
                    prediction_date=grp["date"].iloc[i],
                    predicted_date_end_alert=grp["date"].iloc[i],
                    confidence=conf,
                ))
                model_triggered = True

        # Fallback silence : seulement si le modèle n'a PAS déclenché
        # ET c'est le dernier flash de l'alerte (silence persist jusqu'à la fin)
        # → en real-time : émettre une seule fois quand T min s'écoulent sans nouveau flash
        if not model_triggered:
            last = grp.iloc[-1]
            rows.append(dict(
                airport=airport, airport_alert_id=aid,
                prediction_date=last["date"] + silence_td,
                predicted_date_end_alert=last["date"],
                confidence=CONF_SILENCE,
            ))

    return pd.DataFrame(rows, columns=[
        "airport", "airport_alert_id",
        "prediction_date", "predicted_date_end_alert", "confidence"
    ])


# ── Métriques ─────────────────────────────────────────────────────────────────

def best_metrics(df_raw: pd.DataFrame, preds: pd.DataFrame):
    if preds.empty:
        return 0.0, None, None, 0, 0
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    preds  = preds.copy()
    preds["predicted_date_end_alert"] = pd.to_datetime(
        preds["predicted_date_end_alert"], utc=True
    )
    tot_dangerous = len(df_raw[df_raw["dist"] < MIN_DIST_KM])
    alerts        = df_raw.groupby(["airport", "airport_alert_id"])
    total_alerts  = df_raw.groupby(["airport", "airport_alert_id"]).ngroups

    best = (0, None, None, 0)
    for theta in np.round(np.linspace(0.01, 0.95, 48), 3).tolist():
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
    df_a = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    airports = df_a["airport"].unique().to_list()
    p75_map, p95_map = {}, {}
    for ap in airports:
        ili = df_a.filter(pl.col("airport") == ap)["ili_s"].drop_nulls().to_numpy()
        if len(ili) > 10:
            p75_map[ap] = float(np.percentile(ili, 75))
            p95_map[ap] = float(np.percentile(ili, 95))
    p75e = pl.lit(None, dtype=pl.Float64)
    p95e = pl.lit(None, dtype=pl.Float64)
    for ap, v in p75_map.items():
        p75e = pl.when(pl.col("airport") == ap).then(pl.lit(v)).otherwise(p75e)
    for ap, v in p95_map.items():
        p95e = pl.when(pl.col("airport") == ap).then(pl.lit(v)).otherwise(p95e)
    return df_feat.with_columns([
        (pl.col("rolling_ili_max_5") / (p95e + 1e-6)).alias("ili_vs_p95"),
        (pl.col("rolling_ili_max_5") / (p75e + 1e-6)).alias("ili_vs_p75"),
    ])


# ── Chargement ───────────────────────────────────────────────────────────────

print("Chargement eval local...")
airports      = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]
eval_feat_all = pl.concat([pl.read_parquet(str(PROC / f"{ap}_eval.parquet")) for ap in airports])
eval_scored   = score_alerts(eval_feat_all)
eval_raw      = pd.concat([
    pl.read_parquet(str(PROC / f"{ap}_eval.parquet"))
      .select(["airport","airport_alert_id","date","dist"]).to_pandas()
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
for ap in ["Ajaccio","Bastia","Biarritz","Nantes","Pise"]:
    sub = test_feats_raw.filter(pl.col("airport") == ap)
    sub = cf.add_weather_features(sub, ap)
    ap_dfs.append(sub)
all_cols = set().union(*[set(d.columns) for d in ap_dfs])
ap_dfs   = [d.with_columns([pl.lit(None).cast(pl.Float32).alias(c)
                              for c in all_cols if c not in d.columns]) for d in ap_dfs]
test_feats  = recompute_ili_percentiles(pl.concat(ap_dfs))
test_scored = score_alerts(test_feats)
test_raw_pd = test_raw_pl.select(["airport","airport_alert_id","date","dist"]).to_pandas()
test_raw_pd = test_raw_pd[test_raw_pd["airport_alert_id"].notna()]
has_labels  = (test_feats.filter(pl.col("airport_alert_id").is_not_null())
               ["is_last_lightning_cloud_ground"].is_not_null().any())


# ── Run ───────────────────────────────────────────────────────────────────────

print("\n" + "="*105)
print("STRATÉGIE TEMPS RÉEL COMPLÈTE : Modèle + Fallback Silence")
print("="*105)
hdr = f"{'Stratégie':<42} {'Gain eval':>10} {'θ':>6} {'Risk':>7} {'Cov%':>6}"
if has_labels:
    hdr += f"  {'Gain test':>10} {'θ':>6} {'Risk':>7} {'Cov%':>6}"
print(hdr)
print("─" * len(hdr))

# Référence AorJ sans silence
from functools import partial

def aorj_only(s):
    """AorJ sans fallback silence (référence)."""
    rows = []
    for (airport, aid), grp in s.groupby(["airport","airport_alert_id"], sort=False):
        grp      = grp.sort_values("date").reset_index(drop=True)
        sc       = grp["score"].to_numpy()
        ili_vals = grp["ili_s"].fillna(0).to_numpy()
        ili_hist = []
        emitted  = set()
        for i in range(len(sc)):
            ili_now = float(ili_vals[i])
            if ili_now > 0:
                ili_hist.append(ili_now)
            conf = None
            if i >= 1:
                w = sc[i-1:i+1]
                if (w >= THR_OPT).all():
                    conf = float(w.min())
            pct_thr = (float(np.percentile(ili_hist[:-1], 75))
                       if len(ili_hist) >= 4 else 0.0)
            if sc[i] >= 0.60 and ili_now >= pct_thr and ili_now > 0:
                c_j = float(sc[i])
                if conf is None or c_j > conf:
                    conf = c_j
            if conf is not None and i not in emitted:
                rows.append(dict(airport=airport, airport_alert_id=aid,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=conf))
                emitted.add(i)
    return pd.DataFrame(rows, columns=["airport","airport_alert_id",
                                        "prediction_date","predicted_date_end_alert","confidence"])

for name, fn in [
    ("AorJ (sans silence, référence)",    aorj_only),
    ("AorJ + silence  5min",  lambda s: build_full_realtime(s,  5, k2_thr=THR_OPT)),
    ("AorJ + silence 10min",  lambda s: build_full_realtime(s, 10, k2_thr=THR_OPT)),
    ("AorJ + silence 15min",  lambda s: build_full_realtime(s, 15, k2_thr=THR_OPT)),
    ("AorJ + silence 20min",  lambda s: build_full_realtime(s, 20, k2_thr=THR_OPT)),
    ("AorJ + silence 25min",  lambda s: build_full_realtime(s, 25, k2_thr=THR_OPT)),
]:
    pe = fn(eval_scored)
    ge, te, re, ce, ne = best_metrics(eval_raw, pe)
    cep = 100*ce/ne if ne else 0
    line = f"{name:<42} {ge:>9.1f}h {str(te):>6} {str(round(re,4)) if re is not None else 'N/A':>7} {cep:>5.1f}%"
    if has_labels:
        pt = fn(test_scored)
        gt, tt, rt, ct, nt = best_metrics(test_raw_pd, pt)
        ctp = 100*ct/nt if nt else 0
        line += f"  {gt:>9.1f}h {str(tt):>6} {str(round(rt,4)) if rt is not None else 'N/A':>7} {ctp:>5.1f}%"
    print(line)

print("─" * len(hdr))
print(f"\nNote : silence CONF_SILENCE={CONF_SILENCE}, THR_OPT={THR_OPT:.3f}")

# ── Sauvegarde meilleure stratégie (AorJ + silence 15min) ────────────────────
print("\nSauvegarde predictions.csv avec AorJ + silence 15min...")
best_preds = build_full_realtime(test_scored, silence_minutes=15, k2_thr=THR_OPT)
out = ROOT / "dataset_test" / "predictions.csv"
best_preds.to_csv(out, index=False)
n_model   = sum(best_preds["confidence"] > CONF_SILENCE)
n_silence = sum(best_preds["confidence"] == CONF_SILENCE)
print(f"  {len(best_preds):,} prédictions total")
print(f"  {n_model:,} prédictions modèle")
print(f"  {n_silence:,} prédictions fallback silence (confidence = {CONF_SILENCE})")
print(f"  → {out}")
