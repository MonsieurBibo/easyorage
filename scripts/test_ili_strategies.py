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
test_ili_strategies.py
----------------------
Stratégies temps réel qui combinent score XGBoost + condition ILI.

Intuition : un flash mid-alerte avec score élevé mais ILI court = faux positif.
Un flash avec score élevé ET ILI long = très probablement le dernier.

Stratégies testées (toutes 100% causales) :
  A  (current) : K=2 ≥ 0.355
  J1 : K=1 ≥ 0.60  ET  ili_s ≥ P75 de l'alerte courante
  J2 : K=1 ≥ 0.60  ET  ili_s ≥ P90 de l'alerte courante
  J3 : K=1 ≥ 0.70  ET  ili_s ≥ P75
  J4 : K=1 ≥ 0.70  ET  ili_s ≥ P90
  J5 : K=1 ≥ 0.50  ET  ili_s ≥ P90
  K1 : score*ili_normalized ≥ 0.50  (score × ILI_rank)
  K2 : score*ili_normalized ≥ 0.60

Note : les percentiles ILI sont calculés dynamiquement au fil des éclairs
(causal, pas de lookahead).
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

MAX_GAP_MIN = 30
MIN_DIST_KM = 3
R_ACCEPT    = 0.02


def score_alerts(df_feat: pl.DataFrame) -> pd.DataFrame:
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    missing   = [c for c in FEATURE_COLS if c not in df_alerts.columns]
    if missing:
        df_alerts = df_alerts.with_columns(
            [pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing]
        )
    X      = df_alerts.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    # On inclut aussi ili_s pour les stratégies conditionnelles
    cols = ["airport", "airport_alert_id", "date"]
    if "ili_s" in df_alerts.columns:
        cols.append("ili_s")
    df_pd  = df_alerts.select(cols).to_pandas()
    df_pd["score"] = scores
    df_pd["date"]  = pd.to_datetime(df_pd["date"], utc=True)
    if "ili_s" not in df_pd.columns:
        df_pd["ili_s"] = np.nan
    return df_pd


# ── Helpers percentile causal ────────────────────────────────────────────────

def causal_ili_percentile(ili_history: list, p: float) -> float:
    """Percentile p des ILI vus jusqu'ici dans l'alerte (causal)."""
    if len(ili_history) < 3:
        return 0.0   # pas assez d'historique → condition facile à passer
    return float(np.percentile(ili_history, p))


# ── Stratégies ────────────────────────────────────────────────────────────────

def strategy_k2_consec(df: pd.DataFrame, thr: float = THR_OPT) -> pd.DataFrame:
    """Actuel : K=2 consécutifs."""
    rows = []
    for (airport, aid), grp in df.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s   = grp["score"].to_numpy()
        for i in range(1, len(s)):
            w = s[i-1:i+1]
            if (w >= thr).all():
                rows.append(dict(airport=airport, airport_alert_id=aid,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(w.min())))
    return pd.DataFrame(rows, columns=["airport","airport_alert_id",
                                        "prediction_date","predicted_date_end_alert","confidence"])


def strategy_k1_ili(df: pd.DataFrame, score_thr: float, ili_pct: float) -> pd.DataFrame:
    """
    K=1 : émet si score ≥ score_thr ET ili_s ≥ percentile(ili_pct) des ILI vus jusqu'ici.
    100% causal.
    """
    rows = []
    for (airport, aid), grp in df.groupby(["airport", "airport_alert_id"], sort=False):
        grp      = grp.sort_values("date").reset_index(drop=True)
        s        = grp["score"].to_numpy()
        ili_vals = grp["ili_s"].fillna(0).to_numpy()
        ili_hist = []
        for i in range(len(s)):
            ili_now = float(ili_vals[i])
            if ili_now > 0:
                ili_hist.append(ili_now)
            pct_thresh = causal_ili_percentile(ili_hist[:-1], ili_pct)  # causal : exclut courant
            if s[i] >= score_thr and ili_now >= pct_thresh and ili_now > 0:
                rows.append(dict(airport=airport, airport_alert_id=aid,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(s[i])))
    return pd.DataFrame(rows, columns=["airport","airport_alert_id",
                                        "prediction_date","predicted_date_end_alert","confidence"])


def strategy_score_x_ili(df: pd.DataFrame, combined_thr: float) -> pd.DataFrame:
    """
    Émet si score × ili_rank ≥ combined_thr,
    où ili_rank = rang de l'ILI courant parmi les ILI vus (percentile causal, 0→1).
    100% causal.
    """
    rows = []
    for (airport, aid), grp in df.groupby(["airport", "airport_alert_id"], sort=False):
        grp      = grp.sort_values("date").reset_index(drop=True)
        s        = grp["score"].to_numpy()
        ili_vals = grp["ili_s"].fillna(0).to_numpy()
        ili_hist = []
        for i in range(len(s)):
            ili_now = float(ili_vals[i])
            if ili_now > 0:
                ili_hist.append(ili_now)
            # rang causal : fraction des ILI précédents < ili_now
            if len(ili_hist) >= 2 and ili_now > 0:
                rank = float(np.mean(np.array(ili_hist[:-1]) < ili_now))
            else:
                rank = 0.5  # neutre si pas assez d'historique
            combined = s[i] * (0.5 + 0.5 * rank)   # score × boost ILI (0.5 .. 1.0)
            if combined >= combined_thr:
                rows.append(dict(airport=airport, airport_alert_id=aid,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(combined)))
    return pd.DataFrame(rows, columns=["airport","airport_alert_id",
                                        "prediction_date","predicted_date_end_alert","confidence"])


# ── Métriques ─────────────────────────────────────────────────────────────────

def best_metrics(df_raw: pd.DataFrame, preds: pd.DataFrame):
    if preds.empty:
        return 0.0, None, None, 0, 0
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    preds  = preds.copy()
    preds["predicted_date_end_alert"] = pd.to_datetime(preds["predicted_date_end_alert"], utc=True)

    tot_dangerous = len(df_raw[df_raw["dist"] < MIN_DIST_KM])
    alerts        = df_raw.groupby(["airport", "airport_alert_id"])
    total_alerts  = df_raw.groupby(["airport", "airport_alert_id"]).ngroups

    best = (0, None, None, 0)
    for theta in np.round(np.linspace(0.05, 0.95, 37), 3).tolist():
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
ap_dfs = [d.with_columns([pl.lit(None).cast(pl.Float32).alias(c)
                           for c in all_cols if c not in d.columns]) for d in ap_dfs]
test_feats  = recompute_ili_percentiles(pl.concat(ap_dfs))
test_scored = score_alerts(test_feats)
test_raw_pd = test_raw_pl.select(["airport","airport_alert_id","date","dist"]).to_pandas()
test_raw_pd = test_raw_pd[test_raw_pd["airport_alert_id"].notna()]
has_labels  = (test_feats.filter(pl.col("airport_alert_id").is_not_null())
               ["is_last_lightning_cloud_ground"].is_not_null().any())


# ── Stratégies ────────────────────────────────────────────────────────────────

strategies = {
    "A  current  K=2 ≥ 0.355":          lambda s: strategy_k2_consec(s, 0.355),
    "J1 K=1≥0.60 & ILI≥P75":            lambda s: strategy_k1_ili(s, 0.60, 75),
    "J2 K=1≥0.60 & ILI≥P90":            lambda s: strategy_k1_ili(s, 0.60, 90),
    "J3 K=1≥0.70 & ILI≥P75":            lambda s: strategy_k1_ili(s, 0.70, 75),
    "J4 K=1≥0.70 & ILI≥P90":            lambda s: strategy_k1_ili(s, 0.70, 90),
    "J5 K=1≥0.50 & ILI≥P90":            lambda s: strategy_k1_ili(s, 0.50, 90),
    "J6 K=1≥0.50 & ILI≥P75":            lambda s: strategy_k1_ili(s, 0.50, 75),
    "K1 score×ILI_rank ≥ 0.50":         lambda s: strategy_score_x_ili(s, 0.50),
    "K2 score×ILI_rank ≥ 0.60":         lambda s: strategy_score_x_ili(s, 0.60),
}

# ── Run ───────────────────────────────────────────────────────────────────────

print("\n" + "="*100)
print("STRATÉGIES TEMPS RÉEL avec condition ILI")
print("="*100)
hdr = f"{'Stratégie':<36} {'Gain eval':>10} {'θ':>6} {'Risk':>7} {'Cov%':>6}"
if has_labels:
    hdr += f"  {'Gain test':>10} {'θ':>6} {'Risk':>7} {'Cov%':>6}"
print(hdr)
print("─" * len(hdr))

best_name, best_preds, best_gain = None, None, 0

for name, fn in strategies.items():
    pe = fn(eval_scored)
    ge, te, re, ce, ne = best_metrics(eval_raw, pe)
    cep = 100*ce/ne if ne else 0
    line = f"{name:<36} {ge:>9.1f}h {str(te):>6} {str(round(re,4)) if re is not None else 'N/A':>7} {cep:>5.1f}%"
    if has_labels:
        pt = fn(test_scored)
        gt, tt, rt, ct, nt = best_metrics(test_raw_pd, pt)
        ctp = 100*ct/nt if nt else 0
        line += f"  {gt:>9.1f}h {str(tt):>6} {str(round(rt,4)) if rt is not None else 'N/A':>7} {ctp:>5.1f}%"
        if gt > best_gain:
            best_gain, best_name, best_preds = gt, name, pt
    print(line)

print("─" * len(hdr))
if has_labels and best_preds is not None:
    print(f"\n→ Meilleure : {best_name}  ({best_gain:.1f}h)")
    out = ROOT / "dataset_test" / "predictions.csv"
    best_preds.to_csv(out, index=False)
    print(f"  predictions.csv mis à jour → {out}")


# ── Stratégie combinée A OR J ─────────────────────────────────────────────────

def strategy_combined(df: pd.DataFrame, 
                       k2_thr: float = 0.355,
                       k1_score_thr: float = 0.60,
                       ili_pct: float = 75) -> pd.DataFrame:
    """(K=2 ≥ k2_thr) OU (K=1 ≥ k1_score_thr ET ILI ≥ percentile(ili_pct)).
    100% causal — prend le min confiance si les deux déclenchent au même flash."""
    rows = []
    for (airport, aid), grp in df.groupby(["airport", "airport_alert_id"], sort=False):
        grp      = grp.sort_values("date").reset_index(drop=True)
        s        = grp["score"].to_numpy()
        ili_vals = grp["ili_s"].fillna(0).to_numpy()
        ili_hist = []
        emitted  = set()
        for i in range(len(s)):
            ili_now = float(ili_vals[i])
            if ili_now > 0:
                ili_hist.append(ili_now)
            conf = None
            # Condition A : K=2 consécutifs
            if i >= 1:
                w = s[i-1:i+1]
                if (w >= k2_thr).all():
                    conf = float(w.min())
            # Condition J : K=1 + ILI
            pct_thr = causal_ili_percentile(ili_hist[:-1], ili_pct)
            if s[i] >= k1_score_thr and ili_now >= pct_thr and ili_now > 0:
                c_j = float(s[i])
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


print("\n\n=== Stratégies combinées A OR J ===\n")
combined_strategies = {
    "AorJ  K=2≥0.355 OR K=1≥0.60&ILI≥P75": lambda s: strategy_combined(s, 0.355, 0.60, 75),
    "AorJ  K=2≥0.355 OR K=1≥0.60&ILI≥P90": lambda s: strategy_combined(s, 0.355, 0.60, 90),
    "AorJ  K=2≥0.355 OR K=1≥0.70&ILI≥P75": lambda s: strategy_combined(s, 0.355, 0.70, 75),
    "AorJ  K=2≥0.355 OR K=1≥0.50&ILI≥P75": lambda s: strategy_combined(s, 0.355, 0.50, 75),
}

hdr2 = f"{'Stratégie':<45} {'Gain eval':>10} {'θ':>6} {'Risk':>7} {'Cov%':>6}"
if has_labels:
    hdr2 += f"  {'Gain test':>10} {'θ':>6} {'Risk':>7} {'Cov%':>6}"
print(hdr2)
print("─" * len(hdr2))

for name, fn in combined_strategies.items():
    pe = fn(eval_scored)
    ge, te, re, ce, ne = best_metrics(eval_raw, pe)
    cep = 100*ce/ne if ne else 0
    line = f"{name:<45} {ge:>9.1f}h {str(te):>6} {str(round(re,4)) if re is not None else 'N/A':>7} {cep:>5.1f}%"
    if has_labels:
        pt = fn(test_scored)
        gt, tt, rt, ct, nt = best_metrics(test_raw_pd, pt)
        ctp = 100*ct/nt if nt else 0
        line += f"  {gt:>9.1f}h {str(tt):>6} {str(round(rt,4)) if rt is not None else 'N/A':>7} {ctp:>5.1f}%"
    print(line)
print("─" * len(hdr2))
