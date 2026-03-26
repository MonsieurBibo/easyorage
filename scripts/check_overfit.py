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
check_overfit.py
----------------
Analyse complète overfitting / généralisation :

1. AUC sur train, eval local (2016-2022), eval officiel (2023-2025)
2. Distribution des scores par split
3. Risk/Gain sur eval officiel (protocol officiel)
4. Comparaison distribution alertes train vs eval officiel
"""

import json
import pathlib
import importlib.util

import numpy as np
import pandas as pd
import polars as pl
import joblib
from sklearn.metrics import roc_auc_score

ROOT  = pathlib.Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
MODEL = ROOT / "models"
EVAL_CSV = ROOT / "segment_alerts_all_airports_eval.csv"

spec = importlib.util.spec_from_file_location(
    "compute_features", ROOT / "scripts" / "compute_features.py"
)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

model        = joblib.load(MODEL / "xgb_best.joblib")
meta         = json.loads((PROC / "feature_cols.json").read_text())
FEATURE_COLS = meta["feature_cols"]
_pp          = joblib.load(MODEL / "predict_params.joblib")
K_OPT, THR_OPT = _pp["k"], _pp["base_threshold"]

MAX_GAP_MIN = 30
MIN_DIST_KM = 3
R_ACCEPT    = 0.02
AIRPORTS    = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_scores(df_feat: pl.DataFrame):
    df_a    = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    missing = [c for c in FEATURE_COLS if c not in df_a.columns]
    if missing:
        df_a = df_a.with_columns([pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing])
    X      = df_a.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    y      = df_a["is_last_lightning_cloud_ground"].to_numpy().astype(float)
    return scores, y, df_a

def auc_label(scores, y):
    mask = ~np.isnan(y)
    return roc_auc_score(y[mask], scores[mask]) if mask.sum() > 0 else np.nan

def make_predictions_k2(df_pd: pd.DataFrame, k: int, thr: float) -> pd.DataFrame:
    rows = []
    for (airport, aid), grp in df_pd.groupby(["airport","airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s   = grp["score"].to_numpy()
        triggered = False
        for i in range(len(s)):
            if i < k - 1:
                continue
            w = s[i - k + 1 : i + 1]
            if (w >= thr).all():
                rows.append(dict(airport=airport, airport_alert_id=aid,
                                 prediction_date=grp["date"].iloc[i],
                                 predicted_date_end_alert=grp["date"].iloc[i],
                                 confidence=float(w.min())))
                triggered = True
        if not triggered:
            last = grp.iloc[-1]
            rows.append(dict(airport=airport, airport_alert_id=aid,
                             prediction_date=last["date"],
                             predicted_date_end_alert=last["date"],
                             confidence=float(s.max())))
    return pd.DataFrame(rows, columns=["airport","airport_alert_id",
                                        "prediction_date","predicted_date_end_alert","confidence"])

def best_metrics(df_raw: pd.DataFrame, preds: pd.DataFrame):
    if preds.empty:
        return 0.0, None, None, 0, 0
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    preds = preds.copy()
    preds["predicted_date_end_alert"] = pd.to_datetime(preds["predicted_date_end_alert"], utc=True)
    tot_d  = len(df_raw[df_raw["dist"] < MIN_DIST_KM])
    alerts = df_raw.groupby(["airport","airport_alert_id"])
    n_tot  = df_raw.groupby(["airport","airport_alert_id"]).ngroups
    best   = (0, None, None, 0)
    for theta in np.round(np.linspace(0.05, 0.95, 37), 3):
        over = preds[preds["confidence"] >= theta]
        if over.empty:
            continue
        pm = over.groupby(["airport","airport_alert_id"])["predicted_date_end_alert"].min()
        g, m = 0, 0
        for (ap, aid), ep in pm.items():
            try:
                lts = alerts.get_group((ap, aid))
            except KeyError:
                continue
            g += (pd.to_datetime(lts["date"], utc=True).max()
                  + pd.Timedelta(minutes=MAX_GAP_MIN) - ep).total_seconds()
            m += int((pd.to_datetime(lts[lts["dist"]<MIN_DIST_KM]["date"], utc=True) > ep).sum())
        r = m / max(tot_d, 1)
        if r < R_ACCEPT and g > best[0]:
            best = (g, theta, r, len(pm))
    return best[0]/3600, best[1], best[2], best[3], n_tot

def recompute_ili(df_feat: pl.DataFrame) -> pl.DataFrame:
    df_a = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    p75_map, p95_map = {}, {}
    for ap in df_a["airport"].unique().to_list():
        ili = df_a.filter(pl.col("airport")==ap)["ili_s"].drop_nulls().to_numpy()
        if len(ili) > 10:
            p75_map[ap] = float(np.percentile(ili, 75))
            p95_map[ap] = float(np.percentile(ili, 95))
    p75e = pl.lit(None, dtype=pl.Float64)
    p95e = pl.lit(None, dtype=pl.Float64)
    for ap, v in p75_map.items():
        p75e = pl.when(pl.col("airport")==ap).then(pl.lit(v)).otherwise(p75e)
    for ap, v in p95_map.items():
        p95e = pl.when(pl.col("airport")==ap).then(pl.lit(v)).otherwise(p95e)
    return df_feat.with_columns([
        (pl.col("rolling_ili_max_5")/(p95e+1e-6)).alias("ili_vs_p95"),
        (pl.col("rolling_ili_max_5")/(p75e+1e-6)).alias("ili_vs_p75"),
    ])


# ── 1. AUC train ─────────────────────────────────────────────────────────────
print("Calcul AUC train...")
train_dfs = [pl.read_parquet(str(PROC/f"{ap}_train.parquet")) for ap in AIRPORTS]
train_all  = pl.concat(train_dfs)
s_train, y_train, _ = get_scores(train_all)
auc_train = auc_label(s_train, y_train)

# ── 2. AUC eval local (2016-2022) ─────────────────────────────────────────────
print("Calcul AUC eval local...")
eval_dfs = [pl.read_parquet(str(PROC/f"{ap}_eval.parquet")) for ap in AIRPORTS]
eval_all  = pl.concat(eval_dfs)
s_eval, y_eval, df_eval_a = get_scores(eval_all)
auc_eval  = auc_label(s_eval, y_eval)

# ── 3. Features sur eval officiel (2023-2025) ─────────────────────────────────
print("Feature engineering eval officiel (2023-2025)...")
raw_off = pl.read_csv(
    str(EVAL_CSV),
    schema_overrides={"airport_alert_id": pl.Float64,
                      "is_last_lightning_cloud_ground": pl.Boolean},
).with_columns(pl.col("date").str.to_datetime(time_unit="us", time_zone="UTC"))

feats_raw = cf.compute_features(raw_off)
feats_raw = cf.add_terrain_features(feats_raw)
ap_dfs = []
for ap in ["Ajaccio","Bastia","Biarritz","Nantes","Pise"]:
    sub = feats_raw.filter(pl.col("airport")==ap)
    sub = cf.add_weather_features(sub, ap)
    ap_dfs.append(sub)
all_cols = set().union(*[set(d.columns) for d in ap_dfs])
ap_dfs = [d.with_columns([pl.lit(None).cast(pl.Float32).alias(c)
                           for c in all_cols if c not in d.columns]) for d in ap_dfs]
feats_off = recompute_ili(pl.concat(ap_dfs))

s_off, y_off, df_off_a = get_scores(feats_off)
auc_off = auc_label(s_off, y_off)

# ── 4. Risk/Gain eval officiel ────────────────────────────────────────────────
print("Calcul Risk/Gain eval officiel...")
df_off_pd = df_off_a.select(["airport","airport_alert_id","date"]).to_pandas()
df_off_pd["score"] = s_off
df_off_pd["date"]  = pd.to_datetime(df_off_pd["date"], utc=True)
preds_off = make_predictions_k2(df_off_pd, K_OPT, THR_OPT)

raw_off_pd = raw_off.select(["airport","airport_alert_id","date","dist"]).to_pandas()
raw_off_pd = raw_off_pd[raw_off_pd["airport_alert_id"].notna()]
gain_off, theta_off, risk_off, cov_off, tot_off = best_metrics(raw_off_pd, preds_off)

# Même chose eval local pour comparaison
eval_raw_pd = pd.concat([
    pl.read_parquet(str(PROC/f"{ap}_eval.parquet"))
      .select(["airport","airport_alert_id","date","dist"]).to_pandas()
    for ap in AIRPORTS
])
eval_raw_pd = eval_raw_pd[eval_raw_pd["airport_alert_id"].notna()]
df_eval_pd = df_eval_a.select(["airport","airport_alert_id","date"]).to_pandas()
df_eval_pd["score"] = s_eval
df_eval_pd["date"]  = pd.to_datetime(df_eval_pd["date"], utc=True)
preds_local = make_predictions_k2(df_eval_pd, K_OPT, THR_OPT)
gain_local, theta_local, risk_local, cov_local, tot_local = best_metrics(eval_raw_pd, preds_local)

# ── 5. Distribution des scores ────────────────────────────────────────────────
pos_train = s_train[y_train == 1]
neg_train = s_train[y_train == 0]
pos_off   = s_off[y_off == 1]
neg_off   = s_off[y_off == 0]

# ── Rapport ────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ANALYSE OVERFITTING / GÉNÉRALISATION")
print("="*65)

print(f"\n{'':40} {'Train':>10} {'Eval local':>10} {'Eval off.':>10}")
print(f"{'':40} {'2016-2022':>10} {'2016-2022':>10} {'2023-2025':>10}")
print("─"*72)
print(f"{'AUC':40} {auc_train:>10.4f} {auc_eval:>10.4f} {auc_off:>10.4f}")
print(f"{'Gain G (h)':40} {'—':>10} {gain_local:>10.1f} {gain_off:>10.1f}")
print(f"{'Risk R':40} {'—':>10} {risk_local:>7.4f}{'✓' if risk_local<R_ACCEPT else '✗':>3} {risk_off:>7.4f}{'✓' if risk_off is not None and risk_off<R_ACCEPT else '✗':>3}")
print(f"{'θ optimal':40} {'—':>10} {theta_local:>10} {theta_off:>10}")
print(f"{'Couverture alertes':40} {'—':>10} {100*cov_local/tot_local:>9.1f}% {100*cov_off/tot_off:>9.1f}%")

print(f"\n{'':40} {'Train':>10} {'Eval off.':>10}")
print("─"*52)
print(f"{'Score médian positifs (vrais derniers)':40} {np.median(pos_train):>10.3f} {np.median(pos_off):>10.3f}")
print(f"{'Score médian négatifs (non-derniers)':40} {np.median(neg_train):>10.3f} {np.median(neg_off):>10.3f}")
print(f"{'Score P90 positifs':40} {np.percentile(pos_train,90):>10.3f} {np.percentile(pos_off,90):>10.3f}")
print(f"{'Taux positifs (% derniers éclairs)':40} {y_train.mean():>10.3%} {y_off.mean():>10.3%}")

print(f"\n{'':40} {'Eval local':>10} {'Eval off.':>10}")
print("─"*52)

# Statistiques alertes
eval_len = eval_raw_pd.groupby(["airport","airport_alert_id"]).size()
off_len  = raw_off_pd.groupby(["airport","airport_alert_id"]).size()
print(f"{'Nb alertes':40} {len(eval_len):>10} {len(off_len):>10}")
print(f"{'Médiane éclairs/alerte':40} {eval_len.median():>10.0f} {off_len.median():>10.0f}")
print(f"{'Alertes 1 éclair':40} {(eval_len==1).mean()*100:>9.1f}% {(off_len==1).mean()*100:>9.1f}%")

# ILI
def flat_ili(raw_pd):
    vals = []
    for _, g in raw_pd.groupby(["airport","airport_alert_id"]):
        g = g.sort_values("date")
        ili = g["date"].diff().dt.total_seconds().dropna().tolist()
        vals.extend(ili)
    return np.array(vals)

ili_local = flat_ili(eval_raw_pd)
ili_off   = flat_ili(raw_off_pd)
print(f"{'ILI médian':40} {np.median(ili_local):>9.0f}s {np.median(ili_off):>9.0f}s")
print(f"{'ILI P95':40} {np.percentile(ili_local,95):>9.0f}s {np.percentile(ili_off,95):>9.0f}s")
print("─"*52)

# Verdict
gap_auc = auc_train - auc_off
print(f"\n→ Gap AUC train→eval officiel : {gap_auc:.4f} ({'+' if gap_auc>=0 else ''}{gap_auc:.1%})")
if gap_auc > 0.05:
    print("  ⚠️  Overfitting probable (gap > 5pts)")
elif gap_auc > 0.02:
    print("  ⚠️  Légère généralisation difficile (gap 2-5pts)")
else:
    print("  ✓  Pas d'overfitting significatif")

gap_gain = gain_local - gain_off
print(f"→ Gap Gain eval local→officiel : {gap_gain:.1f}h")
print(f"  Distribution shift (ILI: {np.median(ili_local):.0f}s→{np.median(ili_off):.0f}s, "
      f"médiane éclairs: {eval_len.median():.0f}→{off_len.median():.0f})")
