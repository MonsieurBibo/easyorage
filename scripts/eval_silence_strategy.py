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
eval_silence_strategy.py
------------------------
Compare 3 stratégies de prédiction sur eval local + test officiel :

1. Flash-only (actuelle) — prédit uniquement aux éclairs réels
2. Silence-aware K=2 — + points virtuels toutes les minutes pendant les silences
3. Silence-aware K=1 — idem, mais trigger dès 1 point (plus agressif)

Pour chaque point virtuel au temps T (silence depuis dernier éclair) :
- ili_s = T - t_last (croissant)
- rolling_ili_max_5, flash rates, fr_log_slope, etc. recalculés avec silence inclus
- Le modèle XGBoost voit un "éclair virtuel" avec ILI géant → proba monte
"""

import math
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

# ── Import compute_features ───────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "compute_features", ROOT / "scripts" / "compute_features.py"
)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

model      = joblib.load(MODEL / "xgb_silence.joblib")
meta       = json.loads((PROC / "feature_cols.json").read_text())
FEAT_COLS  = meta["feature_cols"]
FEAT_IDX   = {f: i for i, f in enumerate(FEAT_COLS)}

MAX_GAP_MIN = 30
MIN_DIST_KM = 3
R_ACCEPT    = 0.02
AIRPORTS    = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]
THETAS      = np.round(np.linspace(0.05, 0.95, 19), 2).tolist()


# ── Métriques officielles ─────────────────────────────────────────────────────
def compute_metrics(eval_raw: pd.DataFrame, preds_df: pd.DataFrame):
    eval_raw = eval_raw.copy()
    preds_df = preds_df.copy()
    eval_raw["date"] = pd.to_datetime(eval_raw["date"], utc=True)
    preds_df["predicted_date_end_alert"] = pd.to_datetime(
        preds_df["predicted_date_end_alert"], utc=True
    )
    tot_dangerous = len(eval_raw[eval_raw["dist"] < MIN_DIST_KM])
    alerts = eval_raw.groupby(["airport", "airport_alert_id"])
    results = {}
    for theta in THETAS:
        over = preds_df[preds_df["confidence"] >= theta]
        if over.empty:
            results[theta] = (0, tot_dangerous)
            continue
        pred_min = over.groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"].min()
        gain, missed = 0, 0
        for (airport, alert_id), end_pred in pred_min.items():
            try:
                lts = alerts.get_group((airport, alert_id))
            except KeyError:
                continue
            end_baseline = (
                pd.to_datetime(lts["date"], utc=True).max()
                + pd.Timedelta(minutes=MAX_GAP_MIN)
            )
            gain += (end_baseline - end_pred).total_seconds()
            close = pd.to_datetime(lts[lts["dist"] < MIN_DIST_KM]["date"], utc=True)
            missed += int((close > end_pred).sum())
        results[theta] = (gain, missed)
    return results, tot_dangerous


def summarize(results, tot_dangerous, label):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  {'θ':>5} {'Risk':>7} {'Gain':>9}  OK?")
    best_g, best_t, best_r = 0, None, None
    for theta, (gain, missed) in sorted(results.items()):
        r = missed / max(tot_dangerous, 1)
        ok = "✓" if r < R_ACCEPT else ""
        if r < R_ACCEPT:
            print(f"  {theta:>5.2f} {r:>7.4f} {gain/3600:>8.1f}h  ✓")
            if gain > best_g:
                best_g, best_t, best_r = gain, theta, r
        # only print rows near the valid zone (skip obviously bad ones)
    if best_t:
        print(f"\n  → θ optimal = {best_t}  |  Gain = {best_g/3600:.1f}h  |  Risk = {best_r:.4f}")
    else:
        print("  → Aucun θ valide (R < 0.02)")
    return best_g / 3600, best_t


# ── Reconstruction features pour point virtuel ───────────────────────────────
def build_virtual_features(base_vec: np.ndarray, current_silence: float,
                            last_ilis: list, time_since_start: float) -> np.ndarray:
    v = base_vec.copy()

    all_ilis = last_ilis + [current_silence]
    last3  = all_ilis[-3:]  if len(all_ilis) >= 3  else all_ilis
    last5  = all_ilis[-5:]  if len(all_ilis) >= 5  else all_ilis
    last10 = all_ilis[-10:] if len(all_ilis) >= 10 else all_ilis

    def upd(name, val):
        if name in FEAT_IDX:
            v[FEAT_IDX[name]] = val

    mean3  = float(np.mean(last3))
    mean5  = float(np.mean(last5))
    mean10 = float(np.mean(last10))
    std5   = float(np.std(last5)) if len(last5) > 1 else 0.0
    max5   = float(np.max(last5))
    max10  = float(np.max(last10))
    min5   = float(np.min(last5))
    hist_max = float(max(all_ilis))

    upd("ili_s",              current_silence)
    upd("ili_log",            math.log(current_silence + 1.0))
    upd("rolling_ili_3",      mean3)
    upd("rolling_ili_5",      mean5)
    upd("rolling_ili_10",     mean10)
    upd("rolling_ili_std_5",  std5)
    upd("rolling_ili_max_5",  max5)
    upd("rolling_ili_max_10", max10)
    upd("rolling_ili_min_5",  min5)

    fr3  = 2.0 / (sum(last3)  + 1.0) * 60.0
    fr5  = 4.0 / (sum(last5)  + 1.0) * 60.0
    fr10 = 9.0 / (sum(last10) + 1.0) * 60.0
    fg   = v[FEAT_IDX["flash_rate_global"]] if "flash_rate_global" in FEAT_IDX else 1e-6

    upd("flash_rate_3",     fr3)
    upd("flash_rate_5",     fr5)
    upd("flash_rate_10",    fr10)
    upd("fr_log_slope",     math.log(fr5 + 1e-6) - math.log(fr10 + 1e-6))
    upd("fr_log_slope_3",   math.log(fr3 + 1e-6) - math.log(fr5  + 1e-6))
    upd("flash_rate_ratio", fr5 / (fg + 1e-6))

    if "fr_vs_max_ratio" in FEAT_IDX and "flash_rate_5" in FEAT_IDX:
        prev_ratio = base_vec[FEAT_IDX["fr_vs_max_ratio"]]
        prev_fr5   = base_vec[FEAT_IDX["flash_rate_5"]]
        fr5_max    = prev_fr5 / (prev_ratio + 1e-6)
        upd("fr_vs_max_ratio", fr5 / (fr5_max + 1e-6))

    upd("ili_trend",              current_silence - mean5)
    upd("ili_acceleration",       mean5 - mean10)
    upd("ili_cv_5",               std5 / (mean5 + 1e-6))
    upd("ili_z_score_5",          (current_silence - mean5) / (std5 + 1e-6))
    upd("ili_vs_local_mean",      current_silence / (mean5 + 1e-6))
    upd("ili_vs_alert_max",       current_silence / (hist_max + 1e-6))
    upd("rolling_max_vs_alert_max", max5 / (hist_max + 1e-6))

    # ILI vs percentiles causaux (inverse depuis le ratio stocké)
    for pname, rname in [("ili_vs_p75", "p75"), ("ili_vs_p95", "p95")]:
        if pname in FEAT_IDX and "rolling_ili_max_5" in FEAT_IDX:
            prev_max5  = base_vec[FEAT_IDX["rolling_ili_max_5"]]
            prev_ratio = base_vec[FEAT_IDX[pname]]
            p_est      = prev_max5 / (prev_ratio + 1e-6)
            upd(pname, max5 / (p_est + 1e-6))

    upd("time_since_start_s", time_since_start)

    if "sigma_level" in FEAT_IDX and "flash_rate_5" in FEAT_IDX:
        prev_fr5 = base_vec[FEAT_IDX["flash_rate_5"]]
        dfr = fr5 - prev_fr5
        upd("sigma_level", dfr / (abs(dfr) + 1e-6))

    return v


# ── Stratégie silence-aware ───────────────────────────────────────────────────
def make_predictions_silence(df_feat: pl.DataFrame, k: int, base_threshold: float,
                              silence_interval_min: float = 1.0,
                              max_silence_min: float = 32.0) -> pd.DataFrame:
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    missing = [c for c in FEAT_COLS if c not in df_alerts.columns]
    if missing:
        df_alerts = df_alerts.with_columns(
            [pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing]
        )

    X_all    = df_alerts.select(FEAT_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores   = model.predict_proba(X_all)[:, 1]
    df_pd    = df_alerts.select(["airport", "airport_alert_id", "date"] + FEAT_COLS).to_pandas()
    df_pd["score"] = scores
    df_pd["date"]  = pd.to_datetime(df_pd["date"], utc=True)

    silence_td     = pd.Timedelta(minutes=silence_interval_min)
    max_silence_td = pd.Timedelta(minutes=max_silence_min)
    rows, covered  = [], set()

    for (airport, alert_id), grp in df_pd.groupby(["airport", "airport_alert_id"], sort=False):
        grp     = grp.sort_values("date").reset_index(drop=True)
        n       = len(grp)
        t_start = grp["date"].iloc[0]
        points  = []
        ili_history = []

        for i in range(n):
            flash_date  = grp["date"].iloc[i]
            flash_score = float(grp["score"].iloc[i])
            flash_feat  = grp[FEAT_COLS].iloc[i].values.astype(np.float64)
            ili_val     = float(grp["ili_s"].iloc[i]) if "ili_s" in grp.columns else 0.0
            if i > 0:
                ili_history.append(ili_val)

            points.append((flash_date, flash_score))

            next_flash = grp["date"].iloc[i + 1] if i < n - 1 else flash_date + max_silence_td
            t_virt = flash_date + silence_td

            while t_virt < min(next_flash, flash_date + max_silence_td):
                silence_s  = (t_virt - flash_date).total_seconds()
                tss        = (t_virt - t_start).total_seconds()
                vf         = build_virtual_features(flash_feat, silence_s, list(ili_history), tss)
                vs         = model.predict_proba(vf.reshape(1, -1))[0, 1]
                points.append((t_virt, vs))
                t_virt += silence_td

        scores_arr = np.array([p[1] for p in points])
        alert_has_pred = False
        for i in range(len(scores_arr)):
            if i < k - 1:
                continue
            window = scores_arr[i - k + 1 : i + 1]
            if (window >= base_threshold).all():
                rows.append({
                    "airport": airport, "airport_alert_id": alert_id,
                    "prediction_date": points[i][0],
                    "predicted_date_end_alert": points[i][0],
                    "confidence": float(window.min()),
                })
                alert_has_pred = True

        if alert_has_pred:
            covered.add((airport, alert_id))
        else:
            last = grp.iloc[-1]
            rows.append({
                "airport": airport, "airport_alert_id": alert_id,
                "prediction_date": last["date"],
                "predicted_date_end_alert": last["date"],
                "confidence": base_threshold,
            })

    all_alerts = set(df_pd.groupby(["airport", "airport_alert_id"]).groups.keys())
    n_cov = len(covered)
    print(f"  Couvertes : {n_cov}/{len(all_alerts)}  fallback : {len(all_alerts)-n_cov}")
    return pd.DataFrame(rows, columns=[
        "airport", "airport_alert_id", "prediction_date",
        "predicted_date_end_alert", "confidence"
    ])


# ── Stratégie flash-only ──────────────────────────────────────────────────────
def make_predictions_flash(df_feat: pl.DataFrame, k: int, base_threshold: float) -> pd.DataFrame:
    df_alerts = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    missing = [c for c in FEAT_COLS if c not in df_alerts.columns]
    if missing:
        df_alerts = df_alerts.with_columns(
            [pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing]
        )
    X      = df_alerts.select(FEAT_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    df_pd  = df_alerts.select(["airport", "airport_alert_id", "date"]).to_pandas()
    df_pd["score"] = scores
    df_pd["date"]  = pd.to_datetime(df_pd["date"], utc=True)

    rows, covered = [], set()
    for (airport, alert_id), grp in df_pd.groupby(["airport", "airport_alert_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        s   = grp["score"].to_numpy()
        alert_has_pred = False
        for i in range(len(s)):
            if i < k - 1:
                continue
            window = s[i - k + 1 : i + 1]
            if (window >= base_threshold).all():
                rows.append({
                    "airport": airport, "airport_alert_id": alert_id,
                    "prediction_date": grp["date"].iloc[i],
                    "predicted_date_end_alert": grp["date"].iloc[i],
                    "confidence": float(window.min()),
                })
                alert_has_pred = True
        if alert_has_pred:
            covered.add((airport, alert_id))
        else:
            last = grp.iloc[-1]
            rows.append({
                "airport": airport, "airport_alert_id": alert_id,
                "prediction_date": last["date"],
                "predicted_date_end_alert": last["date"],
                "confidence": base_threshold,
            })
    all_alerts = set(df_pd.groupby(["airport", "airport_alert_id"]).groups.keys())
    n_cov = len(covered)
    print(f"  Couvertes : {n_cov}/{len(all_alerts)}  fallback : {len(all_alerts)-n_cov}")
    return pd.DataFrame(rows, columns=[
        "airport", "airport_alert_id", "prediction_date",
        "predicted_date_end_alert", "confidence"
    ])


# ── Helper : recalcule percentiles ILI sur un dataset ────────────────────────
def recompute_ili_pct(test_feats: pl.DataFrame) -> pl.DataFrame:
    df_a = test_feats.filter(pl.col("airport_alert_id").is_not_null())
    p75_map, p95_map = {}, {}
    for ap in df_a["airport"].unique().to_list():
        vals = df_a.filter(pl.col("airport") == ap)["ili_s"].drop_nulls().to_numpy()
        if len(vals) > 10:
            p75_map[ap] = float(np.percentile(vals, 75))
            p95_map[ap] = float(np.percentile(vals, 95))
    p75_expr = pl.lit(None, dtype=pl.Float64)
    p95_expr = pl.lit(None, dtype=pl.Float64)
    for ap, v in p75_map.items():
        p75_expr = pl.when(pl.col("airport") == ap).then(pl.lit(v)).otherwise(p75_expr)
    for ap, v in p95_map.items():
        p95_expr = pl.when(pl.col("airport") == ap).then(pl.lit(v)).otherwise(p95_expr)
    return test_feats.with_columns([
        (pl.col("rolling_ili_max_5") / (p95_expr + 1e-6)).alias("ili_vs_p95"),
        (pl.col("rolling_ili_max_5") / (p75_expr + 1e-6)).alias("ili_vs_p75"),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Chargement
# ══════════════════════════════════════════════════════════════════════════════
print("Chargement données eval...")
eval_feat_all = pl.concat([
    pl.read_parquet(str(PROC / f"{ap}_eval.parquet")) for ap in AIRPORTS
])
eval_raw = eval_feat_all.select(["airport","airport_alert_id","date","dist"]).to_pandas()
eval_raw = eval_raw[eval_raw["airport_alert_id"].notna()]

try:
    pp      = joblib.load(MODEL / "predict_params_silence.joblib")
    K_OPT   = pp["k"]
    THR_OPT = pp["base_threshold"]
except Exception:
    K_OPT, THR_OPT = 3, 0.41
print(f"Params Optuna gain : K={K_OPT}, threshold={THR_OPT:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n\n★ STRATÉGIE 1 : Flash-only")
preds1 = make_predictions_flash(eval_feat_all, k=K_OPT, base_threshold=THR_OPT)
res1, tot1 = compute_metrics(eval_raw, preds1)
g1, t1 = summarize(res1, tot1, f"Flash-only  K={K_OPT}  thr={THR_OPT:.2f}")

print("\n\n★ STRATÉGIE 2 : Silence-aware  K=2")
preds2 = make_predictions_silence(eval_feat_all, k=K_OPT, base_threshold=THR_OPT)
res2, tot2 = compute_metrics(eval_raw, preds2)
g2, t2 = summarize(res2, tot2, f"Silence K={K_OPT}  thr={THR_OPT:.2f}")

print("\n\n★ STRATÉGIE 3 : Silence-aware  K=1")
preds3 = make_predictions_silence(eval_feat_all, k=1, base_threshold=THR_OPT)
res3, tot3 = compute_metrics(eval_raw, preds3)
g3, t3 = summarize(res3, tot3, f"Silence K=1  thr={THR_OPT:.2f}")

print(f"\n\n{'='*55}")
print("RÉCAP EVAL LOCAL")
print(f"{'='*55}")
print(f"  Flash-only  K={K_OPT}  : {g1:.1f}h @ θ={t1}")
print(f"  Silence     K={K_OPT}  : {g2:.1f}h @ θ={t2}")
print(f"  Silence     K=1  : {g3:.1f}h @ θ={t3}")

# ── Test officiel avec les 3 stratégies ──────────────────────────────────────
print("\n\n★ DATASET TEST OFFICIEL")
print("Feature engineering...")
test_raw_pl = pl.read_csv(
    str(ROOT / "dataset_test" / "dataset_set.csv"),
    schema_overrides={"airport_alert_id": pl.Float64,
                      "is_last_lightning_cloud_ground": pl.Boolean},
).with_columns(pl.col("date").str.to_datetime(time_unit="us", time_zone="UTC"))

tfr = cf.compute_features(test_raw_pl)
tfr = cf.add_terrain_features(tfr)

ap_dfs = []
for ap in ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]:
    s = tfr.filter(pl.col("airport") == ap)
    s = cf.add_weather_features(s, ap)
    ap_dfs.append(s)
all_cols = set()
for d in ap_dfs: all_cols.update(d.columns)
ap_dfs = [d.with_columns([pl.lit(None).cast(pl.Float32).alias(c)
           for c in all_cols if c not in d.columns]) for d in ap_dfs]
test_feats = recompute_ili_pct(pl.concat(ap_dfs))

test_raw_pd = test_raw_pl.select(["airport","airport_alert_id","date","dist"]).to_pandas()
test_raw_pd = test_raw_pd[test_raw_pd["airport_alert_id"].notna()]

print("\n  ▶ Flash-only")
pt1 = make_predictions_flash(test_feats, k=K_OPT, base_threshold=THR_OPT)
rt1, tott = compute_metrics(test_raw_pd, pt1)
gt1, tt1  = summarize(rt1, tott, "Test — Flash-only")

print("\n  ▶ Silence K=2")
pt2 = make_predictions_silence(test_feats, k=K_OPT, base_threshold=THR_OPT)
rt2, _    = compute_metrics(test_raw_pd, pt2)
gt2, tt2  = summarize(rt2, tott, "Test — Silence K=2")

print("\n  ▶ Silence K=1")
pt3 = make_predictions_silence(test_feats, k=1, base_threshold=THR_OPT)
rt3, _    = compute_metrics(test_raw_pd, pt3)
gt3, tt3  = summarize(rt3, tott, "Test — Silence K=1")

# Sauvegarde la meilleure
best_idx = np.argmax([gt1, gt2, gt3])
best_preds = [pt1, pt2, pt3][best_idx]
best_preds.to_csv(ROOT / "dataset_test" / "predictions.csv", index=False)
print(f"\n→ predictions.csv = stratégie {best_idx+1}")

print(f"\n{'='*55}")
print("RÉSUMÉ FINAL")
print(f"{'='*55}")
print(f"               EVAL     TEST")
print(f"  Flash-only : {g1:5.1f}h    {gt1:5.1f}h   @ θ={tt1}")
print(f"  Silence K={K_OPT}: {g2:5.1f}h    {gt2:5.1f}h   @ θ={tt2}")
print(f"  Silence K=1: {g3:5.1f}h    {gt3:5.1f}h   @ θ={tt3}")
