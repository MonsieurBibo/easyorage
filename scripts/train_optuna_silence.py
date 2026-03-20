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
train_optuna_silence.py
-----------------------
Entraîne XGBoost avec des points virtuels de silence dans le training set.

Pour chaque alerte, on génère des points toutes les minutes :
- Entre deux éclairs (gap > 60s) → label = 0 (alerte encore active)
- Après le dernier éclair (jusqu'à 32min) → label = 1 (alerte terminée)

Cela permet au modèle de prédire la fin d'alerte même sans nouvel éclair,
en utilisant le silence croissant comme signal.

Optimise directement le Gain G (métrique officielle) avec Optuna.
Résultat : models/xgb_silence.joblib + models/predict_params_silence.joblib
"""

import math
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

N_TRIALS        = 60
R_ACCEPT        = 0.02
MAX_GAP_MIN     = 30
MIN_DIST_KM     = 3
SILENCE_INT_MIN = 1.0      # 1 point virtuel par minute
MAX_SILENCE_MIN = 32.0     # max silence après dernier éclair
AIRPORTS        = ["ajaccio", "bastia", "biarritz", "nantes", "pise"]
THETAS          = np.round(np.linspace(0.05, 0.95, 19), 2).tolist()

meta       = json.loads((PROC / "feature_cols.json").read_text())
FEAT_COLS  = meta["feature_cols"]
TARGET_COL = meta["target_col"]
FEAT_IDX   = {f: i for i, f in enumerate(FEAT_COLS)}


# ── Reconstruction features pour point virtuel ────────────────────────────────
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

    mean3 = float(np.mean(last3)); mean5 = float(np.mean(last5))
    mean10= float(np.mean(last10)); std5 = float(np.std(last5)) if len(last5)>1 else 0.0
    max5  = float(np.max(last5));  max10 = float(np.max(last10))
    min5  = float(np.min(last5));  hist_max = float(max(all_ilis))

    upd("ili_s", current_silence); upd("ili_log", math.log(current_silence+1.0))
    upd("rolling_ili_3", mean3);   upd("rolling_ili_5", mean5)
    upd("rolling_ili_10", mean10); upd("rolling_ili_std_5", std5)
    upd("rolling_ili_max_5", max5); upd("rolling_ili_max_10", max10)
    upd("rolling_ili_min_5", min5)

    fr3  = 2.0/(sum(last3)+1.0)*60; fr5  = 4.0/(sum(last5)+1.0)*60
    fr10 = 9.0/(sum(last10)+1.0)*60
    fg   = v[FEAT_IDX["flash_rate_global"]] if "flash_rate_global" in FEAT_IDX else 1e-6

    upd("flash_rate_3",fr3); upd("flash_rate_5",fr5); upd("flash_rate_10",fr10)
    upd("fr_log_slope",   math.log(fr5+1e-6)-math.log(fr10+1e-6))
    upd("fr_log_slope_3", math.log(fr3+1e-6)-math.log(fr5+1e-6))
    upd("flash_rate_ratio", fr5/(fg+1e-6))

    if "fr_vs_max_ratio" in FEAT_IDX and "flash_rate_5" in FEAT_IDX:
        prev_r = base_vec[FEAT_IDX["fr_vs_max_ratio"]]
        prev_f = base_vec[FEAT_IDX["flash_rate_5"]]
        fr5_max = prev_f/(prev_r+1e-6)
        upd("fr_vs_max_ratio", fr5/(fr5_max+1e-6))

    upd("ili_trend",              current_silence-mean5)
    upd("ili_acceleration",       mean5-mean10)
    upd("ili_cv_5",               std5/(mean5+1e-6))
    upd("ili_z_score_5",          (current_silence-mean5)/(std5+1e-6))
    upd("ili_vs_local_mean",      current_silence/(mean5+1e-6))
    upd("ili_vs_alert_max",       current_silence/(hist_max+1e-6))
    upd("rolling_max_vs_alert_max", max5/(hist_max+1e-6))

    for pname in ["ili_vs_p75", "ili_vs_p95"]:
        if pname in FEAT_IDX and "rolling_ili_max_5" in FEAT_IDX:
            pm5 = base_vec[FEAT_IDX["rolling_ili_max_5"]]
            pr  = base_vec[FEAT_IDX[pname]]
            p_est = pm5/(pr+1e-6)
            upd(pname, max5/(p_est+1e-6))

    upd("time_since_start_s", time_since_start)

    if "sigma_level" in FEAT_IDX and "flash_rate_5" in FEAT_IDX:
        prev_fr5 = base_vec[FEAT_IDX["flash_rate_5"]]
        dfr = fr5 - prev_fr5
        upd("sigma_level", dfr/(abs(dfr)+1e-6))

    return v


# ── Augmentation d'un split avec points virtuels ─────────────────────────────
def augment_with_silence(df: pl.DataFrame, split_label: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne (X_augmented, y_augmented) incluant les points virtuels.
    - Entre flash i et i+1 (gap > 60s) : label = 0
    - Après le dernier flash (jusqu'à MAX_SILENCE_MIN) : label = 1
    """
    df_alerts = df.filter(pl.col("airport_alert_id").is_not_null())
    X_real = df_alerts.select(FEAT_COLS).fill_nan(0).fill_null(0).to_numpy()
    y_real = df_alerts[TARGET_COL].cast(pl.Int8).to_numpy()

    df_pd  = df_alerts.select(["airport","airport_alert_id","date"] + FEAT_COLS).to_pandas()
    df_pd["date"] = pd.to_datetime(df_pd["date"], utc=True)
    df_pd["_y"]   = y_real

    silence_td     = pd.Timedelta(minutes=SILENCE_INT_MIN)
    max_silence_td = pd.Timedelta(minutes=MAX_SILENCE_MIN)

    X_virt, y_virt = [], []
    n_pos_virt, n_neg_virt = 0, 0

    for (airport, alert_id), grp in df_pd.groupby(["airport","airport_alert_id"], sort=False):
        grp     = grp.sort_values("date").reset_index(drop=True)
        n       = len(grp)
        t_start = grp["date"].iloc[0]
        ili_hist = []

        for i in range(n):
            flash_date = grp["date"].iloc[i]
            flash_feat = grp[FEAT_COLS].iloc[i].values.astype(np.float64)
            ili_val    = float(grp["ili_s"].iloc[i]) if "ili_s" in grp.columns else 0.0
            if i > 0:
                ili_hist.append(ili_val)

            is_last = bool(grp["_y"].iloc[i])
            if is_last:
                # Après le dernier éclair → points virtuels label=1
                next_wall = flash_date + max_silence_td
                label_virt = 1
            else:
                # Entre flash i et i+1 → points virtuels label=0
                next_wall = grp["date"].iloc[i+1]
                label_virt = 0

            t_virt = flash_date + silence_td
            while t_virt < min(next_wall, flash_date + max_silence_td):
                s_s  = (t_virt - flash_date).total_seconds()
                tss  = (t_virt - t_start).total_seconds()
                vf   = build_virtual_features(flash_feat, s_s, list(ili_hist), tss)
                X_virt.append(vf)
                y_virt.append(label_virt)
                if label_virt == 1: n_pos_virt += 1
                else: n_neg_virt += 1
                t_virt += silence_td

    print(f"  [{split_label}] réels={len(X_real)} · virtuels={len(X_virt)}"
          f" (pos={n_pos_virt}, neg={n_neg_virt})")

    if X_virt:
        X_out = np.vstack([X_real, np.array(X_virt)])
        y_out = np.concatenate([y_real, np.array(y_virt, dtype=np.int8)])
    else:
        X_out, y_out = X_real, y_real

    return X_out.astype(np.float32), y_out


# ── Chargement + augmentation données ────────────────────────────────────────
print("Augmentation des données avec points de silence...")
splits = {}
for ap in AIRPORTS:
    splits[ap] = {
        "train": pl.read_parquet(str(PROC / f"{ap}_train.parquet")),
        "eval":  pl.read_parquet(str(PROC / f"{ap}_eval.parquet")),
    }

X_tr_parts, y_tr_parts = [], []
for ap in AIRPORTS:
    X, y = augment_with_silence(splits[ap]["train"], f"train/{ap}")
    X_tr_parts.append(X); y_tr_parts.append(y)

X_tr = np.vstack(X_tr_parts)
y_tr = np.concatenate(y_tr_parts)

n_pos = y_tr.sum(); n_neg = len(y_tr)-n_pos
pos_weight = n_neg/max(n_pos,1)
print(f"\nDataset train augmenté : {len(X_tr):,} points · pos_weight={pos_weight:.1f}")

# Eval augmenté (pour métriques Gain)
eval_feat_all = pl.concat([splits[ap]["eval"] for ap in AIRPORTS])
eval_raw = eval_feat_all.select(["airport","airport_alert_id","date","dist"]).to_pandas()
eval_raw = eval_raw[eval_raw["airport_alert_id"].notna()]
eval_raw["date"] = pd.to_datetime(eval_raw["date"], utc=True)
tot_dangerous = len(eval_raw[eval_raw["dist"] < MIN_DIST_KM])
alerts_grouped = eval_raw.groupby(["airport","airport_alert_id"])


# ── Prédictions silence-aware ─────────────────────────────────────────────────
def make_preds_silence(m, df_feat: pl.DataFrame, k: int, thr: float) -> pd.DataFrame:
    df_a = df_feat.filter(pl.col("airport_alert_id").is_not_null())
    missing = [c for c in FEAT_COLS if c not in df_a.columns]
    if missing:
        df_a = df_a.with_columns([pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing])
    X_all  = df_a.select(FEAT_COLS).fill_nan(0).fill_null(0).to_numpy()
    scores = m.predict_proba(X_all)[:,1]
    df_pd  = df_a.select(["airport","airport_alert_id","date"]+FEAT_COLS).to_pandas()
    df_pd["score"] = scores
    df_pd["date"]  = pd.to_datetime(df_pd["date"], utc=True)

    silence_td = pd.Timedelta(minutes=SILENCE_INT_MIN)
    max_sil_td = pd.Timedelta(minutes=MAX_SILENCE_MIN)
    rows, covered = [], set()

    for (airport, alert_id), grp in df_pd.groupby(["airport","airport_alert_id"], sort=False):
        grp    = grp.sort_values("date").reset_index(drop=True)
        n      = len(grp)
        t_start= grp["date"].iloc[0]
        points = []
        ili_hist = []

        for i in range(n):
            fd  = grp["date"].iloc[i]
            fs  = float(grp["score"].iloc[i])
            ff  = grp[FEAT_COLS].iloc[i].values.astype(np.float64)
            ili = float(grp["ili_s"].iloc[i]) if "ili_s" in grp.columns else 0.0
            if i > 0: ili_hist.append(ili)
            points.append((fd, fs))

            nf  = grp["date"].iloc[i+1] if i<n-1 else fd+max_sil_td
            tv  = fd + silence_td
            while tv < min(nf, fd+max_sil_td):
                ss = (tv-fd).total_seconds()
                tss= (tv-t_start).total_seconds()
                vf = build_virtual_features(ff, ss, list(ili_hist), tss)
                vs = m.predict_proba(vf.reshape(1,-1))[0,1]
                points.append((tv, vs))
                tv += silence_td

        sa = np.array([p[1] for p in points])
        has = False
        for i in range(len(sa)):
            if i < k-1: continue
            w = sa[i-k+1:i+1]
            if (w >= thr).all():
                rows.append({"airport":airport,"airport_alert_id":alert_id,
                             "prediction_date":points[i][0],
                             "predicted_date_end_alert":points[i][0],
                             "confidence":float(w.min())})
                has = True
        if has:
            covered.add((airport,alert_id))
        else:
            last=grp.iloc[-1]
            rows.append({"airport":airport,"airport_alert_id":alert_id,
                         "prediction_date":last["date"],
                         "predicted_date_end_alert":last["date"],
                         "confidence":thr})

    all_a = set(df_pd.groupby(["airport","airport_alert_id"]).groups.keys())
    print(f"  Couvertes : {len(covered)}/{len(all_a)}  fallback: {len(all_a)-len(covered)}")
    return pd.DataFrame(rows, columns=["airport","airport_alert_id","prediction_date",
                                        "predicted_date_end_alert","confidence"])


def best_gain_eval(m, k, thr):
    preds = make_preds_silence(m, eval_feat_all, k, thr)
    preds["predicted_date_end_alert"] = pd.to_datetime(
        preds["predicted_date_end_alert"], utc=True
    )
    best = 0.0
    for theta in THETAS:
        over = preds[preds["confidence"] >= theta]
        if over.empty: continue
        pred_min = over.groupby(["airport","airport_alert_id"])["predicted_date_end_alert"].min()
        gain, missed = 0, 0
        for (airport, alert_id), end_pred in pred_min.items():
            try: lts = alerts_grouped.get_group((airport, alert_id))
            except KeyError: continue
            end_base = pd.to_datetime(lts["date"],utc=True).max()+pd.Timedelta(minutes=MAX_GAP_MIN)
            gain  += (end_base - end_pred).total_seconds()
            close  = pd.to_datetime(lts[lts["dist"]<MIN_DIST_KM]["date"],utc=True)
            missed += int((close > end_pred).sum())
        r = missed/max(tot_dangerous,1)
        if r < R_ACCEPT:
            best = max(best, gain/3600)
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
    k   = trial.suggest_int("k", 1, 4)
    thr = trial.suggest_float("base_threshold", 0.15, 0.6)

    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr)
    return best_gain_eval(m, k, thr)


# ── Étude Optuna ──────────────────────────────────────────────────────────────
storage = f"sqlite:///{MODEL}/optuna_silence.db"
study = optuna.create_study(study_name="xgb_silence", storage=storage,
                             direction="maximize", load_if_exists=True)
n_done = len(study.trials)
n_rem  = max(0, N_TRIALS - n_done)
print(f"\nTrials : {n_done}/{N_TRIALS} complétés")

if n_rem == 0:
    print("Étude déjà complète.")
else:
    print(f"Lancement de {n_rem} trial(s)...")
    study.optimize(objective, n_trials=n_rem, show_progress_bar=True)

# ── Résultats ─────────────────────────────────────────────────────────────────
print(f"\n=== Meilleur trial ===")
print(f"Gain = {study.best_value:.1f}h")
print(f"Params : {study.best_params}")

bp = study.best_params.copy()
k_best   = bp.pop("k")
thr_best = bp.pop("base_threshold")

best_model = xgb.XGBClassifier(**bp, scale_pos_weight=pos_weight, verbosity=0, random_state=42)
best_model.fit(X_tr, y_tr)

joblib.dump(best_model, MODEL / "xgb_silence.joblib")
joblib.dump({"k": k_best, "base_threshold": thr_best}, MODEL / "predict_params_silence.joblib")
print(f"\nModèle → models/xgb_silence.joblib")
print(f"K={k_best}, threshold={thr_best:.2f}")
print(f"Étude  → models/optuna_silence.db")
