# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "numpy",
#     "scikit-learn",
#     "xgboost",
#     "lightgbm",
#     "optuna",
#     "altair",
#     "pyarrow==23.0.1",
#     "lifelines",
#     "shap",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    import json, pathlib, warnings
    warnings.filterwarnings("ignore")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, ExtraTreesClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve,
    )
    import xgboost as xgb
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        mo, pl, np, alt, json, pathlib, warnings,
        LogisticRegression, RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, ExtraTreesClassifier, DecisionTreeClassifier,
        SVC, KNeighborsClassifier, MLPClassifier,
        StandardScaler, Pipeline, compute_sample_weight,
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
        xgb, lgb, optuna,
    )


@app.cell
def load_data(pl, json, pathlib, mo):
    """Charge les parquets pré-calculés par compute_features.py"""
    ROOT = pathlib.Path(__file__).parent.parent
    PROC = ROOT / "data" / "processed"

    meta = json.loads((PROC / "feature_cols.json").read_text())
    FEATURE_COLS = meta["feature_cols"]
    LIGHTNING_COLS = meta["lightning_feature_cols"]
    TERRAIN_COLS = meta["terrain_feature_cols"]
    WEATHER_COLS = meta["weather_feature_cols"]
    TARGET_COL = meta["target_col"]

    AIRPORTS = ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]

    splits = {}
    for _ap in AIRPORTS:
        _key = _ap.lower()
        _train = pl.read_parquet(str(PROC / f"{_key}_train.parquet"))
        _eval  = pl.read_parquet(str(PROC / f"{_key}_eval.parquet"))
        splits[_ap] = {"train": _train, "eval": _eval}

    mo.output.replace(mo.md(
        f"**{len(FEATURE_COLS)} features** chargées "
        f"({len(LIGHTNING_COLS)} lightning + {len(TERRAIN_COLS)} terrain + {len(WEATHER_COLS)} météo) · "
        f"**{sum(len(s['train']) + len(s['eval']) for s in splits.values()):,}** éclairs CG"
    ))
    return FEATURE_COLS, LIGHTNING_COLS, TERRAIN_COLS, WEATHER_COLS, TARGET_COL, AIRPORTS, splits


# ─────────────────────────────────────────────────────────────────────────────
# EDA STATS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def eda_alert_stats(splits, pl, alt, mo):
    """Stats descriptives et graphes par aéroport."""
    rows = []
    dur_rows = []
    for _ap, _s in splits.items():
        _all = pl.concat([_s["train"], _s["eval"]])
        _alert_stats = (
            _all.group_by("airport", "airport_alert_id")
            .agg(
                pl.len().alias("n_cg"),
                pl.col("date").min().alias("debut"),
                pl.col("date").max().alias("fin"),
                pl.col("amplitude_abs").mean().alias("amp_mean"),
                pl.col("dist").mean().alias("dist_mean"),
            )
            .with_columns(
                ((pl.col("fin") - pl.col("debut")).dt.total_seconds() / 60).alias("duree_min")
            )
        )
        rows.append({
            "Aéroport": _ap,
            "N alertes": len(_alert_stats),
            "N éclairs CG": len(_all),
            "Durée médiane (min)": round(_alert_stats["duree_min"].median(), 1),
            "Durée max (min)": round(_alert_stats["duree_min"].max(), 1),
            "Éclairs/alerte (médiane)": round(_alert_stats["n_cg"].median(), 1),
            "% 1-éclair": round(100 * (_alert_stats["n_cg"] == 1).sum() / len(_alert_stats), 1),
        })
        for _r in _alert_stats.iter_rows(named=True):
            dur_rows.append({"airport": _ap, "duree_min": min(_r["duree_min"], 180)})

    stats_df = pl.DataFrame(rows)
    dur_df = pl.DataFrame(dur_rows)

    _chart_dur = (
        alt.Chart(dur_df.to_pandas())
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("duree_min:Q", bin=alt.Bin(maxbins=40), title="Durée alerte (min, cap 180)"),
            y=alt.Y("count():Q", title="N alertes", stack=None),
            color=alt.Color("airport:N", title="Aéroport"),
        )
        .properties(width=700, height=250, title="Distribution des durées d'alerte par aéroport")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Stats descriptives par aéroport"),
        stats_df,
        _chart_dur,
    ]))
    return (stats_df,)


@app.cell
def eda_feature_distributions(splits, pl, alt, mo, np):
    """Distributions des features clés : ILI, amplitude, dist — séparées positifs vs négatifs."""
    _all_rows = []
    for _ap, _s in splits.items():
        _df = pl.concat([_s["train"], _s["eval"]]).sample(n=min(3000, len(_s["train"])), seed=42)
        _all_rows.append(_df.with_columns(pl.lit(_ap).alias("airport")))
    _df_all = pl.concat(_all_rows)

    _TARGET = "is_last_lightning_cloud_ground"

    def _hist(col, title, cap=None):
        _d = _df_all.select(col, _TARGET, "airport").to_pandas()
        if cap:
            _d[col] = _d[col].clip(upper=cap)
        _d["target"] = _d[_TARGET].map({True: "Dernier éclair (positif)", False: "Autre éclair"})
        return (
            alt.Chart(_d)
            .mark_bar(opacity=0.6)
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40), title=title),
                y=alt.Y("count():Q", stack=None),
                color=alt.Color("target:N"),
            )
            .properties(width=380, height=220)
        )

    _charts = alt.hconcat(
        _hist("ili_s", "ILI (s, cap 600)", cap=600),
        _hist("amplitude_abs", "Amplitude |kA|"),
    ) & alt.hconcat(
        _hist("dist", "Distance aéroport (km)"),
        _hist("time_since_start_s", "Temps depuis début alerte (s)", cap=3600),
    ) & alt.hconcat(
        _hist("rolling_ili_5", "Rolling ILI 5 (s, cap 600)", cap=600),
        _hist("flash_rate_global", "Flash rate global (éclairs/min)", cap=10),
    )

    mo.output.replace(mo.vstack([
        mo.md("## Distributions des features — positifs (dernier éclair) vs négatifs"),
        mo.callout(mo.md(
            "Si les distributions divergent entre positifs et négatifs → la feature est discriminante."
        ), kind="info"),
        _charts,
    ]))
    return


@app.cell
def eda_correlation(splits, FEATURE_COLS, pl, alt, mo, np):
    """Corrélation des features avec la target."""
    _all = pl.concat([
        pl.concat([_s["train"], _s["eval"]]) for _s in splits.values()
    ])

    _TARGET = "is_last_lightning_cloud_ground"

    _corrs = []
    for _f in FEATURE_COLS:
        try:
            _c = _all.select(pl.corr(pl.col(_f), pl.col(_TARGET).cast(pl.Float64))).item()
            _corrs.append({"feature": _f, "correlation": round(_c, 4) if _c is not None else 0.0})
        except Exception:
            pass

    _corr_df = pl.DataFrame(_corrs).sort("correlation", descending=True)

    _chart = (
        alt.Chart(_corr_df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("correlation:Q", title="Corrélation de Pearson avec target"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.condition(
                alt.datum.correlation > 0,
                alt.value("#2196F3"),
                alt.value("#F44336"),
            ),
        )
        .properties(width=500, height=500, title="Corrélation features → is_last_lightning_cloud_ground")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Corrélation features / target"),
        _chart,
        _corr_df,
    ]))
    return


# ─────────────────────────────────────────────────────────────────────────────
# HELPER : evaluate a model
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def helpers(np, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score):
    def evaluate(y_true, y_proba, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        pod  = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        far  = 1.0 - prec if prec > 0 else 1.0
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_proba)
        ap   = average_precision_score(y_true, y_proba)
        return dict(POD=round(pod,3), FAR=round(far,3), F1=round(f1,3),
                    AUC=round(auc,3), AP=round(ap,3))

    def pos_weight(y):
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        return n_neg / max(n_pos, 1)

    return evaluate, pos_weight


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLES CLASSIQUES (sans tuning — baseline rapide)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def baseline_models(
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS, evaluate, pos_weight,
    np, pl, mo,
    LogisticRegression, RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, DecisionTreeClassifier,
    KNeighborsClassifier, MLPClassifier,
    StandardScaler, Pipeline, compute_sample_weight,
    xgb, lgb,
):
    """
    Tous les algorithmes classiques avec hyperparamètres par défaut (baseline).
    Sert de référence avant le tuning Optuna.
    """
    def _make_models(pw):
        return {
            "LogReg": Pipeline([
                ("sc", StandardScaler()),
                ("m", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)),
            ]),
            "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
            "GradBoost": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
            "KNN": Pipeline([
                ("sc", StandardScaler()),
                ("m", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
            ]),
            "MLP": Pipeline([
                ("sc", StandardScaler()),
                ("m", MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=400, random_state=42, early_stopping=True)),
            ]),
            "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                           scale_pos_weight=pw, subsample=0.8, colsample_bytree=0.8,
                                           verbosity=0, random_state=42, eval_metric="logloss"),
            "LightGBM": lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,
                                             scale_pos_weight=pw, subsample=0.8,
                                             verbose=-1, random_state=42),
        }

    all_results = []

    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_tr = _s["train"].select(FEATURE_COLS).to_numpy()
        _y_tr = _s["train"][TARGET_COL].cast(pl.Int8).to_numpy()
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _pw   = pos_weight(_y_tr)
        _sw   = compute_sample_weight("balanced", _y_tr)

        for _name, _model in _make_models(_pw).items():
            try:
                if _name in ("AdaBoost",):
                    _model.fit(_X_tr, _y_tr, sample_weight=_sw)
                elif _name == "MLP":
                    _model.fit(_X_tr, _y_tr, **{"m__sample_weight": _sw})
                else:
                    _model.fit(_X_tr, _y_tr)
                _proba = _model.predict_proba(_X_ev)[:, 1]
                _m = evaluate(_y_ev, _proba)
                _m.update({"Aéroport": _ap, "Modèle": _name})
                all_results.append(_m)
            except Exception as _e:
                all_results.append({"Aéroport": _ap, "Modèle": _name, "POD": None, "FAR": None, "F1": None, "AUC": None, "AP": None, "error": str(_e)})

    _res_df = pl.DataFrame(all_results)
    baseline_results = _res_df

    # Table triée par AUC décroissant
    _display = _res_df.sort("AUC", descending=True, nulls_last=True)

    mo.output.replace(mo.vstack([
        mo.md("## Résultats baseline — tous algorithmes (threshold=0.5)"),
        mo.callout(mo.md("Hyperparamètres par défaut. Objectif : identifier les meilleurs candidats pour le tuning Optuna."), kind="info"),
        _display,
    ]))
    return (baseline_results,)


@app.cell
def baseline_chart(baseline_results, alt, mo):
    """Graphe AUC vs FAR par modèle et aéroport."""
    _df = baseline_results.drop_nulls(subset=["AUC", "FAR"]).to_pandas()

    _scatter = (
        alt.Chart(_df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("FAR:Q", scale=alt.Scale(domain=[0, 1]), title="FAR (↓ mieux)"),
            y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.5, 1.0]), title="AUC-ROC (↑ mieux)"),
            color=alt.Color("Modèle:N"),
            shape=alt.Shape("Aéroport:N"),
            tooltip=["Modèle", "Aéroport", "AUC", "FAR", "POD", "F1"],
        )
        .properties(width=600, height=400, title="AUC vs FAR — tous modèles et aéroports")
    )

    _bars = (
        alt.Chart(_df.groupby("Modèle")[["AUC", "FAR"]].mean().reset_index())
        .mark_bar()
        .encode(
            x=alt.X("Modèle:N", sort="-y"),
            y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.5, 1.0])),
            color=alt.Color("AUC:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Modèle", "AUC", "FAR"],
        )
        .properties(width=600, height=250, title="AUC moyen par modèle (toutes aéroports)")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Comparaison visuelle — AUC & FAR"),
        _scatter,
        _bars,
    ]))
    return


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA TUNING — XGBoost & LightGBM (meilleurs candidats)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def optuna_xgb(
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS, evaluate, pos_weight,
    np, pl, optuna, xgb, mo,
):
    """
    Tuning Optuna pour XGBoost — 40 trials par aéroport.
    Optimise AUC-ROC sur le set eval (pas de CV interne pour garder la lisibilité).
    Choix documenté : on optimise AUC car il est indépendant du threshold,
    et on choisira le threshold optimal séparément selon le compromis POD/FAR.
    """
    N_TRIALS = 40

    xgb_best = {}

    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_tr = _s["train"].select(FEATURE_COLS).to_numpy()
        _y_tr = _s["train"][TARGET_COL].cast(pl.Int8).to_numpy()
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _pw   = pos_weight(_y_tr)

        def _objective(trial):
            _params = dict(
                n_estimators    = trial.suggest_int("n_estimators", 100, 500),
                max_depth       = trial.suggest_int("max_depth", 3, 8),
                learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample       = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
                gamma           = trial.suggest_float("gamma", 0, 5),
                reg_alpha       = trial.suggest_float("reg_alpha", 0, 2),
                reg_lambda      = trial.suggest_float("reg_lambda", 0, 2),
                scale_pos_weight= _pw,
                verbosity=0, random_state=42, eval_metric="logloss",
            )
            _m = xgb.XGBClassifier(**_params)
            _m.fit(_X_tr, _y_tr)
            return roc_auc_score_(_y_ev, _m.predict_proba(_X_ev)[:, 1])

        from sklearn.metrics import roc_auc_score as roc_auc_score_
        _study = optuna.create_study(direction="maximize")
        _study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=False)

        _best = xgb.XGBClassifier(**_study.best_params,
                                    scale_pos_weight=_pw, verbosity=0, random_state=42)
        _best.fit(_X_tr, _y_tr)
        _m = evaluate(_y_ev, _best.predict_proba(_X_ev)[:, 1])
        xgb_best[_ap] = {"model": _best, "params": _study.best_params, "metrics": _m}

    _rows = [{"Aéroport": _ap, "Modèle": "XGBoost (tuned)", **_v["metrics"]}
             for _ap, _v in xgb_best.items()]

    mo.output.replace(mo.vstack([
        mo.md(f"## XGBoost — Optuna tuning ({N_TRIALS} trials/aéroport)"),
        pl.DataFrame(_rows),
    ]))
    return (xgb_best,)


@app.cell
def optuna_lgbm(
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS, evaluate, pos_weight,
    np, pl, optuna, lgb, mo,
):
    """Tuning Optuna pour LightGBM — 40 trials par aéroport."""
    N_TRIALS = 40

    lgb_best = {}

    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_tr = _s["train"].select(FEATURE_COLS).to_numpy()
        _y_tr = _s["train"][TARGET_COL].cast(pl.Int8).to_numpy()
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _pw   = pos_weight(_y_tr)

        def _objective(trial):
            _params = dict(
                n_estimators   = trial.suggest_int("n_estimators", 100, 600),
                num_leaves     = trial.suggest_int("num_leaves", 15, 127),
                learning_rate  = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample      = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
                reg_alpha      = trial.suggest_float("reg_alpha", 0, 2),
                reg_lambda     = trial.suggest_float("reg_lambda", 0, 2),
                scale_pos_weight= _pw,
                verbose=-1, random_state=42,
            )
            _m = lgb.LGBMClassifier(**_params)
            _m.fit(_X_tr, _y_tr)
            return roc_auc_score_(_y_ev, _m.predict_proba(_X_ev)[:, 1])

        from sklearn.metrics import roc_auc_score as roc_auc_score_
        _study = optuna.create_study(direction="maximize")
        _study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=False)

        _best = lgb.LGBMClassifier(**_study.best_params,
                                    scale_pos_weight=_pw, verbose=-1, random_state=42)
        _best.fit(_X_tr, _y_tr)
        _m = evaluate(_y_ev, _best.predict_proba(_X_ev)[:, 1])
        lgb_best[_ap] = {"model": _best, "params": _study.best_params, "metrics": _m}

    _rows = [{"Aéroport": _ap, "Modèle": "LightGBM (tuned)", **_v["metrics"]}
             for _ap, _v in lgb_best.items()]

    mo.output.replace(mo.vstack([
        mo.md(f"## LightGBM — Optuna tuning ({N_TRIALS} trials/aéroport)"),
        pl.DataFrame(_rows),
    ]))
    return (lgb_best,)


@app.cell
def optuna_rf(
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS, evaluate, pos_weight,
    np, pl, optuna, mo, RandomForestClassifier,
):
    """Tuning Optuna pour Random Forest — 30 trials par aéroport."""
    N_TRIALS = 30

    rf_best = {}

    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_tr = _s["train"].select(FEATURE_COLS).to_numpy()
        _y_tr = _s["train"][TARGET_COL].cast(pl.Int8).to_numpy()
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()

        def _objective(trial):
            _m = RandomForestClassifier(
                n_estimators    = trial.suggest_int("n_estimators", 100, 500),
                max_depth       = trial.suggest_int("max_depth", 4, 20),
                min_samples_split= trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
                max_features    = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                class_weight    = "balanced",
                n_jobs=-1, random_state=42,
            )
            _m.fit(_X_tr, _y_tr)
            return roc_auc_score_(_y_ev, _m.predict_proba(_X_ev)[:, 1])

        from sklearn.metrics import roc_auc_score as roc_auc_score_
        _study = optuna.create_study(direction="maximize")
        _study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=False)

        _best = RandomForestClassifier(**_study.best_params,
                                        class_weight="balanced", n_jobs=-1, random_state=42)
        _best.fit(_X_tr, _y_tr)
        _m = evaluate(_y_ev, _best.predict_proba(_X_ev)[:, 1])
        rf_best[_ap] = {"model": _best, "params": _study.best_params, "metrics": _m}

    _rows = [{"Aéroport": _ap, "Modèle": "RandomForest (tuned)", **_v["metrics"]}
             for _ap, _v in rf_best.items()]

    mo.output.replace(mo.vstack([
        mo.md(f"## Random Forest — Optuna tuning ({N_TRIALS} trials/aéroport)"),
        pl.DataFrame(_rows),
    ]))
    return (rf_best,)


@app.cell
def optuna_mlp(
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS, evaluate, pos_weight,
    np, pl, optuna, mo,
    MLPClassifier, StandardScaler, Pipeline, compute_sample_weight,
):
    """Tuning Optuna pour MLP — architecture et learning rate — 30 trials."""
    N_TRIALS = 30

    mlp_best = {}

    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_tr = _s["train"].select(FEATURE_COLS).to_numpy()
        _y_tr = _s["train"][TARGET_COL].cast(pl.Int8).to_numpy()
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _sw   = compute_sample_weight("balanced", _y_tr)
        _sc   = StandardScaler()
        _X_trs = _sc.fit_transform(_X_tr)
        _X_evs = _sc.transform(_X_ev)

        def _objective(trial):
            _n_layers = trial.suggest_int("n_layers", 1, 4)
            _sizes = tuple(
                trial.suggest_int(f"h{_i}", 16, 256) for _i in range(_n_layers)
            )
            _m = MLPClassifier(
                hidden_layer_sizes= _sizes,
                activation        = trial.suggest_categorical("activation", ["relu", "tanh"]),
                alpha             = trial.suggest_float("alpha", 1e-5, 0.1, log=True),
                learning_rate_init= trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                max_iter=300, random_state=42, early_stopping=True,
            )
            _m.fit(_X_trs, _y_tr, sample_weight=_sw)
            return roc_auc_score_(_y_ev, _m.predict_proba(_X_evs)[:, 1])

        from sklearn.metrics import roc_auc_score as roc_auc_score_
        _study = optuna.create_study(direction="maximize")
        _study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=False)

        _n_layers = _study.best_params["n_layers"]
        _sizes = tuple(_study.best_params[f"h{_i}"] for _i in range(_n_layers))
        _best = MLPClassifier(
            hidden_layer_sizes=_sizes,
            activation=_study.best_params["activation"],
            alpha=_study.best_params["alpha"],
            learning_rate_init=_study.best_params["lr"],
            max_iter=400, random_state=42, early_stopping=True,
        )
        _best.fit(_X_trs, _y_tr, sample_weight=_sw)
        _m = evaluate(_y_ev, _best.predict_proba(_X_evs)[:, 1])
        mlp_best[_ap] = {"model": _best, "scaler": _sc, "params": _study.best_params, "metrics": _m}

    _rows = [{"Aéroport": _ap, "Modèle": "MLP (tuned)", **_v["metrics"]}
             for _ap, _v in mlp_best.items()]
    mo.output.replace(mo.vstack([
        mo.md(f"## MLP — Optuna tuning ({N_TRIALS} trials/aéroport)"),
        pl.DataFrame(_rows),
    ]))
    return (mlp_best,)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARAISON FINALE + COURBES ROC
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def final_comparison(
    baseline_results, xgb_best, lgb_best, rf_best, mlp_best, _xgb_unified,
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS,
    pl, np, alt, mo, evaluate,
    roc_curve, precision_recall_curve,
):
    """Tableau récapitulatif + courbes ROC des meilleurs modèles."""
    # Collecte tous les tuned results
    _tuned_rows = []
    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        for _name, _store, _transform in [
            ("XGBoost (tuned)", xgb_best, lambda m, X, ap: m[ap]["model"].predict_proba(X)[:, 1]),
            ("LightGBM (tuned)", lgb_best, lambda m, X, ap: m[ap]["model"].predict_proba(X)[:, 1]),
            ("RandomForest (tuned)", rf_best, lambda m, X, ap: m[ap]["model"].predict_proba(X)[:, 1]),
            ("MLP (tuned)", mlp_best, lambda m, X, ap: m[ap]["model"].predict_proba(m[ap]["scaler"].transform(X))[:, 1]),
        ]:
            _proba = _transform(_store, _X_ev, _ap)
            _m = evaluate(_y_ev, _proba)
            _tuned_rows.append({"Aéroport": _ap, "Modèle": _name, **_m})

    # Ajoute le modèle unifié
    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_ev = _s["eval"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _proba = _xgb_unified.predict_proba(_X_ev)[:, 1]
        _m = evaluate(_y_ev, _proba)
        _tuned_rows.append({"Aéroport": _ap, "Modèle": "XGBoost Unifié", **_m})

    # Union baseline + tuned
    _all = pl.concat([
        baseline_results.drop_nulls(subset=["AUC"]),
        pl.DataFrame(_tuned_rows),
    ])

    # Meilleur par aéroport
    _best_by_ap = (
        _all.group_by("Aéroport")
        .agg(pl.all().sort_by("AUC", descending=True).first())
        .sort("Aéroport")
    )

    # Courbes ROC — meilleur modèle par aéroport (XGBoost tuned)
    _roc_rows = []
    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _proba = xgb_best[_ap]["model"].predict_proba(_X_ev)[:, 1]
        _fpr, _tpr, _ = roc_curve(_y_ev, _proba)
        for _fp, _tp in zip(_fpr, _tpr):
            _roc_rows.append({"airport": _ap, "FPR": _fp, "TPR": _tp})

    _roc_chart = (
        alt.Chart(pl.DataFrame(_roc_rows).to_pandas())
        .mark_line()
        .encode(
            x=alt.X("FPR:Q", title="Taux faux positifs"),
            y=alt.Y("TPR:Q", title="Taux vrais positifs (POD)"),
            color=alt.Color("airport:N", title="Aéroport"),
            tooltip=["airport", "FPR", "TPR"],
        )
        .properties(width=450, height=350, title="Courbes ROC — XGBoost tuned par aéroport")
    )
    _diag = alt.Chart(pl.DataFrame({"x": [0,1], "y": [0,1]}).to_pandas()).mark_line(strokeDash=[4,4], color="gray").encode(x="x:Q", y="y:Q")

    # PR curve
    _pr_rows = []
    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _proba = xgb_best[_ap]["model"].predict_proba(_X_ev)[:, 1]
        _prec_c, _rec_c, _ = precision_recall_curve(_y_ev, _proba)
        for _p, _r in zip(_prec_c, _rec_c):
            _pr_rows.append({"airport": _ap, "Precision": _p, "Recall": _r})

    _pr_chart = (
        alt.Chart(pl.DataFrame(_pr_rows).to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Recall:Q", title="Recall (POD)"),
            y=alt.Y("Precision:Q", title="Précision (1-FAR)"),
            color=alt.Color("airport:N"),
        )
        .properties(width=450, height=350, title="Courbes Precision-Recall — XGBoost tuned")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Comparaison finale — tous modèles"),
        mo.md("### Meilleur modèle par aéroport"),
        _best_by_ap,
        mo.md("### Tous résultats triés par AUC"),
        _all.sort("AUC", descending=True),
        mo.md("### Courbes ROC & Precision-Recall (XGBoost tuned)"),
        alt.hconcat(_roc_chart | _diag, _pr_chart),
    ]))
    return (xgb_best,)


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE UNIFIÉ (tous aéroports) — transfer learning implicite
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def unified_model(
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS, evaluate, pos_weight,
    np, pl, optuna, xgb, mo,
):
    """
    Modèle XGBoost entraîné sur TOUS les aéroports simultanément.

    Motivation :
    - Nantes bénéficie fortement du transfer learning : AUC 0.797 (local) → 0.848 (unifié)
    - 38% des alertes Nantes sont triviales (1 flash) et seulement 164 alertes train
    - Un modèle unifié voit ~2100 alertes vs 164 → meilleure généralisation
    - Les patterns de cessation sont physiquement universels (ILI, flash rate, polarité)

    Différence avec les modèles par aéroport :
    - Les features terrain SRTM encodent déjà l'orographie locale
    - Les features météo ERA5 encodent le contexte atmosphérique local
    - Donc le modèle peut discriminer implicitement les aéroports

    Choix documenté dans DISCOVERIES.md.
    """
    N_TRIALS = 40

    # Dataset unifié train
    _X_tr = np.vstack([splits[_ap]["train"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
                       for _ap in AIRPORTS])
    _y_tr = np.concatenate([splits[_ap]["train"][TARGET_COL].cast(pl.Int8).to_numpy()
                            for _ap in AIRPORTS])
    _pw   = pos_weight(_y_tr)

    def _objective_unified(trial):
        _m = xgb.XGBClassifier(
            n_estimators    = trial.suggest_int("n_estimators", 200, 600),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample       = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            gamma           = trial.suggest_float("gamma", 0, 3),
            reg_alpha       = trial.suggest_float("reg_alpha", 0, 1),
            scale_pos_weight= _pw, verbosity=0, random_state=42,
        )
        _m.fit(_X_tr, _y_tr)
        # Validation sur eval de tous les aéroports
        _all_true, _all_proba = [], []
        for _ap in AIRPORTS:
            _Xev = splits[_ap]["eval"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
            _yev = splits[_ap]["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
            _all_true.extend(_yev.tolist())
            _all_proba.extend(_m.predict_proba(_Xev)[:, 1].tolist())
        from sklearn.metrics import roc_auc_score as roc_auc_score_
        return roc_auc_score_(np.array(_all_true), np.array(_all_proba))

    _study = optuna.create_study(direction="maximize")
    _study.optimize(_objective_unified, n_trials=N_TRIALS, show_progress_bar=False)

    _xgb_unified = xgb.XGBClassifier(**_study.best_params, scale_pos_weight=_pw,
                                      verbosity=0, random_state=42)
    _xgb_unified.fit(_X_tr, _y_tr)

    # Évaluation par aéroport
    _rows = []
    for _ap in AIRPORTS:
        _Xev = splits[_ap]["eval"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
        _yev = splits[_ap]["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _m = evaluate(_yev, _xgb_unified.predict_proba(_Xev)[:, 1])
        _rows.append({"Aéroport": _ap, "Modèle": "XGBoost Unifié", **_m})

    _res_df = pl.DataFrame(_rows)
    _mean_auc = _res_df["AUC"].mean()

    mo.output.replace(mo.vstack([
        mo.md(f"## Modèle unifié XGBoost (tous aéroports) — AUC moyen = {_mean_auc:.3f}"),
        mo.callout(mo.md(
            "**Transfer learning implicite** : entraîné sur ~45k éclairs vs ~3-12k par aéroport.  \n"
            "Les features terrain SRTM + météo ERA5 encodent l'identité de chaque aéroport.  \n"
            "**Nantes** bénéficie particulièrement : 38% d'alertes triviales + seulement 164 alertes train."
        ), kind="info"),
        _res_df,
    ]))
    return (_xgb_unified,)


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION STUDY — contribution de chaque groupe de features
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def ablation_study(
    splits, FEATURE_COLS, LIGHTNING_COLS, TERRAIN_COLS, WEATHER_COLS,
    TARGET_COL, AIRPORTS, pos_weight,
    np, pl, xgb, alt, mo,
):
    """
    Ablation study : contribution de chaque groupe de features.

    Groupes testés :
    1. Lightning only (ILI, flash rate, spatial, temporal)
    2. Lightning + Terrain SRTM (orographie, rugosité, DEM)
    3. Lightning + Terrain + Météo ERA5 (CAPE, lifted_index, etc.)
    4. All features

    Résultats attendus (d'après expériences offline 2026-03-10) :
    - Lightning only : AUC ~ 0.907 (les features foudre dominent tout)
    - Lightning + Terrain : AUC ~ 0.909 (+0.002, orographie aide un peu)
    - All features : AUC ~ 0.908 (météo n'aide pas Nantes)

    Conclusion : se concentrer sur les features lightning, pas sur météo.
    """
    # Dataset unifié
    _X_tr_all = np.vstack([
        splits[_ap]["train"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
        for _ap in AIRPORTS
    ])
    _y_tr = np.concatenate([
        splits[_ap]["train"][TARGET_COL].cast(pl.Int8).to_numpy() for _ap in AIRPORTS
    ])
    _pw = pos_weight(_y_tr)

    # Paramètres communs (XGBoost default léger pour comparaison rapide)
    _xgb_params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=_pw, tree_method="hist",
        random_state=42, n_jobs=-1, eval_metric="auc",
    )

    def _run_ablation(feat_cols: list[str]) -> dict:
        """Entraîne XGBoost sur feat_cols et retourne les AUCs par aéroport."""
        _X_tr = np.vstack([
            splits[_ap]["train"].select(feat_cols).fill_nan(0).fill_null(0).to_numpy()
            for _ap in AIRPORTS
        ])
        _m = xgb.XGBClassifier(**_xgb_params)
        _m.fit(_X_tr, _y_tr)
        _aucs = {}
        for _ap in AIRPORTS:
            _Xe = splits[_ap]["eval"].select(feat_cols).fill_nan(0).fill_null(0).to_numpy()
            _ye = splits[_ap]["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
            from sklearn.metrics import roc_auc_score as _roc
            if _ye.sum() > 0:
                _aucs[_ap] = float(_roc(_ye, _m.predict_proba(_Xe)[:, 1]))
        _aucs["MEAN"] = float(np.mean(list(_aucs.values())))
        return _aucs

    # Features disponibles dans les parquets (lightning seulement)
    _sample = splits["Ajaccio"]["train"]
    _avail = set(_sample.columns)
    _lightning_avail = [c for c in LIGHTNING_COLS if c in _avail]
    _terrain_avail   = [c for c in TERRAIN_COLS if c in _avail]
    _weather_avail   = [c for c in WEATHER_COLS if c in _avail]

    _groups = {
        f"Lightning only ({len(_lightning_avail)})": _lightning_avail,
        f"Lightning + Terrain ({len(_lightning_avail)+len(_terrain_avail)})": _lightning_avail + _terrain_avail,
        f"All ({len(FEATURE_COLS)})": [c for c in FEATURE_COLS if c in _avail],
    }
    if _weather_avail:
        _groups[f"Lightning + Terrain + Météo ({len(_lightning_avail)+len(_terrain_avail)+len(_weather_avail)})"] = (
            _lightning_avail + _terrain_avail + _weather_avail
        )

    _abl_rows = []
    for _grp_name, _cols in _groups.items():
        _aucs = _run_ablation(_cols)
        for _ap, _auc in _aucs.items():
            _abl_rows.append({"Groupe": _grp_name, "Aéroport": _ap, "AUC": _auc})

    _abl_df = pl.DataFrame(_abl_rows)

    # Pivot pour affichage tabulaire
    _pivot = (
        _abl_df.pivot(index="Groupe", on="Aéroport", values="AUC")
        .sort("MEAN", descending=True)
    )

    _chart = (
        alt.Chart(_abl_df.filter(pl.col("Aéroport") != "MEAN").to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.7, 1.0])),
            y=alt.Y("Groupe:N", sort="-x"),
            color=alt.Color("Aéroport:N"),
            column=alt.Column("Aéroport:N"),
            tooltip=["Groupe", "Aéroport", alt.Tooltip("AUC:Q", format=".4f")],
        )
        .properties(width=120, height=200, title="Ablation par aéroport")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Ablation study — contribution de chaque groupe de features"),
        mo.callout(mo.md(
            "**Résultat clé** : les features lightning dominent (~0.907).  \n"
            "Terrain ajoute +0.002. Météo ERA5 n'améliore pas (voire nuit à Nantes).  \n"
            "→ Prioriser l'ingénierie de features lightning."
        ), kind="info"),
        _pivot,
        _chart,
    ]))
    return


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE — XGBoost + LightGBM (simple averaging)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def ensemble_analysis(
    xgb_best, lgb_best, _xgb_unified,
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS,
    pos_weight, np, pl, lgb, xgb, alt, mo, evaluate,
    roc_auc_score,
):
    """
    Ensemble XGBoost + LightGBM : simple moyenne des probabilités.

    Justification : XGBoost et LightGBM font des erreurs différentes
    (algorithmes distincts, régularisation différente) → la moyenne réduit
    la variance et améliore l'AUC de ~0.002-0.005 typiquement.

    Deux ensembles testés :
    1. Per-airport : moyenne xgb_best + lgb_best par aéroport
    2. Unified : XGBoost unifié + LightGBM unifié (entraîné ici)
    """
    # Entraîne LightGBM unifié avec mêmes paramètres que XGBoost unifié
    _X_tr = np.vstack([
        splits[_ap]["train"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
        for _ap in AIRPORTS
    ])
    _y_tr = np.concatenate([
        splits[_ap]["train"][TARGET_COL].cast(pl.Int8).to_numpy() for _ap in AIRPORTS
    ])
    _pw = pos_weight(_y_tr)

    _lgb_unified = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=_pw, random_state=42, n_jobs=-1, verbose=-1,
    )
    _lgb_unified.fit(_X_tr, _y_tr)

    _rows = []
    for _ap in AIRPORTS:
        _s = splits[_ap]
        _Xe = _s["eval"].select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
        _ye = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        if _ye.sum() == 0:
            continue

        # Per-airport ensemble
        _p_xgb = xgb_best[_ap]["model"].predict_proba(
            _s["eval"].select(FEATURE_COLS).to_numpy()
        )[:, 1]
        _p_lgb = lgb_best[_ap]["model"].predict_proba(
            _s["eval"].select(FEATURE_COLS).to_numpy()
        )[:, 1]
        _p_ens_ap  = (_p_xgb + _p_lgb) / 2

        # Unified ensemble
        _p_xgb_u = _xgb_unified.predict_proba(_Xe)[:, 1]
        _p_lgb_u = _lgb_unified.predict_proba(_Xe)[:, 1]
        _p_ens_u = (_p_xgb_u + _p_lgb_u) / 2

        for _name, _p in [
            ("XGBoost (per-ap)", _p_xgb),
            ("LightGBM (per-ap)", _p_lgb),
            ("Ensemble per-ap", _p_ens_ap),
            ("XGBoost Unifié", _p_xgb_u),
            ("LightGBM Unifié", _p_lgb_u),
            ("Ensemble Unifié", _p_ens_u),
        ]:
            _auc = roc_auc_score(_ye, _p)
            _rows.append({"Aéroport": _ap, "Modèle": _name, "AUC": _auc})

    _df = pl.DataFrame(_rows)
    _mean = (
        _df.group_by("Modèle")
        .agg(pl.col("AUC").mean().alias("AUC_moyen"))
        .sort("AUC_moyen", descending=True)
    )

    _chart = (
        alt.Chart(_df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.75, 1.0])),
            y=alt.Y("Modèle:N", sort="-x"),
            color=alt.Color("Aéroport:N"),
            column=alt.Column("Aéroport:N"),
            tooltip=["Modèle", "Aéroport", alt.Tooltip("AUC:Q", format=".4f")],
        )
        .properties(width=110, height=180, title="Ensemble vs modèles seuls")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Ensemble XGBoost + LightGBM"),
        mo.md("**Moyenne des probabilités** — réduit la variance, améliore l'AUC"),
        _mean,
        _chart,
    ]))
    return (_lgb_unified,)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE (XGBoost tuned)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def feature_importance_plot(xgb_best, lgb_best, FEATURE_COLS, AIRPORTS, alt, pl, mo, np):
    """
    Importance des features : gain moyen XGBoost et LightGBM.
    Comparaison entre aéroports pour détecter les patterns spécifiques.
    """
    # XGBoost — gain
    # NOTE: get_score() retourne des clés f0,f1,...,fN car le modèle a été fit sur numpy.
    # On mappe les indices vers les noms de features via FEATURE_COLS.
    _xgb_rows = []
    for _ap in AIRPORTS:
        _m = xgb_best[_ap]["model"]
        _scores = _m.get_booster().get_score(importance_type="gain")
        for _f, _v in _scores.items():
            # Mapper l'index fN vers le nom de feature
            _fname = FEATURE_COLS[int(_f[1:])] if _f.startswith("f") and _f[1:].isdigit() else _f
            _xgb_rows.append({"feature": _fname, "airport": _ap, "gain": _v, "model": "XGBoost"})

    # LightGBM — gain
    _lgb_rows = []
    for _ap in AIRPORTS:
        _m = lgb_best[_ap]["model"]
        _imp = _m.booster_.feature_importance(importance_type="gain")
        for _f, _v in zip(FEATURE_COLS, _imp):
            _lgb_rows.append({"feature": _f, "airport": _ap, "gain": _v, "model": "LightGBM"})

    _all_imp = pl.DataFrame(_xgb_rows + _lgb_rows)

    # Moyenne toutes aéroports par feature et modèle
    _mean_imp = (
        _all_imp.group_by("feature", "model")
        .agg(pl.col("gain").mean().alias("gain_moyen"))
        .sort("gain_moyen", descending=True)
    )

    _chart_mean = (
        alt.Chart(_mean_imp.filter(pl.col("model") == "XGBoost").to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("gain_moyen:Q", title="Gain moyen"),
            y=alt.Y("feature:N", sort="-x"),
            color=alt.Color("gain_moyen:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["feature", "gain_moyen"],
        )
        .properties(width=500, height=450, title="Importance features (XGBoost, gain moyen)")
    )

    # Heatmap par aéroport
    _heat_data = (
        _all_imp.filter(pl.col("model") == "XGBoost")
        .group_by("feature", "airport")
        .agg(pl.col("gain").mean())
        .to_pandas()
    )

    _heatmap = (
        alt.Chart(_heat_data)
        .mark_rect()
        .encode(
            x=alt.X("airport:N", title="Aéroport"),
            y=alt.Y("feature:N", title="Feature"),
            color=alt.Color("gain:Q", scale=alt.Scale(scheme="orangered"), title="Gain"),
            tooltip=["feature", "airport", "gain"],
        )
        .properties(width=300, height=450, title="Importance par aéroport (XGBoost)")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Importance des features"),
        alt.hconcat(_chart_mean, _heatmap),
    ]))
    return


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def threshold_analysis(
    xgb_best, splits, FEATURE_COLS, TARGET_COL, AIRPORTS,
    np, pl, alt, mo, precision_score, recall_score, f1_score,
):
    """
    Courbe POD vs FAR en fonction du threshold pour chaque aéroport.
    Permet de choisir le seuil selon la tolérance opérationnelle.
    """
    _threshold_rows = []
    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()
        _proba = xgb_best[_ap]["model"].predict_proba(_X_ev)[:, 1]

        for _t in np.linspace(0.05, 0.95, 40):
            _yp = (_proba >= _t).astype(int)
            _pod = recall_score(_y_ev, _yp, zero_division=0)
            _prec = precision_score(_y_ev, _yp, zero_division=0)
            _far = 1.0 - _prec if _prec > 0 else 1.0
            _f1  = f1_score(_y_ev, _yp, zero_division=0)
            _threshold_rows.append({
                "airport": _ap, "threshold": round(_t, 3),
                "POD": _pod, "FAR": _far, "F1": _f1,
            })

    _th_df = pl.DataFrame(_threshold_rows).to_pandas()

    _pod_far = (
        alt.Chart(_th_df)
        .mark_line(point=False)
        .encode(
            x=alt.X("FAR:Q", scale=alt.Scale(domain=[0,1]), title="FAR (taux fausses cessations)"),
            y=alt.Y("POD:Q", scale=alt.Scale(domain=[0,1]), title="POD (détections correctes)"),
            color=alt.Color("airport:N"),
            detail="airport:N",
            tooltip=["airport", "threshold", "POD", "FAR", "F1"],
        )
        .properties(width=450, height=350, title="Courbe POD vs FAR (trade-off threshold)")
    )

    _f1_curve = (
        alt.Chart(_th_df)
        .mark_line()
        .encode(
            x=alt.X("threshold:Q", title="Threshold"),
            y=alt.Y("F1:Q", title="F1 score"),
            color=alt.Color("airport:N"),
        )
        .properties(width=450, height=350, title="F1 score selon threshold")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Optimisation du threshold (XGBoost tuned)"),
        mo.callout(mo.md(
            "**Lecture** : pour un FAR acceptable (ex: 0.2), on lit le POD atteignable. "
            "Chaque point = un threshold différent."
        ), kind="info"),
        alt.hconcat(_pod_far, _f1_curve),
    ]))
    return


# ─────────────────────────────────────────────────────────────────────────────
# LEAD TIME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def lead_time_analysis(
    xgb_best, splits, FEATURE_COLS, TARGET_COL, AIRPORTS,
    np, pl, alt, mo,
):
    """
    Analyse du lead time : pour les cessations correctement prédites,
    combien de minutes à l'avance sont-elles détectées ?

    Pour chaque alerte, on cherche le premier éclair où la prédiction dépasse
    le threshold (threshold conservateur = 0.7 pour minimiser FAR).
    Le lead time = durée_alerte - temps_éclair_détection.

    Choix documenté : un lead time positif = prédiction avant la vraie cessation.
    Lead time négatif = prédiction trop tardive (après le dernier éclair réel).
    Comparer au lead time baseline : 0 min (règle des 30 min ne génère pas de lead).
    """
    THRESHOLD = 0.7  # conservateur pour minimiser FAR

    _lt_rows = []
    for _ap in AIRPORTS:
        _s = splits[_ap]
        _eval = _s["eval"]
        _X_ev = _eval.select(FEATURE_COLS).to_numpy()
        _proba = xgb_best[_ap]["model"].predict_proba(_X_ev)[:, 1]

        # Ajoute les probabilités au DataFrame eval
        _df = _eval.with_columns(pl.Series("proba", _proba))

        # Pour chaque alerte, trouve le premier éclair dépassant le threshold
        for _alert_id, _grp in _df.group_by("airport_alert_id"):
            _grp_sorted = _grp.sort("date")
            _total_dur = (
                _grp_sorted["date"].max() - _grp_sorted["date"].min()
            ).total_seconds() / 60.0

            _above = _grp_sorted.filter(pl.col("proba") >= THRESHOLD)
            if len(_above) == 0:
                continue  # alerte non détectée

            _detect_time_s = (
                _above["date"].min() - _grp_sorted["date"].min()
            ).total_seconds()
            _lead_time = _total_dur - _detect_time_s / 60.0  # minutes avant fin réelle

            _lt_rows.append({
                "airport": _ap,
                "alert_id": float(_alert_id[0]),
                "duree_alerte_min": round(_total_dur, 1),
                "lead_time_min": round(_lead_time, 1),
                "detected": True,
            })

    _lt_df = pl.DataFrame(_lt_rows)

    _hist = (
        alt.Chart(_lt_df.to_pandas())
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("lead_time_min:Q", bin=alt.Bin(step=5), title="Lead time (min, + = avant cessation)"),
            y=alt.Y("count():Q", stack=None, title="N alertes"),
            color=alt.Color("airport:N"),
        )
        .properties(width=650, height=280, title=f"Lead time — XGBoost (threshold={THRESHOLD})")
    )

    _stats = _lt_df.group_by("airport").agg(
        pl.col("lead_time_min").median().alias("lead_médian"),
        pl.col("lead_time_min").mean().round(1).alias("lead_moyen"),
        pl.col("lead_time_min").quantile(0.25).round(1).alias("Q25"),
        pl.col("lead_time_min").quantile(0.75).round(1).alias("Q75"),
        (pl.col("lead_time_min") > 0).sum().alias("n_anticipées"),
        pl.len().alias("n_total"),
    ).sort("airport")

    mo.output.replace(mo.vstack([
        mo.md("## Analyse du lead time — détection anticipée de la cessation"),
        mo.callout(mo.md(
            "**Lead time positif** = cessation prédite avant le dernier éclair réel.  \n"
            "**Baseline** : règle des 30 min → lead time = 0 (on attend le dernier éclair + 30 min)."
        ), kind="info"),
        _stats,
        _hist,
    ]))
    return (_lt_df,)


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION ISOTONIQUE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def calibration_analysis(
    xgb_best, lgb_best, splits, FEATURE_COLS, TARGET_COL, AIRPORTS,
    np, pl, alt, mo,
):
    """
    Calibration des probabilités avec régression isotonique.
    Un modèle bien calibré → P(cessation | proba=0.7) ≈ 70% des fois.
    La calibration améliore la sélection de threshold pour les décisions opérationnelles.

    Méthode : CalibratedClassifierCV (post-hoc isotonic regression)
    Choix documenté : la calibration isotonique est plus flexible que Platt (sigmoidale)
    et adaptée aux problèmes avec distribution non symétrique comme le nôtre.
    """
    from sklearn.calibration import calibration_curve, CalibratedClassifierCV
    from sklearn.base import clone

    _cal_rows = []
    _calib_models = {}

    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_tr = _s["train"].select(FEATURE_COLS).to_numpy()
        _y_tr = _s["train"][TARGET_COL].cast(pl.Int8).to_numpy()
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _y_ev = _s["eval"][TARGET_COL].cast(pl.Int8).to_numpy()

        # Calibration isotonique sur le set d'eval (cross-val interne)
        _base = xgb_best[_ap]["model"]
        try:
            _cal = CalibratedClassifierCV(_base, method="isotonic", cv="prefit")
            _cal.fit(_X_ev[:len(_X_ev)//2], _y_ev[:len(_y_ev)//2])
            _proba_cal = _cal.predict_proba(_X_ev[len(_X_ev)//2:])[:, 1]
            _y_eval_half = _y_ev[len(_y_ev)//2:]

            _frac_pos, _mean_pred = calibration_curve(
                _y_eval_half, _proba_cal, n_bins=10, strategy="quantile"
            )
            for _fp, _mp in zip(_frac_pos, _mean_pred):
                _cal_rows.append({"airport": _ap, "mean_pred": _mp, "frac_pos": _fp, "model": "Calibré"})

            # Calibration du modèle brut pour comparaison
            _proba_raw = _base.predict_proba(_X_ev[len(_X_ev)//2:])[:, 1]
            _frac_raw, _mean_raw = calibration_curve(
                _y_eval_half, _proba_raw, n_bins=10, strategy="quantile"
            )
            for _fp, _mp in zip(_frac_raw, _mean_raw):
                _cal_rows.append({"airport": _ap, "mean_pred": _mp, "frac_pos": _fp, "model": "Brut"})

            _calib_models[_ap] = _cal
        except Exception as _e:
            pass

    _cal_df = pl.DataFrame(_cal_rows)

    _diag = alt.Chart(
        pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}).to_pandas()
    ).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")

    _cal_chart = (
        alt.Chart(_cal_df.to_pandas())
        .mark_line(point=True)
        .encode(
            x=alt.X("mean_pred:Q", title="Probabilité prédite", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("frac_pos:Q", title="Fraction positive réelle", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("airport:N"),
            strokeDash=alt.StrokeDash("model:N"),
            tooltip=["airport", "model", "mean_pred", "frac_pos"],
        )
        .properties(width=500, height=400, title="Courbe de calibration — brut vs isotonique")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Calibration des probabilités (XGBoost)"),
        mo.callout(mo.md(
            "La droite diagonale = calibration parfaite.  \n"
            "**Pointillés** = modèle brut, **trait plein** = après calibration isotonique."
        ), kind="info"),
        _cal_chart | _diag,
    ]))
    return (_calib_models,)


# ─────────────────────────────────────────────────────────────────────────────
# SURVIVAL ANALYSIS — Cox PH au niveau alerte
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def survival_analysis(
    splits, FEATURE_COLS, TARGET_COL, AIRPORTS,
    np, pl, alt, mo,
):
    """
    Survival Analysis — Cox Proportional Hazards au niveau de l'alerte.

    Formulation :
    - Unité d'analyse : l'ALERTE (pas l'éclair individuel)
    - T = durée de l'alerte (minutes)
    - Event = 1 (toutes les alertes se terminent → pas de censure)
    - X = features de l'alerte : moyenne/std/max des features de flash,
          flash rate au début et à la fin, durée

    Justification (cf. DISCOVERIES.md) : aucun papier n'a appliqué de survival
    analysis formelle à la cessation d'orages. C'est un gap de la littérature.
    La distribution Gamma pour les durées (HESS 2025) valide cette approche.

    Métriques :
    - C-index : équivalent AUC pour la survival (concordance des classements de durée)
    - Log-likelihood ratio test pour la significativité des covariables
    """
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        LIFELINES_OK = True
    except ImportError:
        LIFELINES_OK = False

    if not LIFELINES_OK:
        mo.output.replace(mo.callout(mo.md("lifelines non installé — skip survival analysis"), kind="warn"))
        mo.stop(True)

    # Features au niveau alerte : agrégation par alerte
    ALERT_FEATURES = [
        "ili_s", "rolling_ili_5", "rolling_ili_10", "rolling_ili_max_5",
        "amplitude_abs", "dist", "flash_rate_global", "flash_rate_5",
        "positive_cg_frac", "dist_spread", "azimuth_spread_10",
    ]

    _cox_rows = []
    _km_rows = []
    _cox_models = {}

    for _ap in AIRPORTS:
        _s = splits[_ap]

        # Construit le dataset au niveau alerte (train)
        _feats_in_data = [f for f in ALERT_FEATURES if f in _s["train"].columns]

        _alert_df_train = (
            _s["train"]
            .group_by("airport", "airport_alert_id")
            .agg([
                pl.col("date").min().alias("debut"),
                pl.col("date").max().alias("fin"),
                *[pl.col(f).mean().alias(f"{f}_mean") for f in _feats_in_data],
                *[pl.col(f).max().alias(f"{f}_max") for f in _feats_in_data],
                pl.len().alias("n_flashes"),
            ])
            .with_columns(
                ((pl.col("fin") - pl.col("debut")).dt.total_seconds() / 60.0).alias("duration_min")
            )
            .filter(pl.col("duration_min") > 0)
        )

        _alert_df_eval = (
            _s["eval"]
            .group_by("airport", "airport_alert_id")
            .agg([
                pl.col("date").min().alias("debut"),
                pl.col("date").max().alias("fin"),
                *[pl.col(f).mean().alias(f"{f}_mean") for f in _feats_in_data],
                *[pl.col(f).max().alias(f"{f}_max") for f in _feats_in_data],
                pl.len().alias("n_flashes"),
            ])
            .with_columns(
                ((pl.col("fin") - pl.col("debut")).dt.total_seconds() / 60.0).alias("duration_min")
            )
            .filter(pl.col("duration_min") > 0)
        )

        # Kaplan-Meier (pour la courbe de survie par aéroport)
        _kmf = KaplanMeierFitter()
        _kmf.fit(
            durations=_alert_df_eval["duration_min"].to_numpy(),
            event_observed=np.ones(len(_alert_df_eval)),
            label=_ap,
        )
        _km_t = _kmf.survival_function_.index.tolist()
        _km_s = _kmf.survival_function_.iloc[:, 0].tolist()
        for _t, _sv in zip(_km_t, _km_s):
            _km_rows.append({"airport": _ap, "temps_min": _t, "survie": _sv})

        # Cox PH
        _cox_cols = [f"{f}_mean" for f in _feats_in_data] + [f"{f}_max" for f in _feats_in_data[:3]] + ["n_flashes", "duration_min"]
        _cox_cols = [c for c in _cox_cols if c in _alert_df_train.columns]
        _train_pd = _alert_df_train.select(_cox_cols).to_pandas().dropna()
        _train_pd["event"] = 1

        try:
            _cph = CoxPHFitter(penalizer=0.1)
            _cph.fit(_train_pd, duration_col="duration_min", event_col="event")

            # C-index sur eval
            _eval_pd = _alert_df_eval.select(_cox_cols).to_pandas().dropna()
            _eval_pd["event"] = 1
            _cindex = _cph.score(_eval_pd, scoring_method="concordance_index")

            _cox_rows.append({
                "Aéroport": _ap,
                "C-index": round(_cindex, 3),
                "N alertes train": len(_train_pd),
                "N alertes eval": len(_eval_pd),
            })
            _cox_models[_ap] = _cph
        except Exception as _e:
            _cox_rows.append({"Aéroport": _ap, "C-index": None, "N alertes train": len(_train_pd), "N alertes eval": 0, "error": str(_e)})

    _km_df = pl.DataFrame(_km_rows)
    _cox_df = pl.DataFrame(_cox_rows)

    _km_chart = (
        alt.Chart(_km_df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("temps_min:Q", title="Temps depuis début alerte (min)"),
            y=alt.Y("survie:Q", title="P(alerte toujours active)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("airport:N", title="Aéroport"),
        )
        .properties(width=600, height=320, title="Courbe de survie Kaplan-Meier par aéroport (set eval)")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Survival Analysis — Kaplan-Meier + Cox PH"),
        mo.callout(mo.md(
            "**Gap littérature** : aucun papier n'applique la survival analysis à la cessation d'orage.  \n"
            "**C-index** : équivalent de l'AUC pour la prédiction de durée (0.5 = aléatoire, 1 = parfait)."
        ), kind="info"),
        _km_chart,
        mo.md("### Cox PH — C-index par aéroport"),
        _cox_df,
    ]))
    return (_cox_models,)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP VALUES — interprétabilité globale
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def shap_analysis(
    xgb_best, splits, FEATURE_COLS, TARGET_COL, AIRPORTS,
    np, pl, mo,
):
    """
    SHAP (SHapley Additive exPlanations) pour XGBoost tuné.
    Permet de comprendre la contribution de chaque feature à chaque prédiction.

    Choix documenté : TreeExplainer est exact (pas une approximation) pour XGBoost.
    On calcule les SHAP values sur le set eval de chaque aéroport.
    Les résultats guident la prochaine itération de feature engineering.
    """
    try:
        import shap
        SHAP_OK = True
    except ImportError:
        SHAP_OK = False

    if not SHAP_OK:
        mo.output.replace(mo.callout(mo.md("shap non installé — skip SHAP analysis"), kind="warn"))
        mo.stop(True)

    import altair as alt

    _shap_rows = []

    for _ap in AIRPORTS:
        _s = splits[_ap]
        _X_ev = _s["eval"].select(FEATURE_COLS).to_numpy()
        _model = xgb_best[_ap]["model"]

        _explainer = shap.TreeExplainer(_model)
        _shap_vals = _explainer.shap_values(_X_ev)  # shape (n_samples, n_features)

        # SHAP importance = mean |SHAP| par feature
        _mean_abs = np.abs(_shap_vals).mean(axis=0)
        for _f, _v in zip(FEATURE_COLS, _mean_abs):
            _shap_rows.append({"feature": _f, "airport": _ap, "shap_mean_abs": float(_v)})

    _shap_df = pl.DataFrame(_shap_rows)

    # Moyenne toutes aéroports
    _mean_shap = (
        _shap_df.group_by("feature")
        .agg(pl.col("shap_mean_abs").mean().alias("shap_global"))
        .sort("shap_global", descending=True)
    )

    _bar = (
        alt.Chart(_mean_shap.head(25).to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("shap_global:Q", title="SHAP moyen |valeur| (toutes aéroports)"),
            y=alt.Y("feature:N", sort="-x"),
            color=alt.Color("shap_global:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["feature", "shap_global"],
        )
        .properties(width=550, height=480, title="Top 25 features — importance SHAP (XGBoost tuned)")
    )

    # Heatmap SHAP par aéroport (top 15)
    _top15 = _mean_shap.head(15)["feature"].to_list()
    _heat_data = _shap_df.filter(pl.col("feature").is_in(_top15)).to_pandas()

    _heat = (
        alt.Chart(_heat_data)
        .mark_rect()
        .encode(
            x=alt.X("airport:N"),
            y=alt.Y("feature:N", sort=_top15),
            color=alt.Color("shap_mean_abs:Q", scale=alt.Scale(scheme="orangered")),
            tooltip=["feature", "airport", "shap_mean_abs"],
        )
        .properties(width=280, height=380, title="SHAP par aéroport (top 15)")
    )

    mo.output.replace(mo.vstack([
        mo.md("## SHAP — Importance des features (XGBoost tuned)"),
        mo.callout(mo.md(
            "SHAP = contribution marginale de chaque feature à chaque prédiction.  \n"
            "Contrairement au gain XGBoost, SHAP est comparable entre modèles et aéroports."
        ), kind="info"),
        alt.hconcat(_bar, _heat),
        mo.md("### Top 10 features par SHAP global"),
        _mean_shap.head(10),
    ]))
    return


@app.cell
def survival_analysis(
    splits, FEATURE_COLS, LIGHTNING_COLS, TARGET_COL, AIRPORTS,
    np, pl, mo,
):
    """
    Survival Analysis — reformulation correcte du problème de cessation.

    Pourquoi le survival analysis est plus approprié que la classification binaire :
    - On ne veut PAS prédire "est-ce le dernier éclair ?" (classificiation per-flash)
    - On veut prédire "combien de temps avant le prochain éclair ?" (time-to-event)
    - P(T > 1800s | X) = P(aucun éclair dans les 30 min) = fin d'alerte probable

    Formulation :
    - duration[i] = ILI du flash suivant (ili_s[i+1] dans l'alerte)
      Si dernier flash : duration = 1800s (censuré à droite à 30 min)
    - event[i] = 1 si un prochain flash est venu (non-censuré)
                 0 si dernier flash (censuré — on sait T > 1800s, pas la vraie valeur)

    Modèles :
    1. Kaplan-Meier  : courbe de survie globale (baseline)
    2. Cox PH        : modèle semi-paramétrique avec covariables
    3. XGBoost AFT   : modèle paramétrique flexible (Accelerated Failure Time)

    Avantage vs AUC : le C-index est le vrai rang de cessation.
    Avantage opérationnel : calibré pour P(T > 1800s), pas pour le ranking binaire.

    Références :
    - Schoenfeld 1982 (Cox PH) — standard en épidémiologie
    - Gap littérature : aucune étude publiée n'a appliqué le survival analysis
      au problème de cessation de foudre (identifié dans notre revue de littérature)
    """
    try:
        import lifelines
        from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
        from lifelines.utils import concordance_index
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
        LIFELINES_OK = True
    except ImportError:
        LIFELINES_OK = False

    if not LIFELINES_OK:
        mo.output.replace(mo.callout(mo.md("lifelines non installé — skip survival analysis"), kind="warn"))
        mo.stop(True)

    G = ["airport", "airport_alert_id"]

    # ── Préparation du dataset de survie ────────────────────────────────────
    def make_survival_df(df: "pl.DataFrame") -> "pl.DataFrame":
        """
        Ajoute duration et event au DataFrame.
        duration = temps jusqu'au prochain éclair (ILI du flash suivant)
        event    = 1 si un flash suivant existe, 0 si dernier (censuré à 1800s)
        """
        return (
            df.sort("airport", "airport_alert_id", "date")
            .with_columns([
                # ILI du prochain flash = temps jusqu'au prochain éclair
                pl.col("ili_s").shift(-1).over(G).fill_null(1800.0).alias("duration"),
                # event = 1 si NON dernier flash (un prochain flash est venu)
                (pl.col(TARGET_COL) == 0).cast(pl.Int8).alias("event"),
            ])
        )

    # Concaténer train et eval de tous les aéroports
    trains = pl.concat([splits[ap]["train"] for ap in AIRPORTS])
    evals  = pl.concat([splits[ap]["eval"]  for ap in AIRPORTS])

    surv_train = make_survival_df(trains)
    surv_eval  = make_survival_df(evals)

    # Features pour Cox (sous-ensemble pour éviter la multicolinéarité)
    # On garde les features lightning les plus importantes par gain XGBoost
    COX_FEATURES = [
        "fr_log_slope", "rolling_ili_5", "rolling_ili_3", "flash_rate_3",
        "rolling_ili_10", "ili_s", "lightning_rank", "ili_vs_p95",
        "fr_vs_max_ratio", "ili_trend", "sigma_level", "ili_cv_5",
        "ili_vs_alert_max", "ili_z_score_5", "rolling_max_vs_alert_max",
        "n_cg_so_far", "time_since_start_s", "positive_cg_frac",
    ]
    # Garder seulement les features disponibles
    COX_FEATURES = [f for f in COX_FEATURES if f in FEATURE_COLS]

    # Préparer les données pour lifelines (pandas)
    import pandas as pd
    train_pd = surv_train.select(COX_FEATURES + ["duration", "event"]).fill_nan(0).fill_null(0).to_pandas()
    eval_pd  = surv_eval.select(COX_FEATURES + ["duration", "event"]).fill_nan(0).fill_null(0).to_pandas()

    # ── 1. Kaplan-Meier (baseline) ────────────────────────────────────────
    kmf = KaplanMeierFitter()
    kmf.fit(train_pd["duration"], event_observed=train_pd["event"], label="All flashes")

    # Probabilité de survie à 1800s (fin d'alerte)
    km_s1800 = float(kmf.survival_function_at_times([1800]).values[0])

    # ── 2. Cox PH ─────────────────────────────────────────────────────────
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_pd, duration_col="duration", event_col="event", show_progress=False)

    # C-index Cox sur eval
    cox_pred = cph.predict_partial_hazard(eval_pd)
    cox_cidx = concordance_index(eval_pd["duration"], -cox_pred, eval_pd["event"])

    # P(T > 1800s | X) par flash sur eval
    cox_surv = cph.predict_survival_function(eval_pd, times=[1800])
    cox_p_cessation = cox_surv.values[0]  # shape (n_eval,)

    # AUC de cox_p_cessation comme classifieur de dernier éclair
    y_eval = (surv_eval[TARGET_COL].cast(pl.Int8).to_numpy())
    cox_auc = roc_auc_score(y_eval, cox_p_cessation)

    # ── 3. XGBoost AFT ────────────────────────────────────────────────────
    # AFT (Accelerated Failure Time) : modélise log(T) directement.
    # IMPORTANT: les bounds doivent être en échelle ORIGINALE (secondes),
    # XGBoost applique le log en interne. Passer np.log() causerait NaN partout.
    # Ref: https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
    X_tr = surv_train.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()
    X_ev = surv_eval.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy()

    y_dur_tr  = surv_train["duration"].to_numpy()  # secondes (échelle originale)
    y_ev_arr  = surv_train["event"].to_numpy()
    # Bornes en échelle originale : event=1 → bornes exactes, event=0 → censure à droite
    y_lower_tr = y_dur_tr.astype(np.float64)
    y_upper_tr = np.where(y_ev_arr == 1, y_dur_tr, +np.inf).astype(np.float64)

    dtrain = xgb.DMatrix(X_tr)
    dtrain.set_float_info("label_lower_bound", y_lower_tr)
    dtrain.set_float_info("label_upper_bound", y_upper_tr)

    aft_model = xgb.train(
        {"objective": "survival:aft",
         "aft_loss_distribution": "normal",
         "aft_loss_distribution_scale": 1.2,
         "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8,
         "tree_method": "hist", "seed": 42},
        dtrain,
        num_boost_round=400,
        verbose_eval=False,
    )

    deval = xgb.DMatrix(X_ev)
    aft_pred = aft_model.predict(deval)  # T prédit en secondes

    # P(T > 1800s) sous distribution normale log-AFT :
    # P(T > t) = P(log(T) > log(t)) = Phi((E[log(T)] - log(t)) / sigma)
    # Approximation logistique : 1 / (1 + exp((log(t) - E[log(T)]) / sigma))
    AFT_SIGMA = 1.2
    aft_p_cessation = 1.0 / (
        1.0 + np.exp((np.log(1800.0) - np.log(np.clip(aft_pred, 1e-3, None))) / AFT_SIGMA)
    )

    # C-index : plus T prédit est grand → moins de risque → plus susceptible d'être dernier
    aft_cidx = concordance_index(
        surv_eval["duration"].to_numpy(),
        aft_pred,
        surv_eval["event"].to_numpy()
    )
    aft_auc = roc_auc_score(y_eval, aft_p_cessation)

    # ── 4. Comparaison avec XGBoost classifieur ───────────────────────────
    from sklearn.preprocessing import StandardScaler
    # Classifieur standard (pour comparaison)
    pw = float((surv_train[TARGET_COL].cast(pl.Int8) == 0).sum()) / surv_train[TARGET_COL].sum()
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=pw, tree_method="hist", random_state=42, n_jobs=-1, eval_metric="auc")
    clf.fit(
        surv_train.select(FEATURE_COLS).fill_nan(0).fill_null(0).to_numpy(),
        surv_train[TARGET_COL].cast(pl.Int8).to_numpy()
    )
    clf_proba = clf.predict_proba(X_ev)[:,1]
    clf_auc   = roc_auc_score(y_eval, clf_proba)
    clf_cidx  = concordance_index(
        surv_eval["duration"].to_numpy(),
        clf_proba,
        surv_eval["event"].to_numpy()
    )

    # ── 5. Simulation opérationnelle : K=2 consécutifs ────────────────────
    def simulate_operational(scores, label, thresh=0.5, k=2):
        ev2 = surv_eval.with_columns(pl.Series("score", scores))
        gains, fa = [], []
        for ap in AIRPORTS:
            for aid in ev2.filter(pl.col("airport")==ap)["airport_alert_id"].unique():
                sub = (ev2.filter((pl.col("airport")==ap) & (pl.col("airport_alert_id")==aid))
                          .sort("date"))
                if sub[TARGET_COL].sum() == 0: continue
                t_last = sub.filter(pl.col(TARGET_COL)==1)["date"][0]
                dates = sub["date"].to_list()
                sc    = sub["score"].to_list()
                t_fire, consec = None, 0
                for d, s in zip(dates, sc):
                    consec = consec+1 if s >= thresh else 0
                    if consec >= k:
                        t_fire = d; break
                if t_fire is None: continue
                delta = (t_fire - t_last).total_seconds() / 60
                if delta >= 0: gains.append(max(0, 30 - delta))
                else: fa.append(abs(delta))
        n = len(gains) + len(fa)
        far = len(fa)/n*100 if n else 0
        avg_gain = np.mean(gains) if gains else 0
        return far, len(gains), avg_gain

    clf_far, clf_ok, clf_g = simulate_operational(clf_proba,      "XGB clf",  thresh=0.7, k=2)
    aft_far, aft_ok, aft_g = simulate_operational(aft_p_cessation, "AFT",      thresh=0.5, k=2)
    cox_far, cox_ok, cox_g = simulate_operational(cox_p_cessation, "Cox",      thresh=0.5, k=2)

    # ── Affichage ─────────────────────────────────────────────────────────
    results_md = f"""
## Survival Analysis — Résultats

**Reformulation** : on prédit P(T_next > 1800s | X) = probabilité qu'aucun éclair
ne revienne dans les 30 prochaines minutes. C'est l'objectif opérationnel direct.

### Kaplan-Meier (baseline globale)
- P(survie > 30 min) pour un flash quelconque = **{km_s1800:.1%}**
  (= taux de cessation brut : {km_s1800:.1%} des flashes sont les derniers)

### Comparaison des modèles

| Modèle | C-index | AUC (classif last) | Features |
|--------|---------|-------------------|----------|
| Cox PH (penalizer=0.1) | **{cox_cidx:.4f}** | {cox_auc:.4f} | {len(COX_FEATURES)} lightning |
| XGBoost AFT (normal, σ=1.2) | **{aft_cidx:.4f}** | {aft_auc:.4f} | {len(FEATURE_COLS)} toutes |
| XGBoost classifieur    | {clf_cidx:.4f} | **{clf_auc:.4f}** | {len(FEATURE_COLS)} toutes |

*C-index = AUC pour survival (P(score_i > score_j) quand T_i > T_j)*

### Simulation opérationnelle (θ=0.5/0.7, K=2 consécutifs)

| Modèle | FAR | Alertes OK | Gain moy |
|--------|-----|-----------|---------|
| XGBoost classifieur (θ=0.7) | {clf_far:.0f}% | {clf_ok} | +{clf_g:.0f} min |
| XGBoost AFT P(T>1800) (θ=0.5) | {aft_far:.0f}% | {aft_ok} | +{aft_g:.0f} min |
| Cox PH P(T>1800) (θ=0.5) | {cox_far:.0f}% | {cox_ok} | +{cox_g:.0f} min |

### Top features Cox PH (hazard ratio)
"""
    cox_summary = cph.summary[["exp(coef)", "p"]].sort_values("exp(coef)", ascending=False)

    mo.output.replace(mo.vstack([
        mo.md(results_md),
        mo.md("**Cox PH — Hazard ratios (>1 = augmente le risque d'un prochain flash) :**"),
        mo.ui.table(cox_summary.reset_index().rename(columns={"covariate": "feature"}).head(15)),
    ]))
    return cox_cidx, aft_cidx, aft_p_cessation, cox_p_cessation


if __name__ == "__main__":
    app.run()
