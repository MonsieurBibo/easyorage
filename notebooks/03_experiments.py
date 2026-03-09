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
    TARGET_COL = meta["target_col"]

    AIRPORTS = ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]

    splits = {}
    for _ap in AIRPORTS:
        _key = _ap.lower()
        _train = pl.read_parquet(str(PROC / f"{_key}_train.parquet"))
        _eval  = pl.read_parquet(str(PROC / f"{_key}_eval.parquet"))
        splits[_ap] = {"train": _train, "eval": _eval}

    mo.output.replace(mo.md(
        f"**{len(FEATURE_COLS)} features** chargées · "
        f"**{sum(len(s['train']) + len(s['eval']) for s in splits.values()):,}** éclairs CG"
    ))
    return FEATURE_COLS, TARGET_COL, AIRPORTS, splits


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
            _all.group_by("airport_alert_id")
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
def eda_correlation(splits, pl, alt, mo, np):
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
    baseline_results, xgb_best, lgb_best, rf_best, mlp_best,
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
# FEATURE IMPORTANCE (XGBoost tuned)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def feature_importance_plot(xgb_best, lgb_best, FEATURE_COLS, AIRPORTS, alt, pl, mo, np):
    """
    Importance des features : gain moyen XGBoost et LightGBM.
    Comparaison entre aéroports pour détecter les patterns spécifiques.
    """
    # XGBoost — gain
    _xgb_rows = []
    for _ap in AIRPORTS:
        _m = xgb_best[_ap]["model"]
        _scores = _m.get_booster().get_score(importance_type="gain")
        for _f, _v in _scores.items():
            _xgb_rows.append({"feature": _f, "airport": _ap, "gain": _v, "model": "XGBoost"})

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


if __name__ == "__main__":
    app.run()
