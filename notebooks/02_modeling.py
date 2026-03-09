# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "numpy",
#     "scikit-learn",
#     "xgboost",
#     "altair",
#     "pyarrow==23.0.1",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import xgboost as xgb

    return (
        MLPClassifier,
        StandardScaler,
        alt,
        compute_sample_weight,
        f1_score,
        mo,
        np,
        pl,
        precision_score,
        recall_score,
        roc_auc_score,
        xgb,
    )


@app.cell
def load_data(pl):
    """
    Chargement des données.
    On filtre immédiatement sur airport_alert_id not null :
    on ne travaille qu'avec les éclairs en situation d'alerte (<20km).
    Les éclairs hors alerte (20-30km) sont ignorés pour l'instant.
    """
    import pathlib

    DATA_PATH = str(
        pathlib.Path(__file__).parent.parent
        / "data_train_databattle2026"
        / "segment_alerts_all_airports_train.csv"
    )

    df = (
        pl.read_csv(
            DATA_PATH,
            schema_overrides={
                "lightning_id": pl.Int64,
                "lightning_airport_id": pl.Int64,
                "lon": pl.Float64,
                "lat": pl.Float64,
                "amplitude": pl.Float64,
                "maxis": pl.Float64,
                "icloud": pl.Boolean,
                "dist": pl.Float64,
                "azimuth": pl.Float64,
                "airport_alert_id": pl.Float64,
            },
        )
        .with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S%z"),
        )
        .filter(pl.col("airport_alert_id").is_not_null())
    )

    return (df,)


@app.cell
def feature_engineering(df, pl):
    """
    Features simples calculées par éclair dans l'alerte.
    Toutes les features utilisent uniquement les données jusqu'à l'éclair courant (pas de lookahead).

    Le DataFrame est trié par (airport, airport_alert_id, date) avant le calcul
    des features cumulatives, pour garantir l'ordre chronologique dans les over().

    Features :
    - dist, amplitude_abs, icloud_int, azimuth, maxis : données brutes
    - time_since_start_s : secondes depuis le 1er éclair de l'alerte
    - lightning_rank : position de cet éclair dans l'alerte (1 = premier)
    - ili_s : intervalle depuis l'éclair précédent (0 pour le 1er)
    - n_cg_so_far, n_ic_so_far : comptages cumulatifs CG et IC
    - ic_cg_ratio_so_far : ratio IC/(CG+1) cumulatif
    - mean_amplitude_so_far, mean_dist_so_far : moyennes cumulatives
    """
    df_feat = (
        df.sort("airport", "airport_alert_id", "date")
        .with_columns(
            pl.col("amplitude").abs().alias("amplitude_abs"),
            pl.col("icloud").cast(pl.Int8).alias("icloud_int"),
        )
        .with_columns(
            # Secondes depuis le début de l'alerte
            (
                pl.col("date")
                - pl.col("date").min().over("airport", "airport_alert_id")
            )
            .dt.total_seconds()
            .alias("time_since_start_s"),
            # Rang dans l'alerte (1 = premier éclair)
            pl.col("date")
            .rank("ordinal")
            .over("airport", "airport_alert_id")
            .alias("lightning_rank"),
            # Intervalle inter-éclair (0 pour le 1er éclair de l'alerte)
            pl.col("date")
            .diff()
            .over("airport", "airport_alert_id")
            .dt.total_seconds()
            .fill_null(0.0)
            .alias("ili_s"),
            # Comptage cumulatif des CG (icloud=False)
            (pl.col("icloud") == False)
            .cast(pl.Int32)
            .cum_sum()
            .over("airport", "airport_alert_id")
            .alias("n_cg_so_far"),
            # Comptage cumulatif des IC (icloud=True)
            pl.col("icloud")
            .cast(pl.Int32)
            .cum_sum()
            .over("airport", "airport_alert_id")
            .alias("n_ic_so_far"),
            # Moyenne cumulative de l'amplitude absolue (cum_mean absent en polars 1.x)
            (
                pl.col("amplitude_abs").cum_sum().over("airport", "airport_alert_id")
                / pl.col("amplitude_abs").cum_count().over("airport", "airport_alert_id")
            ).alias("mean_amplitude_so_far"),
            # Moyenne cumulative de la distance
            (
                pl.col("dist").cum_sum().over("airport", "airport_alert_id")
                / pl.col("dist").cum_count().over("airport", "airport_alert_id")
            ).alias("mean_dist_so_far"),
        )
        .with_columns(
            # Ratio IC/CG cumulatif (+1 au dénominateur pour éviter div/0)
            (pl.col("n_ic_so_far") / (pl.col("n_cg_so_far") + 1)).alias(
                "ic_cg_ratio_so_far"
            ),
        )
    )

    # Note : tous les éclairs en-alerte (airport_alert_id not null) sont CG (icloud=False).
    # Les IC dans le rayon 20km ne reçoivent pas d'airport_alert_id.
    # → icloud_int, n_ic_so_far, ic_cg_ratio_so_far sont constants = 0 : on les exclut.
    FEATURE_COLS = [
        "dist",
        "amplitude_abs",
        "azimuth",
        "maxis",
        "time_since_start_s",
        "lightning_rank",
        "ili_s",
        "n_cg_so_far",
        "mean_amplitude_so_far",
        "mean_dist_so_far",
    ]
    TARGET_COL = "is_last_lightning_cloud_ground"

    # Pas de nulls sur TARGET_COL dans ce dataset (tous CG), mais on filtre par sécurité
    df_model = df_feat.filter(pl.col(TARGET_COL).is_not_null())

    airports = sorted(df_model["airport"].unique().to_list())

    return FEATURE_COLS, TARGET_COL, airports, df_feat, df_model


@app.cell
def show_data_stats(df_model, mo, pl):
    stats = (
        df_model.group_by("airport")
        .agg(
            pl.len().alias("n_éclairs_CG"),
            pl.col("airport_alert_id").n_unique().alias("n_alertes"),
            pl.col("is_last_lightning_cloud_ground").sum().alias("n_positifs"),
        )
        .sort("airport")
        .with_columns(
            (pl.col("n_positifs") / pl.col("n_éclairs_CG") * 100)
            .round(2)
            .alias("% positifs")
        )
    )

    mo.output.replace(
        mo.vstack(
            [
                mo.md("## Données de modélisation"),
                mo.callout(
                    mo.md(
                        "**Target** : `is_last_lightning_cloud_ground`  \n"
                        "Forte imbalance : 1 positif par alerte, "
                        "N-1 négatifs (N = nombre d'éclairs CG dans l'alerte)"
                    ),
                    kind="warn",
                ),
                stats,
            ]
        )
    )
    return


@app.cell
def temporal_split(df_model, airports, pl):
    """
    Split 80/20 temporel par aéroport.

    Unité de split : l'ALERTE (pas l'éclair individuel).
    Les alertes sont triées par date de début.
    Les 80% les plus anciennes → train, les 20% les plus récentes → eval.

    Choix documenté : on évite ainsi toute fuite de données entre alertes
    (un split aléatoire au niveau éclair mélangerait des éclairs de la même alerte
    entre train et eval, ce qui constituerait du data leakage).
    """
    splits = {}

    for airport in airports:
        df_ap = df_model.filter(pl.col("airport") == airport)

        alert_dates = (
            df_ap.group_by("airport_alert_id")
            .agg(pl.col("date").min().alias("start"))
            .sort("start")
        )

        n = len(alert_dates)
        n_train = int(n * 0.8)

        train_ids = alert_dates[:n_train]["airport_alert_id"].to_list()
        eval_ids = alert_dates[n_train:]["airport_alert_id"].to_list()

        train_df = df_ap.filter(pl.col("airport_alert_id").is_in(train_ids))
        eval_df = df_ap.filter(pl.col("airport_alert_id").is_in(eval_ids))

        splits[airport] = {
            "train": train_df,
            "eval": eval_df,
            "n_train_alerts": n_train,
            "n_eval_alerts": n - n_train,
            "train_end": str(alert_dates[n_train - 1]["start"][0])[:10],
            "eval_start": str(alert_dates[n_train]["start"][0])[:10],
        }

    return (splits,)


@app.cell
def show_split_stats(splits, mo, pl):
    rows = [
        {
            "Aéroport": airport,
            "Train (alertes)": s["n_train_alerts"],
            "Eval (alertes)": s["n_eval_alerts"],
            "Train (éclairs CG)": len(s["train"]),
            "Eval (éclairs CG)": len(s["eval"]),
            "Train jusqu'au": s["train_end"],
            "Eval à partir du": s["eval_start"],
        }
        for airport, s in splits.items()
    ]

    mo.output.replace(
        mo.vstack(
            [
                mo.md("## Split train / eval par aéroport"),
                mo.callout(
                    mo.md(
                        "Split **temporel** sur les alertes — "
                        "les 80% d'alertes les plus anciennes vont en train, "
                        "les 20% les plus récentes en eval."
                    ),
                    kind="info",
                ),
                pl.DataFrame(rows),
            ]
        )
    )
    return


@app.cell
def train_xgboost(splits, FEATURE_COLS, TARGET_COL, pl, xgb):
    """
    XGBoost — un modèle par aéroport.
    scale_pos_weight = (nb négatifs) / (nb positifs) pour corriger l'imbalance.
    """
    xgb_models = {}

    for _ap, _s in splits.items():
        _X_xgb = _s["train"].select(FEATURE_COLS).to_numpy()
        _y_xgb = _s["train"][TARGET_COL].cast(pl.Int8).to_numpy()

        _pos_weight = int((_y_xgb == 0).sum()) / max(int((_y_xgb == 1).sum()), 1)

        _model_xgb = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        )
        _model_xgb.fit(_X_xgb, _y_xgb)
        xgb_models[_ap] = _model_xgb

    return (xgb_models,)


@app.cell
def train_mlp(
    splits,
    FEATURE_COLS,
    TARGET_COL,
    MLPClassifier,
    StandardScaler,
    compute_sample_weight,
    pl,
):
    """
    MLP — un modèle par aéroport.
    StandardScaler obligatoire avant le MLP.
    sample_weight='balanced' pour corriger l'imbalance (MLPClassifier n'a pas class_weight).
    """
    mlp_models = {}

    for _ap_mlp, _s_mlp in splits.items():
        _X_mlp = _s_mlp["train"].select(FEATURE_COLS).to_numpy()
        _y_mlp = _s_mlp["train"][TARGET_COL].cast(pl.Int8).to_numpy()

        _scaler = StandardScaler()
        _X_mlp_scaled = _scaler.fit_transform(_X_mlp)

        _weights = compute_sample_weight("balanced", _y_mlp)

        _model_mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        _model_mlp.fit(_X_mlp_scaled, _y_mlp, sample_weight=_weights)

        mlp_models[_ap_mlp] = {"scaler": _scaler, "model": _model_mlp}

    return (mlp_models,)


@app.cell
def evaluate_models(
    splits,
    xgb_models,
    mlp_models,
    FEATURE_COLS,
    TARGET_COL,
    np,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    pl,
    mo,
):
    """
    Évaluation sur le set eval de chaque aéroport.

    POD (= recall classe 1) = TP / (TP + FN)
        → fraction des vrais derniers éclairs correctement identifiés
    FAR (= 1 - precision) = FP / (FP + TP)
        → fraction des prédictions positives qui étaient fausses (dangereux si élevé)
    F1 = moyenne harmonique precision/recall
    AUC-ROC = performance globale indépendante du threshold
    Threshold fixé à 0.5 pour cette première évaluation.
    """
    results = []

    for _ap_eval in sorted(splits.keys()):
        _s_eval = splits[_ap_eval]
        _X_eval = _s_eval["eval"].select(FEATURE_COLS).to_numpy()
        _y_eval = _s_eval["eval"][TARGET_COL].cast(pl.Int8).to_numpy()

        if len(_y_eval) == 0 or _y_eval.sum() == 0:
            continue

        _xgb_proba = xgb_models[_ap_eval].predict_proba(_X_eval)[:, 1]

        _X_eval_scaled = mlp_models[_ap_eval]["scaler"].transform(_X_eval)
        _mlp_proba = mlp_models[_ap_eval]["model"].predict_proba(_X_eval_scaled)[:, 1]

        for _mname, _proba in [("XGBoost", _xgb_proba), ("MLP", _mlp_proba)]:
            _y_pred = (_proba >= 0.5).astype(int)

            _pod = recall_score(_y_eval, _y_pred, zero_division=0)
            _prec = precision_score(_y_eval, _y_pred, zero_division=0)
            _far = 1.0 - _prec if _prec > 0 else 1.0
            _f1 = f1_score(_y_eval, _y_pred, zero_division=0)
            _auc = roc_auc_score(_y_eval, _proba)

            results.append(
                {
                    "Aéroport": _ap_eval,
                    "Modèle": _mname,
                    "POD": round(_pod, 3),
                    "FAR": round(_far, 3),
                    "F1": round(_f1, 3),
                    "AUC-ROC": round(_auc, 3),
                    "n_positifs_eval": int(_y_eval.sum()),
                }
            )

    results_df = pl.DataFrame(results)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("## Résultats — XGBoost vs MLP par aéroport"),
                mo.callout(
                    mo.md(
                        "**Threshold** : 0.5 (non optimisé)  \n"
                        "Priorité : **FAR bas** (fausse cessation = danger). "
                        "POD élevé en second objectif."
                    ),
                    kind="info",
                ),
                results_df,
            ]
        )
    )
    return (results_df,)


@app.cell
def feature_importance(xgb_models, FEATURE_COLS, alt, mo, pl):
    """Importance des features XGBoost (gain moyen) — moyennée sur tous les aéroports."""
    importance_rows = []

    for _ap_imp, _model_imp in xgb_models.items():
        _scores = _model_imp.get_booster().get_score(importance_type="gain")
        for _feat, _score in _scores.items():
            importance_rows.append({"feature": _feat, "airport": _ap_imp, "gain": _score})

    if not importance_rows:
        mo.stop(True, mo.md("Aucune importance disponible."))

    imp_df = pl.DataFrame(importance_rows)

    mean_imp = (
        imp_df.group_by("feature")
        .agg(pl.col("gain").mean().alias("gain_moyen"))
        .sort("gain_moyen", descending=True)
        .with_columns(
            pl.Series("feature_order", list(range(len(imp_df.group_by("feature")))))
        )
    )

    chart = (
        alt.Chart(mean_imp.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("gain_moyen:Q", title="Gain moyen"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", "gain_moyen"],
        )
        .properties(width=600, height=350, title="Importance des features (XGBoost, gain moyen toutes aéroports)")
    )

    mo.output.replace(
        mo.vstack([mo.md("## Importance des features XGBoost"), chart])
    )
    return


if __name__ == "__main__":
    app.run()
