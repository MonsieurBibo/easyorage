"""
compute_features.py
-------------------
Calcule toutes les features à partir des données brutes et sauvegarde en parquet
par aéroport (train + eval).

Choix documentés :
- Toutes les features sont calculées sans lookahead (données passées uniquement)
- Rolling windows : count-based (3, 5, 10 derniers éclairs), pas time-based
  → plus robuste aux variations de densité d'éclairs
- Flash rate "instantané" : cumul / temps_total (tendance globale)
- Tendances (trend_*) : valeur courante - moyenne des N derniers → signal de changement
- Dispersion spatiale : std de dist et azimuth depuis le début de l'alerte
- Features temporelles : heure UTC et mois pour les patterns diurnes/saisonniers

Usage :
    uv run python scripts/compute_features.py
    # → data/processed/{airport}_train.parquet
    # → data/processed/{airport}_eval.parquet
    # → data/processed/feature_cols.json
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pyarrow",
# ]
# ///

import json
import pathlib
import polars as pl

ROOT = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT / "data_train_databattle2026" / "segment_alerts_all_airports_train.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.8


def load_raw() -> pl.DataFrame:
    return (
        pl.read_csv(
            str(DATA_PATH),
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
        .with_columns(pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S%z"))
        .filter(pl.col("airport_alert_id").is_not_null())
    )


def rolling_n(expr: pl.Expr, n: int, agg: str = "mean") -> pl.Expr:
    """Rolling over last N rows (including current). min_samples=1."""
    if agg == "mean":
        return expr.rolling_mean(window_size=n, min_samples=1)
    elif agg == "std":
        return expr.rolling_std(window_size=n, min_samples=1)
    elif agg == "sum":
        return expr.rolling_sum(window_size=n, min_samples=1)
    raise ValueError(agg)


def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcule ~35 features par éclair CG en alerte.
    Toutes les features respectent la causalité (pas de lookahead).
    """
    df_sorted = df.sort("airport", "airport_alert_id", "date")

    # ── 1. features brutes dérivées ──────────────────────────────────────────
    df1 = df_sorted.with_columns(
        pl.col("amplitude").abs().alias("amplitude_abs"),
        # heure UTC et mois pour patterns diurnes/saisonniers
        pl.col("date").dt.hour().alias("hour_utc"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("weekday"),
    )

    # ── 2. features par-alerte (ordre chronologique via over) ─────────────────
    G = ["airport", "airport_alert_id"]

    df2 = df1.with_columns(
        # temps écoulé depuis le 1er éclair (s)
        (pl.col("date") - pl.col("date").min().over(G))
        .dt.total_seconds()
        .alias("time_since_start_s"),
        # rang dans l'alerte (1 = 1er éclair)
        pl.col("date").rank("ordinal").over(G).alias("lightning_rank"),
        # intervalle inter-éclair (s), 0 pour le 1er
        pl.col("date")
        .diff()
        .over(G)
        .dt.total_seconds()
        .fill_null(0.0)
        .alias("ili_s"),
        # comptage cumulatif
        (pl.col("icloud") == False)
        .cast(pl.Int32)
        .cum_sum()
        .over(G)
        .alias("n_cg_so_far"),
        # moyennes cumulatives
        (
            pl.col("amplitude_abs").cum_sum().over(G)
            / pl.col("amplitude_abs").cum_count().over(G)
        ).alias("mean_amplitude_so_far"),
        (
            pl.col("dist").cum_sum().over(G) / pl.col("dist").cum_count().over(G)
        ).alias("mean_dist_so_far"),
    )

    # ── 3. rolling count-based (3, 5, 10 derniers éclairs) ───────────────────
    # Ces expressions sont appliquées dans l'ordre du DataFrame trié,
    # par groupe (over G), donc pas de lookahead.
    df3 = df2.with_columns(
        # ILI rolling
        rolling_n(pl.col("ili_s"), 3).over(G).alias("rolling_ili_3"),
        rolling_n(pl.col("ili_s"), 5).over(G).alias("rolling_ili_5"),
        rolling_n(pl.col("ili_s"), 10).over(G).alias("rolling_ili_10"),
        rolling_n(pl.col("ili_s"), 5, "std").over(G).alias("rolling_ili_std_5"),
        # Amplitude rolling
        rolling_n(pl.col("amplitude_abs"), 3).over(G).alias("rolling_amp_3"),
        rolling_n(pl.col("amplitude_abs"), 5).over(G).alias("rolling_amp_5"),
        rolling_n(pl.col("amplitude_abs"), 10).over(G).alias("rolling_amp_10"),
        rolling_n(pl.col("amplitude_abs"), 5, "std").over(G).alias("rolling_amp_std_5"),
        # Distance rolling
        rolling_n(pl.col("dist"), 5).over(G).alias("rolling_dist_5"),
        rolling_n(pl.col("dist"), 10).over(G).alias("rolling_dist_10"),
    )

    # ── 4. tendances et dérivées ──────────────────────────────────────────────
    df4 = df3.with_columns(
        # tendance ILI : l'intervalle courant vs moyenne des 5 derniers
        # positif = intervalle s'allonge → orage qui se calme
        (pl.col("ili_s") - pl.col("rolling_ili_5")).alias("ili_trend"),
        # tendance amplitude : négatif = éclairs plus faibles que la moyenne récente
        (pl.col("amplitude_abs") - pl.col("rolling_amp_5")).alias("amp_trend"),
        # tendance distance : positif = éclairs qui s'éloignent
        (pl.col("dist") - pl.col("rolling_dist_5")).alias("dist_trend"),
        # flash rate global : éclairs par minute depuis le début
        (pl.col("n_cg_so_far") / (pl.col("time_since_start_s") / 60 + 1e-6)).alias(
            "flash_rate_global"
        ),
        # dispersion spatiale cumulée
        (
            pl.col("dist").cum_sum().over(G) ** 2 / pl.col("dist").cum_count().over(G)
            - (pl.col("dist").cum_sum().over(G) / pl.col("dist").cum_count().over(G))
            ** 2
        )
        .sqrt()
        .fill_nan(0.0)
        .alias("dist_spread"),
    )

    return df4


def temporal_split(df: pl.DataFrame, airport: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split 80/20 temporel sur les alertes d'un aéroport donné."""
    df_ap = df.filter(pl.col("airport") == airport)
    alert_dates = (
        df_ap.group_by("airport_alert_id")
        .agg(pl.col("date").min().alias("start"))
        .sort("start")
    )
    n = len(alert_dates)
    n_train = int(n * TRAIN_FRAC)
    train_ids = alert_dates[:n_train]["airport_alert_id"].to_list()
    eval_ids = alert_dates[n_train:]["airport_alert_id"].to_list()
    return (
        df_ap.filter(pl.col("airport_alert_id").is_in(train_ids)),
        df_ap.filter(pl.col("airport_alert_id").is_in(eval_ids)),
    )


FEATURE_COLS = [
    # brutes
    "dist", "amplitude_abs", "azimuth", "maxis",
    # temporelles
    "time_since_start_s", "lightning_rank", "ili_s",
    "hour_utc", "month",
    # cumulatives
    "n_cg_so_far", "mean_amplitude_so_far", "mean_dist_so_far",
    # rolling ILI
    "rolling_ili_3", "rolling_ili_5", "rolling_ili_10", "rolling_ili_std_5",
    # rolling amplitude
    "rolling_amp_3", "rolling_amp_5", "rolling_amp_10", "rolling_amp_std_5",
    # rolling dist
    "rolling_dist_5", "rolling_dist_10",
    # tendances
    "ili_trend", "amp_trend", "dist_trend",
    # dérivées
    "flash_rate_global", "dist_spread",
]

TARGET_COL = "is_last_lightning_cloud_ground"


def main():
    print("Chargement des données...")
    df_raw = load_raw()
    print(f"  {len(df_raw):,} éclairs CG en alerte")

    print("Calcul des features...")
    df_feat = compute_features(df_raw)

    airports = sorted(df_feat["airport"].unique().to_list())
    print(f"  Aéroports : {airports}")

    for airport in airports:
        train_df, eval_df = temporal_split(df_feat, airport)
        n_train_alerts = train_df["airport_alert_id"].n_unique()
        n_eval_alerts = eval_df["airport_alert_id"].n_unique()

        train_path = OUT_DIR / f"{airport.lower()}_train.parquet"
        eval_path = OUT_DIR / f"{airport.lower()}_eval.parquet"
        train_df.write_parquet(str(train_path))
        eval_df.write_parquet(str(eval_path))

        print(
            f"  {airport}: {n_train_alerts} alertes train / {n_eval_alerts} eval "
            f"→ {train_path.name}, {eval_path.name}"
        )

    # Sauvegarde la liste des features pour les notebooks
    meta = {"feature_cols": FEATURE_COLS, "target_col": TARGET_COL}
    with open(OUT_DIR / "feature_cols.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ {len(FEATURE_COLS)} features sauvegardées dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
