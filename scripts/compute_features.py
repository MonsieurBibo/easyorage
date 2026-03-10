"""
compute_features.py
-------------------
Calcule toutes les features à partir des données brutes et sauvegarde en parquet
par aéroport (train + eval).

Choix documentés :
- Toutes les features sont calculées sans lookahead (données passées uniquement)
- Rolling windows : count-based (3, 5, 10 derniers éclairs), pas time-based
  → plus robuste aux variations de densité d'éclairs
  → rolling_sum_by (time-based) non supporté avec .over() dans Polars 1.38
- Flash rate "instantané" : cumul / temps_total (tendance globale)
- Flash rate local : N-1 / rolling_sum(ILI, N) → mesure de la décroissance
- Tendances (trend_*) : valeur courante - moyenne des N derniers → signal de changement
- Dispersion spatiale : std de dist et azimuth depuis le début de l'alerte
- Features temporelles : heure UTC et mois pour les patterns diurnes/saisonniers
- Terrain SRTM : features statiques par aéroport (élévation, TRI, fraction montagne)
- Météo ERA5 : CAPE, lifted_index, etc. si disponibles (jointure horaire)

Nouvelles features v2 (par rapport à v1) :
- rolling_ili_max_5/10 : ILI maximum sur les N derniers éclairs
  → le silence le plus long est plus indicatif de fin d'orage que la moyenne
- ili_log : log(ili_s + 1), linéarise la relation avec la cessation
- flash_rate_5/10 : taux d'éclairs local sur les 5/10 derniers éclairs
  → capture la décroissance récente (vs flash_rate_global = tendance totale)
- flash_rate_ratio : flash_rate_5 / flash_rate_global < 1 = orage qui ralentit
- log_flash_rate_global : mieux conditionné pour les modèles linéaires
- positive_cg_frac : fraction d'éclairs CG+ (amplitude > 0) → signal de polarité
- azimuth_spread_10 : dispersion angulaire des 10 derniers éclairs
- dist_from_edge : 20.0 - dist → proximité du bord du rayon d'alerte
- Terrain SRTM (8 features statiques par aéroport)
- Weather ERA5 (17 features horaires si disponibles)

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
import math
import pathlib
import polars as pl

ROOT = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT / "data_train_databattle2026" / "segment_alerts_all_airports_train.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TERRAIN_JSON = ROOT / "data" / "terrain" / "terrain_features.json"
WEATHER_DIR = ROOT / "data" / "weather"

TRAIN_FRAC = 0.8

HOURLY_VARS = [
    "cape", "lifted_index", "convective_inhibition", "k_index",
    "temperature_850hPa", "relative_humidity_850hPa",
    "wind_speed_850hPa", "wind_direction_850hPa",
    "temperature_700hPa", "relative_humidity_700hPa", "wind_speed_700hPa",
    "temperature_500hPa", "wind_speed_500hPa",
    "boundary_layer_height", "total_column_integrated_water_vapour",
    "precipitation", "cloud_cover",
]

# Clés DEM avancées (calculées par compute_dem_features.py)
DEM_ADV_KEYS = [
    "dem_grad_ew_mean", "dem_grad_ns_mean", "dem_grad_mag_mean", "dem_grad_mag_std",
    "dem_grad_ew_asymmetry", "dem_grad_ns_asymmetry",
    "dem_slope_mean_deg", "dem_slope_p90_deg", "dem_frac_steep",
    "dem_rough_3px_mean", "dem_rough_7px_mean", "dem_rough_15px_mean", "dem_rough_ratio_3_15",
    "dem_tpi_5px_std", "dem_tpi_15px_std", "dem_tpi_15_pos_frac",
    "dem_elev_skew", "dem_frac_negative",
]


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


# Percentiles ILI et rolling_ili_max_5 par aéroport — calculés sur les sets TRAIN.
# Servent à normaliser les features ILI de façon airport-spécifique.
# Justification : chaque aéroport a sa propre distribution ILI (Nantes ~24s médiane vs
# Ajaccio ~32s) → une valeur de 400s ILI signifie ~P95 pour Ajaccio mais ~P90 pour Pise.
ILI_PERCENTILES = {
    # (P75_ili_s, P95_ili_s, P95_rolling_ili_max_5)
    "Ajaccio":  (86.0,  383.0, 762.0),
    "Bastia":   (86.0,  416.0, 866.0),
    "Biarritz": (96.0,  400.0, 808.0),
    "Nantes":   (84.0,  418.0, 825.0),
    "Pise":     (84.0,  445.0, 893.0),
}


def rolling_n(expr: pl.Expr, n: int, agg: str = "mean") -> pl.Expr:
    """Rolling over last N rows (including current). min_samples=1."""
    if agg == "mean":
        return expr.rolling_mean(window_size=n, min_samples=1)
    elif agg == "std":
        return expr.rolling_std(window_size=n, min_samples=1)
    elif agg == "sum":
        return expr.rolling_sum(window_size=n, min_samples=1)
    elif agg == "max":
        return expr.rolling_max(window_size=n, min_samples=1)
    elif agg == "min":
        return expr.rolling_min(window_size=n, min_samples=1)
    raise ValueError(agg)


def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcule 57 features lightning par éclair CG en alerte.
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
        # polarité : +CG = amplitude > 0
        (pl.col("amplitude") > 0).cast(pl.Int8).alias("is_positive_cg"),
        # proximité du bord du rayon d'alerte (20km)
        (20.0 - pl.col("dist")).alias("dist_from_edge"),
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
        # comptage cumulatif des CG
        (pl.col("icloud") == False)
        .cast(pl.Int32)
        .cum_sum()
        .over(G)
        .alias("n_cg_so_far"),
        # comptage cumulatif des IC (intra-cloud)
        pl.col("icloud")
        .cast(pl.Int32)
        .cum_sum()
        .over(G)
        .alias("n_ic_so_far"),
        # comptage cumulatif des +CG
        pl.col("is_positive_cg")
        .cast(pl.Int32)
        .cum_sum()
        .over(G)
        .alias("n_pos_cg_so_far"),
        # moyennes cumulatives
        (
            pl.col("amplitude_abs").cum_sum().over(G)
            / pl.col("amplitude_abs").cum_count().over(G)
        ).alias("mean_amplitude_so_far"),
        (
            pl.col("dist").cum_sum().over(G) / pl.col("dist").cum_count().over(G)
        ).alias("mean_dist_so_far"),
    )

    # ── 3. log ILI + rolling count-based ─────────────────────────────────────
    df3 = df2.with_columns(
        # log ILI : linéarise la relation exponentielle avec la cessation
        (pl.col("ili_s") + 1.0).log(base=math.e).alias("ili_log"),
    ).with_columns(
        # ILI rolling mean
        rolling_n(pl.col("ili_s"), 3).over(G).alias("rolling_ili_3"),
        rolling_n(pl.col("ili_s"), 5).over(G).alias("rolling_ili_5"),
        rolling_n(pl.col("ili_s"), 10).over(G).alias("rolling_ili_10"),
        rolling_n(pl.col("ili_s"), 5, "std").over(G).alias("rolling_ili_std_5"),
        # ILI rolling MAX — la pause la plus longue parmi les N derniers éclairs
        # Signal fort : un long silence récent = orage en fin de vie
        rolling_n(pl.col("ili_s"), 5, "max").over(G).alias("rolling_ili_max_5"),
        rolling_n(pl.col("ili_s"), 10, "max").over(G).alias("rolling_ili_max_10"),
        # ILI rolling MIN — la plus courte pause parmi les N derniers éclairs
        # Si min est petit → orage encore actif (burst récent)
        # Si min est grand → toutes les pauses récentes sont longues → cessation imminente
        rolling_n(pl.col("ili_s"), 5, "min").over(G).alias("rolling_ili_min_5"),
        # Amplitude rolling
        rolling_n(pl.col("amplitude_abs"), 3).over(G).alias("rolling_amp_3"),
        rolling_n(pl.col("amplitude_abs"), 5).over(G).alias("rolling_amp_5"),
        rolling_n(pl.col("amplitude_abs"), 10).over(G).alias("rolling_amp_10"),
        rolling_n(pl.col("amplitude_abs"), 5, "std").over(G).alias("rolling_amp_std_5"),
        # Distance rolling
        rolling_n(pl.col("dist"), 5).over(G).alias("rolling_dist_5"),
        rolling_n(pl.col("dist"), 10).over(G).alias("rolling_dist_10"),
        # Flash rate local — N-1 intervalles pour N éclairs, en éclairs/min
        # flash_rate_5 = 4 éclairs sur rolling_sum(ILI,5) secondes
        # rolling_sum(ILI, 5) = somme des 5 derniers ILI = temps des 5 derniers éclairs
        (2.0 / (rolling_n(pl.col("ili_s"), 3, "sum").over(G) + 1.0) * 60.0).alias("flash_rate_3"),
        (4.0 / (rolling_n(pl.col("ili_s"), 5, "sum").over(G) + 1.0) * 60.0).alias("flash_rate_5"),
        (9.0 / (rolling_n(pl.col("ili_s"), 10, "sum").over(G) + 1.0) * 60.0).alias("flash_rate_10"),
        # Azimuth spread (dispersion angulaire des 10 derniers éclairs)
        pl.col("azimuth")
        .rolling_std(window_size=10, min_samples=2)
        .over(G)
        .fill_nan(0.0)
        .alias("azimuth_spread_10"),
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
        (pl.col("n_cg_so_far") / (pl.col("time_since_start_s") / 60.0 + 1e-6)).alias(
            "flash_rate_global"
        ),
        # dispersion spatiale cumulée (std de la distance)
        (
            pl.col("dist").cum_sum().over(G) ** 2 / pl.col("dist").cum_count().over(G)
            - (pl.col("dist").cum_sum().over(G) / pl.col("dist").cum_count().over(G)) ** 2
        )
        .sqrt()
        .fill_nan(0.0)
        .alias("dist_spread"),
        # fraction de +CG cumulée (polarité)
        (pl.col("n_pos_cg_so_far") / (pl.col("n_cg_so_far") + 1)).alias("positive_cg_frac"),
        # ratio IC:CG cumulé (évolution du cycle de vie de l'orage)
        # IC:CG ratio décroît en dissipation selon certaines études (Orville & Huffines 2001)
        # Mais peut être non-monotone — utile comme contexte du régime d'orage
        (pl.col("n_ic_so_far") / (pl.col("n_cg_so_far") + 1)).alias("ic_cg_ratio"),
    )

    # ── 5. features dérivées de second ordre ──────────────────────────────────
    df5 = df4.with_columns(
        # log flash rate global — mieux conditionné pour les modèles linéaires
        (pl.col("flash_rate_global") + 1e-6).log(base=math.e).alias("log_flash_rate_global"),
        # ratio flash rate local / global — signal de décroissance
        # < 1 : l'orage ralentit, > 1 : l'orage s'accélère
        (pl.col("flash_rate_5") / (pl.col("flash_rate_global") + 1e-6)).alias("flash_rate_ratio"),
        # ILI courant vs ILI moyen sur 10 : accélération de la décélération
        (pl.col("rolling_ili_5") - pl.col("rolling_ili_10")).alias("ili_acceleration"),
        # Flash rate vs son maximum historique dans l'alerte (< 1 = orage en déclin)
        # Basé sur Stano 2010 : la décroissance depuis le pic est le signal de dissipation
        (
            pl.col("flash_rate_5") / (pl.col("flash_rate_5").cum_max().over(G) + 1e-6)
        ).alias("fr_vs_max_ratio"),
        # Éclairs de forte amplitude (> 50 kA) : proxy pour les CG+ (enclume)
        # La fraction de CG+ augmente en dissipation (3.5% → 21%) selon Yang et al. 2018
        (pl.col("amplitude_abs") > 50).cast(pl.Int32).cum_sum().over(G).alias("n_high_amp_so_far"),
        # Tracking du centroïde spatial — où se déplace l'orage ?
        rolling_n(pl.col("lat"), 5).over(G).alias("rolling_lat_5"),
        rolling_n(pl.col("lon"), 5).over(G).alias("rolling_lon_5"),
        rolling_n(pl.col("lat"), 10).over(G).alias("rolling_lat_10"),
        rolling_n(pl.col("lon"), 10).over(G).alias("rolling_lon_10"),
    )

    # ── 6. features de troisième ordre (drift centroïde + spatial) ───────────
    df6 = df5.with_columns(
        # Fraction d'éclairs de forte amplitude (proxy dissipation)
        (pl.col("n_high_amp_so_far") / (pl.col("n_cg_so_far") + 1)).alias("high_amp_frac"),
        # Dérive du centroïde (5 vs 10 derniers éclairs) → direction de déplacement
        # Positif = orage qui se déplace vers le Nord/Est
        (pl.col("rolling_lat_5") - pl.col("rolling_lat_10")).alias("lat_drift"),
        (pl.col("rolling_lon_5") - pl.col("rolling_lon_10")).alias("lon_drift"),
        # Bounding box spatiale des 10 derniers éclairs — proxy de la surface convective
        # Une surface qui rétrécit est un signal de dissipation (Bruning 2013)
        # lat_range_10 et lon_range_10 en degrés → surface en km² avec *111²
        (
            rolling_n(pl.col("lat"), 10, "max").over(G)
            - rolling_n(pl.col("lat"), 10, "min").over(G)
        ).alias("lat_range_10"),
        (
            rolling_n(pl.col("lon"), 10, "max").over(G)
            - rolling_n(pl.col("lon"), 10, "min").over(G)
        ).alias("lon_range_10"),
    ).with_columns(
        # Vitesse du centroïde en km (distance entre centroïdes 5 et 10 éclairs)
        # ~111 km/degré en latitude
        (
            ((pl.col("lat_drift") ** 2 + pl.col("lon_drift") ** 2) ** 0.5) * 111.0
        ).alias("centroid_speed_km"),
        # Surface de la bounding box en km² (proxy convex hull)
        # En dissipation, l'orage se contracte → surface décroît
        (pl.col("lat_range_10") * pl.col("lon_range_10") * 111.0 * 111.0).alias(
            "spatial_bbox_km2"
        ),
    ).with_columns(
        # Densité d'éclairs (éclairs par km²) — signal de concentration/dilution
        (pl.col("n_cg_so_far") / (pl.col("spatial_bbox_km2") + 1.0)).alias(
            "flash_density_km2"
        ),
        # sigma_level : taux de décroissance normalisé du flash rate
        # Adapté du "lightning jump" (Schultz 2009) : ici on cherche l'inverse
        # DFR = dérivée du flash rate local (flash_rate_5)
        # sigma_level = DFR / std(DFR[-5:]) → négatif et grand = décroissance rapide
        # Si sigma_level << -2 : orage en chute libre → cessation imminente
        (
            pl.col("flash_rate_5").diff().over(G).fill_null(0.0)
        ).alias("_dfr"),
    ).with_columns(
        (
            pl.col("_dfr")
            / (rolling_n(pl.col("_dfr"), 5, "std").over(G).fill_null(1.0).replace(0.0, 1.0) + 1e-6)
        ).alias("sigma_level"),
    ).drop("_dfr")

    # ── 7. MIFI percentile — normalisation airport-spécifique ─────────────────
    # ILI max normalisé par la distribution historique de chaque aéroport.
    # Source des percentiles : compute_features.py première passe sur train (2026-03-10)
    # Formula : rolling_ili_max_5 / P95_airport → > 1.0 = pause anormalement longue
    # Justification : normalise le signal ILI par le contexte local de chaque aéroport
    # → Nantes a des ILI plus courts (médiane 24s vs 32s Ajaccio), donc P95=418s vs 383s
    _ili_p95_expr  = pl.lit(None, dtype=pl.Float64)
    _ili_p75_expr  = pl.lit(None, dtype=pl.Float64)
    for _ap_name, (_p75, _p95, _) in ILI_PERCENTILES.items():
        _ili_p95_expr = pl.when(pl.col("airport") == _ap_name).then(pl.lit(_p95)).otherwise(_ili_p95_expr)
        _ili_p75_expr = pl.when(pl.col("airport") == _ap_name).then(pl.lit(_p75)).otherwise(_ili_p75_expr)

    df7 = df6.with_columns(
        (pl.col("rolling_ili_max_5") / (_ili_p95_expr.cast(pl.Float64) + 1e-6)).alias("ili_vs_p95"),
        (pl.col("rolling_ili_max_5") / (_ili_p75_expr.cast(pl.Float64) + 1e-6)).alias("ili_vs_p75"),
    )

    # ── 8. Décroissance du log flash rate (pente OLS simplifiée) ──────────────
    # fr_log_slope ≈ log(flash_rate_5) - log(flash_rate_10)
    # Si négatif → flash rate décroît de façon log-linéaire (exponentielle réelle)
    # Capture mieux la décroissance que le ratio car est additif dans le log
    # Justification : flash rate decay est exponentiel → log-linéaire selon Schultz 2009
    df8 = df7.with_columns(
        (
            (pl.col("flash_rate_5") + 1e-6).log(base=math.e)
            - (pl.col("flash_rate_10") + 1e-6).log(base=math.e)
        ).alias("fr_log_slope"),
        # Pente log très récente (3→5 éclairs) — capture la toute dernière tendance
        # Négatif = décélération dans les 3 derniers éclairs (vs 5→10 pour fr_log_slope)
        # Complémentaire à fr_log_slope qui capte la tendance 5→10
        (
            (pl.col("flash_rate_3") + 1e-6).log(base=math.e)
            - (pl.col("flash_rate_5") + 1e-6).log(base=math.e)
        ).alias("fr_log_slope_3"),
        # Ratio ILI actuel / ILI médian de l'alerte jusqu'ici
        # > 1 = intervalle actuel plus long que la normale pour CETTE alerte
        # Capture la tendance intra-alerte (contexte propre à chaque orage)
        (
            pl.col("ili_s")
            / (pl.col("rolling_ili_5") + 1e-6)
        ).alias("ili_vs_local_mean"),
        # ILI actuel vs maximum historique de l'alerte
        # → 1.0 = égal au record de pause → très proche de la cessation
        # Basé sur l'hypothèse que le dernier éclair arrive après un ILI record
        (
            pl.col("ili_s")
            / (pl.col("ili_s").cum_max().over(G) + 1e-6)
        ).alias("ili_vs_alert_max"),
        # Coefficient de variation de l'ILI sur les 5 derniers éclairs
        # CV grand → intervalles très irréguliers → burst récent peu représentatif
        # CV petit + ILI élevé → décroissance stable → cessation imminente
        (
            pl.col("rolling_ili_std_5") / (pl.col("rolling_ili_5") + 1e-6)
        ).alias("ili_cv_5"),
        # Z-score de l'ILI courant dans la fenêtre des 5 derniers éclairs
        # = (ili - mean) / std → anomalie standardisée
        # Différent de ili_vs_local_mean (ratio) : tient compte de la VARIABILITÉ locale
        # Ex : ILI=200s avec mean=50, std=150 → z=1.0 (pas si surprenant)
        #      ILI=200s avec mean=50, std=10  → z=15.0 (très inhabituel → cessation)
        (
            (pl.col("ili_s") - pl.col("rolling_ili_5"))
            / (pl.col("rolling_ili_std_5") + 1e-6)
        ).alias("ili_z_score_5"),
        # La pause record est-elle récente (dans les 5 derniers éclairs) ?
        # = rolling_ili_max_5 / cum_max(ili_s) dans l'alerte
        # → 1.0 : le maximum de pause est dans les 5 derniers éclairs = signal fort
        # → < 1 : le max était plus tôt, l'orage peut encore être actif
        # Différent de ili_vs_alert_max (qui demande si le ILI COURANT est le record)
        (
            pl.col("rolling_ili_max_5")
            / (pl.col("ili_s").cum_max().over(G) + 1e-6)
        ).alias("rolling_max_vs_alert_max"),
    )

    return df8


DEM_ADVANCED_JSON = ROOT / "data" / "terrain" / "dem_advanced_features.json"

# Clés DEM avancées à extraire (calculées par compute_dem_features.py)
def add_terrain_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ajoute des features statiques de terrain SRTM + DEM avancées par aéroport.
    Sources :
    - terrain_features.json : stats globales (élévation, TRI, fraction montagne)
    - dem_advanced_features.json : gradients, rugosité multi-échelles, TPI (Leinonen 2023)

    Ces features sont identiques pour tous les éclairs d'un même aéroport.
    Justification : l'orographie détermine le régime de convection local et
    donc la durée typique des orages (cf. DISCOVERIES.md — Orographie, Leinonen 2023).
    """
    if not TERRAIN_JSON.exists():
        print("  WARN: terrain_features.json non trouvé — features terrain ignorées")
        return df

    with open(TERRAIN_JSON) as f:
        terrain = json.load(f)

    terrain_cols = {
        "t_elev_mean": {},
        "t_elev_std": {},
        "t_elev_max": {},
        "t_elev_range": {},
        "t_tri_mean": {},
        "t_coast_dist": {},
        "t_mountain_frac": {},
        "t_high_mountain_frac": {},
    }

    for airport_name, feats in terrain.items():
        terrain_cols["t_elev_mean"][airport_name] = feats.get("elev_mean_50km", 0.0)
        terrain_cols["t_elev_std"][airport_name] = feats.get("elev_std_50km", 0.0)
        terrain_cols["t_elev_max"][airport_name] = feats.get("elev_max_50km", 0.0)
        terrain_cols["t_elev_range"][airport_name] = feats.get("elev_range_50km", 0.0)
        terrain_cols["t_tri_mean"][airport_name] = feats.get("tri_mean", 0.0)
        terrain_cols["t_coast_dist"][airport_name] = feats.get("coast_dist_km", 50.0)
        terrain_cols["t_mountain_frac"][airport_name] = feats.get("mountain_frac", 0.0)
        terrain_cols["t_high_mountain_frac"][airport_name] = feats.get("high_mountain_frac", 0.0)

    exprs = []
    for col_name, mapping in terrain_cols.items():
        expr = pl.lit(None, dtype=pl.Float64)
        for ap_name, val in mapping.items():
            expr = pl.when(pl.col("airport") == ap_name).then(pl.lit(val)).otherwise(expr)
        exprs.append(expr.cast(pl.Float64).alias(col_name))

    df = df.with_columns(exprs)

    # Ajout des features DEM avancées (gradients, rugosité, TPI) si disponibles
    if DEM_ADVANCED_JSON.exists():
        with open(DEM_ADVANCED_JSON) as f:
            dem_adv = json.load(f)

        adv_cols = {k: {} for k in DEM_ADV_KEYS}
        for airport_name, feats in dem_adv.items():
            for k in DEM_ADV_KEYS:
                adv_cols[k][airport_name] = feats.get(k, 0.0)

        adv_exprs = []
        for col_name, mapping in adv_cols.items():
            expr = pl.lit(None, dtype=pl.Float64)
            for ap_name, val in mapping.items():
                expr = pl.when(pl.col("airport") == ap_name).then(pl.lit(val)).otherwise(expr)
            adv_exprs.append(expr.cast(pl.Float64).alias(col_name))

        df = df.with_columns(adv_exprs)

    return df


def add_weather_features(df: pl.DataFrame, airport: str) -> pl.DataFrame:
    """
    Joint les features météo ERA5 (Open-Meteo) pour un aéroport si disponibles.
    Jointure par heure UTC arrondie au plus proche.

    Variables clés pour la cessation :
    - cape : énergie convective disponible (↓ = orage en fin de vie)
    - lifted_index : stabilité (↑ = plus stable = cessation probable)
    - convective_inhibition : résistance à la convection
    """
    weather_path = WEATHER_DIR / f"{airport.lower()}_all.parquet"
    if not weather_path.exists():
        return df

    df_weather = pl.read_parquet(str(weather_path)).rename(
        {v: f"w_{v}" for v in HOURLY_VARS if v in pl.read_parquet(str(weather_path)).columns}
    )

    # Arrondir l'heure de l'éclair au plus proche pour le join
    df = df.with_columns(
        pl.col("date")
        .dt.replace_time_zone(None)
        .dt.round("1h")
        .alias("_join_hour")
    )

    df_merged = df.join(
        df_weather.rename({"datetime_utc": "_join_hour"}).drop("airport", strict=False),
        on="_join_hour",
        how="left",
    ).drop("_join_hour")

    return df_merged


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


LIGHTNING_FEATURE_COLS = [
    # brutes
    "dist", "amplitude_abs", "azimuth", "maxis",
    "dist_from_edge",
    # temporelles
    "time_since_start_s", "lightning_rank", "ili_s", "ili_log",
    "hour_utc", "month",
    # cumulatives
    "n_cg_so_far", "n_ic_so_far", "mean_amplitude_so_far", "mean_dist_so_far",
    # rolling ILI (mean + std + max)
    "rolling_ili_3", "rolling_ili_5", "rolling_ili_10",
    "rolling_ili_std_5",
    "rolling_ili_max_5", "rolling_ili_max_10",
    "rolling_ili_min_5",
    # rolling amplitude
    "rolling_amp_3", "rolling_amp_5", "rolling_amp_10", "rolling_amp_std_5",
    # rolling dist
    "rolling_dist_5", "rolling_dist_10",
    # flash rate
    "flash_rate_global", "log_flash_rate_global",
    "flash_rate_3", "flash_rate_5", "flash_rate_10", "flash_rate_ratio",
    "fr_vs_max_ratio",
    # tendances et dérivées
    "ili_trend", "ili_acceleration", "amp_trend", "dist_trend",
    "dist_spread", "azimuth_spread_10",
    # polarité et amplitude
    "positive_cg_frac", "high_amp_frac",
    # IC:CG ratio — proxy du régime de l'orage (cycle de vie)
    "ic_cg_ratio",
    # tracking centroïde (déplacement de l'orage)
    "lat_drift", "lon_drift", "centroid_speed_km",
    # surface spatiale et densité (dissipation = contraction)
    "spatial_bbox_km2", "flash_density_km2",
    # sigma_level : taux de décroissance normalisé (lightning crash)
    "sigma_level",
    # MIFI percentile : ILI max normalisé par la distribution historique de l'aéroport
    # rolling_ili_max_5 / P95_airport → > 1.0 = pause anormalement longue = cessation probable
    # Normalisation airport-spécifique : chaque aéroport a sa propre distribution ILI
    "ili_vs_p95", "ili_vs_p75",
    # décroissance log-linéaire du flash rate (Schultz 2009 : exponentielle)
    "fr_log_slope",
    # pente log très récente (3→5 éclairs) — tendance de la toute dernière fenêtre
    "fr_log_slope_3",
    # ILI courant vs moyenne locale de l'alerte (contexte intra-alerte)
    "ili_vs_local_mean",
    # coefficient de variation ILI sur 5 : régularité des pauses
    # CV petit + ILI élevé = décroissance stable = cessation imminente
    "ili_cv_5",
    # ILI courant vs maximum historique de l'alerte
    # → 1.0 = nouveau record de pause = signal fort de cessation
    "ili_vs_alert_max",
    # Z-score ILI : anomalie standardisée par la variabilité locale
    # Différent du ratio : amplifie le signal quand les ILI sont réguliers
    "ili_z_score_5",
    # Pause record récente : rolling_max_5 / cum_max = 1 si record dans les 5 derniers
    "rolling_max_vs_alert_max",
]

TERRAIN_FEATURE_COLS = [
    "t_elev_mean", "t_elev_std", "t_elev_max", "t_elev_range",
    "t_tri_mean", "t_coast_dist", "t_mountain_frac", "t_high_mountain_frac",
    # DEM avancé — gradients, rugosité multi-échelles, TPI (Leinonen 2023)
    *DEM_ADV_KEYS,
]

WEATHER_FEATURE_COLS = [f"w_{v}" for v in HOURLY_VARS]

TARGET_COL = "is_last_lightning_cloud_ground"


def main():
    print("Chargement des données...")
    df_raw = load_raw()
    print(f"  {len(df_raw):,} éclairs CG en alerte")

    print("Calcul des features lightning...")
    df_feat = compute_features(df_raw)

    print("Ajout des features terrain SRTM...")
    df_feat = add_terrain_features(df_feat)

    airports = sorted(df_feat["airport"].unique().to_list())
    print(f"  Aéroports : {airports}")

    all_feature_cols = list(LIGHTNING_FEATURE_COLS) + list(TERRAIN_FEATURE_COLS)

    for airport in airports:
        print(f"\n  {airport}...")

        # Split temporel
        train_df, eval_df = temporal_split(df_feat, airport)

        # Ajout météo si disponible
        weather_path = WEATHER_DIR / f"{airport.lower()}_all.parquet"
        if weather_path.exists():
            print(f"    Jointure météo ERA5...")
            train_df = add_weather_features(train_df, airport)
            eval_df = add_weather_features(eval_df, airport)
            n_weather = sum(1 for c in train_df.columns if c.startswith("w_"))
            print(f"    → {n_weather} features météo ajoutées")

        n_train_alerts = train_df["airport_alert_id"].n_unique()
        n_eval_alerts = eval_df["airport_alert_id"].n_unique()

        train_path = OUT_DIR / f"{airport.lower()}_train.parquet"
        eval_path = OUT_DIR / f"{airport.lower()}_eval.parquet"
        train_df.write_parquet(str(train_path))
        eval_df.write_parquet(str(eval_path))

        print(
            f"    {n_train_alerts} alertes train / {n_eval_alerts} eval "
            f"→ {train_path.name}, {eval_path.name}"
        )

    # Harmonise les schémas : ajoute des colonnes nulles pour les aéroports
    # sans météo ERA5, pour que tous les parquets aient le même schéma.
    # Permet d'utiliser la même liste de features pour tous les modèles.
    has_weather = any(
        (WEATHER_DIR / f"{ap.lower()}_all.parquet").exists() for ap in airports
    )
    if has_weather:
        for airport in airports:
            for split in ["train", "eval"]:
                pq_path = OUT_DIR / f"{airport.lower()}_{split}.parquet"
                df_pq = pl.read_parquet(str(pq_path))
                missing = [c for c in WEATHER_FEATURE_COLS if c not in df_pq.columns]
                if missing:
                    df_pq = df_pq.with_columns(
                        [pl.lit(None, dtype=pl.Float32).alias(c) for c in missing]
                    )
                    df_pq.write_parquet(str(pq_path))
        effective_weather_cols = WEATHER_FEATURE_COLS
    else:
        effective_weather_cols = []
    effective_feature_cols = all_feature_cols + effective_weather_cols

    # Sauvegarde la liste des features pour les notebooks
    meta = {
        "feature_cols": effective_feature_cols,
        "lightning_feature_cols": LIGHTNING_FEATURE_COLS,
        "terrain_feature_cols": TERRAIN_FEATURE_COLS,
        "weather_feature_cols": effective_weather_cols,
        "target_col": TARGET_COL,
    }
    with open(OUT_DIR / "feature_cols.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ {len(effective_feature_cols)} features totales:")
    print(f"  - {len(LIGHTNING_FEATURE_COLS)} lightning")
    print(f"  - {len(TERRAIN_FEATURE_COLS)} terrain SRTM")
    print(f"  - {len(effective_weather_cols)} météo ERA5")
    print(f"  Sauvegardées dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
