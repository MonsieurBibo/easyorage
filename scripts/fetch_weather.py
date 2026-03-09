"""
fetch_weather.py
----------------
Télécharge les données météo horaires historiques via Open-Meteo (ERA5)
pour les 5 aéroports sur 2016-2025, puis les joint aux alertes orageuses.

Source : https://archive-api.open-meteo.com (gratuit, sans clé API, ERA5 depuis 1940)

Variables téléchargées (pertinentes pour la cessation d'orage) :
  - cape : Convective Available Potential Energy (J/kg) — intensité potentielle de l'orage
  - lifted_index : négatif → atmosphère instable, orage possible
  - convective_inhibition : résistance à la convection (J/kg)
  - k_index : indice de stabilité classique en aviation
  - temperature_850hPa, 700hPa, 500hPa : profil thermique vertical
  - relative_humidity_850hPa, 700hPa : humidité à altitude
  - wind_speed_850hPa, wind_direction_850hPa : vent basse couche
  - wind_speed_500hPa : vent d'altitude (cisaillement)
  - boundary_layer_height : hauteur de la couche limite
  - total_column_integrated_water_vapour : eau précipitable totale
  - precipitation : précip horaire (proxy activité convective)
  - cloud_cover : couverture nuageuse

Choix documentés :
  - On télécharge par année (1 requête/aéroport/an) → 50 requêtes totales
  - On joint aux alertes par heure UTC (round à l'heure la plus proche)
  - Le CAPE est le prédicteur météo le plus fort pour l'intensité convective
  - On sauvegarde en parquet pour éviter de re-télécharger

Usage :
    uv run python scripts/fetch_weather.py
    # → data/weather/{airport}_{year}.parquet
    # → data/processed/{airport}_weather_features.parquet
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "requests",
# ]
# ///

import pathlib
import time
import json
import requests
import polars as pl

ROOT = pathlib.Path(__file__).parent.parent
WEATHER_DIR = ROOT / "data" / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = ROOT / "data" / "processed"

AIRPORTS = {
    "Ajaccio":  (41.9236,  8.8017),
    "Bastia":   (42.5527,  9.4837),
    "Biarritz": (43.4680, -1.5302),
    "Nantes":   (47.1532, -1.6106),
    "Pise":     (43.6839, 10.3927),
}

YEARS = list(range(2016, 2026))

HOURLY_VARS = [
    "cape",
    "lifted_index",
    "convective_inhibition",
    "k_index",
    "temperature_850hPa",
    "relative_humidity_850hPa",
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "temperature_700hPa",
    "relative_humidity_700hPa",
    "wind_speed_700hPa",
    "temperature_500hPa",
    "wind_speed_500hPa",
    "boundary_layer_height",
    "total_column_integrated_water_vapour",
    "precipitation",
    "cloud_cover",
]

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_year(airport: str, lat: float, lon: float, year: int) -> pl.DataFrame:
    """Télécharge les données horaires ERA5 pour un aéroport et une année."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pl.DataFrame({
        "datetime_utc": pl.Series(data["time"]).str.to_datetime("%Y-%m-%dT%H:%M"),
        **{var: pl.Series(data.get(var, [None] * len(data["time"])), dtype=pl.Float32)
           for var in HOURLY_VARS},
    }).with_columns(pl.lit(airport).alias("airport"))

    return df


def download_all(force: bool = False):
    """Télécharge toutes les données météo (cache par fichier annuel)."""
    for airport, (lat, lon) in AIRPORTS.items():
        yearly = []
        for year in YEARS:
            path = WEATHER_DIR / f"{airport.lower()}_{year}.parquet"
            if path.exists() and not force:
                df = pl.read_parquet(str(path))
            else:
                print(f"  Téléchargement {airport} {year}...")
                try:
                    df = fetch_year(airport, lat, lon, year)
                    df.write_parquet(str(path))
                    time.sleep(0.5)  # politesse vis-à-vis de l'API
                except Exception as e:
                    print(f"  ERREUR {airport} {year}: {e}")
                    continue
            yearly.append(df)
            print(f"  {airport} {year}: {len(df)} lignes")

        if yearly:
            all_years = pl.concat(yearly)
            out = WEATHER_DIR / f"{airport.lower()}_all.parquet"
            all_years.write_parquet(str(out))
            print(f"  → {airport} complet: {len(all_years):,} lignes → {out.name}")


def join_weather_to_alerts():
    """
    Joint les données météo aux alertes (déjà splittées en parquet).
    Pour chaque éclair CG en alerte, on récupère la météo de l'heure la plus proche.
    """
    weather_feat_cols = [f"w_{v}" for v in HOURLY_VARS]

    for airport in AIRPORTS:
        for split in ["train", "eval"]:
            alert_path = PROC_DIR / f"{airport.lower()}_{split}.parquet"
            if not alert_path.exists():
                print(f"  Skip {airport} {split} (pas de parquet)")
                continue

            weather_path = WEATHER_DIR / f"{airport.lower()}_all.parquet"
            if not weather_path.exists():
                print(f"  Skip {airport} (pas de météo)")
                continue

            df_alerts = pl.read_parquet(str(alert_path))
            df_weather = pl.read_parquet(str(weather_path)).rename(
                {v: f"w_{v}" for v in HOURLY_VARS}
            )

            # Arrondir la date de l'éclair à l'heure la plus proche pour le join
            df_alerts = df_alerts.with_columns(
                pl.col("date")
                .dt.replace_time_zone(None)  # supprimer timezone avant join
                .dt.round("1h")
                .alias("_join_hour"),
            )

            df_merged = df_alerts.join(
                df_weather.rename({"datetime_utc": "_join_hour"}),
                on="_join_hour",
                how="left",
            ).drop("_join_hour")

            out_path = PROC_DIR / f"{airport.lower()}_{split}_weather.parquet"
            df_merged.write_parquet(str(out_path))
            n_matched = df_merged.select(pl.col("w_cape").is_not_null().sum()).item()
            print(
                f"  {airport} {split}: {len(df_merged):,} rows, "
                f"{n_matched:,} avec météo → {out_path.name}"
            )


def main():
    print("=== Téléchargement données météo Open-Meteo (ERA5) ===")
    download_all()
    print("\n=== Jointure avec les alertes ===")
    join_weather_to_alerts()
    print("\n✓ Terminé.")


if __name__ == "__main__":
    main()
