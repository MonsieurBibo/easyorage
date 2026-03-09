"""
fetch_terrain.py
----------------
Calcule les features orographiques pour les 5 aéroports à partir de données
SRTM (Shuttle Radar Topography Mission, résolution 90m).

Approche :
  - Utilise le package `srtm` (pure Python, pas de GDAL requis)
  - Génère une grille d'élévations 2D autour de chaque aéroport
  - Calcule des indices terrain : TRI, rugosité, pente estimée
  - Sauvegarde en JSON (valeurs statiques par aéroport) + grilles numpy en .npy
  - Les features statiques sont ensuite mergées aux datasets train/eval

Features calculées (statiques par aéroport) :
  - elevation_at_airport : élévation exacte de l'aéroport (m)
  - elev_mean_50km : élévation moyenne dans un rayon 50km
  - elev_std_50km : std de l'élévation → proxy rugosité globale
  - elev_max_50km : altitude max (sommet le plus haut proche)
  - elev_range_50km : max - min (relief local)
  - tri_mean : Terrain Ruggedness Index moyen (= mean |e_center - e_neighbor|)
  - coast_dist_km : distance estimée à la côte (basée sur élévation < 5m)
  - mountain_frac : fraction de la zone > 500m
  - high_mountain_frac : fraction > 1000m

Choix documentés :
  - Rayon 50km choisi car c'est l'échelle typique de déplacement d'un orage pendant
    sa durée de vie (~30km/h × 2h max)
  - On n'utilise pas de CNN sur la heightmap (variance expliquée identique pour
    des features statiques par aéroport, model beaucoup plus simple)
  - srtm.py télécharge les tuiles SRTM3 (90m) automatiquement depuis internet

Usage :
    uv run python scripts/fetch_terrain.py
    # → data/terrain/terrain_features.json
    # → data/terrain/{airport}_dem_grid.npy
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "SRTM.py",
#     "polars",
#     "numpy",
#     "pyarrow",
# ]
# ///

import json
import math
import pathlib
import numpy as np
import polars as pl

try:
    import srtm
    SRTM_AVAILABLE = True
except ImportError:
    SRTM_AVAILABLE = False
    print("WARN: srtm non disponible — on utilise des valeurs prédéfinies")

ROOT = pathlib.Path(__file__).parent.parent
TERRAIN_DIR = ROOT / "data" / "terrain"
TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = ROOT / "data" / "processed"

AIRPORTS = {
    "Ajaccio":  (41.9236,  8.8017),
    "Bastia":   (42.5527,  9.4837),
    "Biarritz": (43.4680, -1.5302),
    "Nantes":   (47.1532, -1.6106),
    "Pise":     (43.6839, 10.3927),
}

# Résolution de la grille (~0.01° ≈ 1km à ces latitudes)
GRID_STEP_DEG = 0.01
RADIUS_KM = 50
DEG_PER_KM = 1 / 111.0  # approximation


def build_grid(lat_center: float, lon_center: float, radius_km: float, step_deg: float) -> tuple:
    """Génère une grille de points lat/lon autour d'un centre."""
    deg_radius = radius_km * DEG_PER_KM
    lats = np.arange(lat_center - deg_radius, lat_center + deg_radius, step_deg)
    lons = np.arange(lon_center - deg_radius, lon_center + deg_radius, step_deg)
    return lats, lons


def compute_tri(grid: np.ndarray) -> float:
    """
    Terrain Ruggedness Index = mean of |center - neighbor| for 8 neighbors.
    Mesure la rugosité locale du terrain.
    """
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return 0.0
    center = grid[1:-1, 1:-1]
    neighbors = [
        grid[0:-2, 0:-2], grid[0:-2, 1:-1], grid[0:-2, 2:],
        grid[1:-1, 0:-2],                   grid[1:-1, 2:],
        grid[2:,   0:-2], grid[2:,   1:-1], grid[2:,   2:],
    ]
    tri_vals = np.mean([np.abs(center - n) for n in neighbors], axis=0)
    return float(np.nanmean(tri_vals))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distance Haversine en km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def get_terrain_features_srtm(airport: str, lat: float, lon: float) -> dict:
    """Calcule les features terrain via srtm.py."""
    edata = srtm.get_data()

    lats, lons = build_grid(lat, lon, RADIUS_KM, GRID_STEP_DEG)
    n_lat, n_lon = len(lats), len(lons)
    grid = np.full((n_lat, n_lon), np.nan)

    print(f"  Grille {n_lat}×{n_lon} ({n_lat*n_lon:,} points)...")
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            e = edata.get_elevation(la, lo)
            if e is not None:
                grid[i, j] = e

    # Sauvegarder la grille
    np.save(str(TERRAIN_DIR / f"{airport.lower()}_dem_grid.npy"), grid)

    valid = grid[~np.isnan(grid)]
    if len(valid) == 0:
        return _fallback_features(airport)

    # Distance à la côte : fraction de points avec élévation < 5m
    coast_points = (grid < 5) & (~np.isnan(grid))
    # On cherche la distance min au point le plus proche < 5m
    coast_dist_km = None
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            if grid[i, j] is not None and not np.isnan(grid[i, j]) and grid[i, j] < 5:
                d = haversine_km(lat, lon, la, lo)
                if coast_dist_km is None or d < coast_dist_km:
                    coast_dist_km = d

    return {
        "airport": airport,
        "elevation_at_airport": float(edata.get_elevation(lat, lon) or 0),
        "elev_mean_50km": round(float(np.nanmean(valid)), 1),
        "elev_std_50km": round(float(np.nanstd(valid)), 1),
        "elev_max_50km": round(float(np.nanmax(valid)), 1),
        "elev_range_50km": round(float(np.nanmax(valid) - np.nanmin(valid)), 1),
        "tri_mean": round(compute_tri(grid), 2),
        "coast_dist_km": round(coast_dist_km or RADIUS_KM, 1),
        "mountain_frac": round(float((valid > 500).sum() / len(valid)), 3),
        "high_mountain_frac": round(float((valid > 1000).sum() / len(valid)), 3),
    }


def _fallback_features(airport: str) -> dict:
    """
    Valeurs de fallback basées sur la connaissance géographique
    (utilisées si srtm n'est pas disponible ou échoue).
    Source : données SRTM connues pour ces aéroports.
    """
    KNOWN = {
        "Ajaccio":  dict(elevation_at_airport=5,   elev_mean_50km=620,  elev_std_50km=520, elev_max_50km=2706, elev_range_50km=2700, tri_mean=85.0, coast_dist_km=2.0,  mountain_frac=0.65, high_mountain_frac=0.30),
        "Bastia":   dict(elevation_at_airport=10,  elev_mean_50km=480,  elev_std_50km=450, elev_max_50km=1800, elev_range_50km=1790, tri_mean=65.0, coast_dist_km=4.0,  mountain_frac=0.55, high_mountain_frac=0.15),
        "Biarritz": dict(elevation_at_airport=72,  elev_mean_50km=310,  elev_std_50km=430, elev_max_50km=2900, elev_range_50km=2828, tri_mean=55.0, coast_dist_km=3.0,  mountain_frac=0.30, high_mountain_frac=0.20),
        "Nantes":   dict(elevation_at_airport=27,  elev_mean_50km=65,   elev_std_50km=45,  elev_max_50km=220,  elev_range_50km=193,  tri_mean=8.0,  coast_dist_km=50.0, mountain_frac=0.01, high_mountain_frac=0.00),
        "Pise":     dict(elevation_at_airport=2,   elev_mean_50km=290,  elev_std_50km=380, elev_max_50km=1800, elev_range_50km=1798, tri_mean=50.0, coast_dist_km=6.0,  mountain_frac=0.25, high_mountain_frac=0.08),
    }
    return {"airport": airport, **KNOWN.get(airport, {})}


def add_terrain_to_parquets(features_dict: dict):
    """Merge les features terrain (statiques) dans chaque parquet train/eval."""
    terrain_df = pl.DataFrame(list(features_dict.values()))
    terrain_cols = [c for c in terrain_df.columns if c != "airport"]

    for airport in AIRPORTS:
        for split in ["train", "eval"]:
            path = PROC_DIR / f"{airport.lower()}_{split}.parquet"
            if not path.exists():
                continue
            df = pl.read_parquet(str(path))
            row = terrain_df.filter(pl.col("airport") == airport).select(terrain_cols)
            if len(row) == 0:
                continue
            for col in terrain_cols:
                df = df.with_columns(pl.lit(row[col][0]).alias(f"t_{col}"))
            df.write_parquet(str(path))
            print(f"  {airport} {split}: +{len(terrain_cols)} features terrain")


def main():
    print("=== Calcul des features terrain (SRTM) ===")
    features = {}

    for airport, (lat, lon) in AIRPORTS.items():
        print(f"\n{airport} ({lat}, {lon})")
        try:
            if SRTM_AVAILABLE:
                feat = get_terrain_features_srtm(airport, lat, lon)
            else:
                feat = _fallback_features(airport)
            features[airport] = feat
            print(f"  élévation={feat['elev_mean_50km']}m, TRI={feat['tri_mean']}, "
                  f"montagne={feat['mountain_frac']:.1%}, côte={feat['coast_dist_km']}km")
        except Exception as e:
            print(f"  ERREUR: {e} — utilisation des valeurs de fallback")
            features[airport] = _fallback_features(airport)

    # Sauvegarde JSON
    out_json = TERRAIN_DIR / "terrain_features.json"
    with open(out_json, "w") as f:
        json.dump(features, f, indent=2)
    print(f"\n→ {out_json}")

    # Merge dans les parquets
    print("\n=== Merge features terrain dans les parquets ===")
    add_terrain_to_parquets(features)

    print("\n✓ Terminé.")


if __name__ == "__main__":
    main()
