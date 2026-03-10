"""
compute_dem_features.py
-----------------------
Extrait des features avancées à partir des grilles DEM 91×91 (SRTM 30m).

Ces features vont au-delà des statistiques globales déjà dans terrain_features.json.
Elles capturent la structure spatiale du terrain à différentes échelles.

Features calculées (par aéroport) :
1. Gradients directionnels : pente N-S et E-W, magnitude du gradient
2. Rugosité multi-échelles : variance locale à 3×3, 7×7, 15×15 pixels
3. Topographic Position Index (TPI) : écart à la moyenne dans une fenêtre
4. Indice d'exposition au vent (windward/leeward) : asymétrie des gradients
5. Fraction de zones à forte pente (> 15°)

Justification (cf. DISCOVERIES.md — Leinonen 2023) :
- Leinonen intègre 3 canaux DEM : élévation + gradient EW + gradient NS
- Ces features capturent mieux l'effet de déclenchement convectif lié à l'orographie
- La rugosité multi-échelle identifie les crêtes et vallées qui influencent la canalisation des orages

Usage :
    uv run python scripts/compute_dem_features.py
    → data/terrain/dem_advanced_features.json
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy<2",
#     "scipy",
# ]
# ///

import json
import pathlib
import numpy as np
from scipy import ndimage

ROOT = pathlib.Path(__file__).parent.parent
TERRAIN_DIR = ROOT / "data" / "terrain"
OUT_JSON = TERRAIN_DIR / "dem_advanced_features.json"

AIRPORTS = ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]

# Résolution SRTM : ~1km/pixel à 50km de rayon → 91 pixels ≈ 50km × 2 / 1.1km/pixel
# En réalité SRTM 1 arc-seconde ≈ 30m → 91 pixels × 30m ≈ 2.7km
# Mais le grid fait 50km de rayon → résolution effective ≈ 1.1 km/pixel
RESOLUTION_KM = 1.1  # km par pixel (approx)


def compute_gradient(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients N-S et E-W (Sobel). NaN remplis par interpolation simple."""
    dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    gy = ndimage.sobel(dem_filled, axis=0)  # N-S
    gx = ndimage.sobel(dem_filled, axis=1)  # E-W
    return gx, gy


def compute_tpi(dem: np.ndarray, radius_px: int) -> np.ndarray:
    """Topographic Position Index : écart à la moyenne dans une fenêtre circulaire."""
    dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    footprint = np.zeros((2 * radius_px + 1, 2 * radius_px + 1))
    cy, cx = radius_px, radius_px
    for y in range(2 * radius_px + 1):
        for x in range(2 * radius_px + 1):
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius_px ** 2:
                footprint[y, x] = 1
    footprint[cy, cx] = 0  # exclut le pixel central
    footprint = footprint / footprint.sum()
    local_mean = ndimage.convolve(dem_filled, footprint)
    return dem_filled - local_mean


def compute_roughness(dem: np.ndarray, kernel_size: int) -> np.ndarray:
    """Rugosité locale = std locale dans une fenêtre glissante."""
    dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    k = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    local_mean = ndimage.convolve(dem_filled, k)
    local_sq_mean = ndimage.convolve(dem_filled ** 2, k)
    variance = np.maximum(local_sq_mean - local_mean ** 2, 0)
    return np.sqrt(variance)


def extract_features(dem: np.ndarray, airport: str) -> dict:
    """Extrait toutes les features avancées pour un DEM."""
    valid = dem[~np.isnan(dem)]

    gx, gy = compute_gradient(dem)
    grad_magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Pentes en degrés (gradient en m/pixel × m_per_pixel)
    deg_per_px = RESOLUTION_KM * 1000  # m/pixel
    slope_rad = np.arctan(grad_magnitude / deg_per_px)
    slope_deg = np.degrees(slope_rad)

    # Rugosité multi-échelles
    rough_3  = compute_roughness(dem, 3)
    rough_7  = compute_roughness(dem, 7)
    rough_15 = compute_roughness(dem, 15)

    # TPI multi-échelles (+ = crête, - = vallée)
    tpi_5  = compute_tpi(dem, 5)
    tpi_15 = compute_tpi(dem, 15)

    feats = {
        "airport": airport,
        # Gradients directionnels
        "dem_grad_ew_mean": float(np.nanmean(np.abs(gx))),
        "dem_grad_ns_mean": float(np.nanmean(np.abs(gy))),
        "dem_grad_mag_mean": float(np.nanmean(grad_magnitude)),
        "dem_grad_mag_std": float(np.nanstd(grad_magnitude)),
        # Asymétrie directionnelle (leeward vs windward)
        # Pour la France atlantique (Biarritz) les vents viennent de l'Ouest
        "dem_grad_ew_asymmetry": float(np.nanmean(gx)),   # + = pente vers l'E
        "dem_grad_ns_asymmetry": float(np.nanmean(gy)),   # + = pente vers le N
        # Pentes
        "dem_slope_mean_deg": float(np.nanmean(slope_deg)),
        "dem_slope_p90_deg": float(np.nanpercentile(slope_deg, 90)),
        "dem_frac_steep": float((slope_deg > 15).mean()),  # fraction > 15°
        # Rugosité multi-échelles
        "dem_rough_3px_mean": float(np.nanmean(rough_3)),
        "dem_rough_7px_mean": float(np.nanmean(rough_7)),
        "dem_rough_15px_mean": float(np.nanmean(rough_15)),
        "dem_rough_ratio_3_15": float(np.nanmean(rough_3) / (np.nanmean(rough_15) + 1e-6)),
        # TPI — identifie les crêtes vs vallées
        "dem_tpi_5px_std": float(np.nanstd(tpi_5)),
        "dem_tpi_15px_std": float(np.nanstd(tpi_15)),
        "dem_tpi_15_pos_frac": float((tpi_15 > 0).mean()),  # fraction crêtes
        # Elevation stats complémentaires
        "dem_elev_skew": float(
            np.nanmean((valid - np.nanmean(valid)) ** 3) / (np.nanstd(valid) ** 3 + 1e-6)
        ),
        "dem_frac_negative": float((valid < 0).mean()),  # zones maritimes/estuaires
    }
    return feats


def main():
    results = {}
    for airport in AIRPORTS:
        dem_path = TERRAIN_DIR / f"{airport.lower()}_dem_grid.npy"
        if not dem_path.exists():
            print(f"  WARN: {dem_path} non trouvé — skip")
            continue

        dem = np.load(str(dem_path))
        print(f"  {airport}: {dem.shape} · {(~np.isnan(dem)).sum()}/{dem.size} valides")

        feats = extract_features(dem, airport)
        results[airport] = feats

        print(f"    grad_mag={feats['dem_grad_mag_mean']:.1f} | "
              f"slope={feats['dem_slope_mean_deg']:.1f}° | "
              f"rough_15={feats['dem_rough_15px_mean']:.0f}m | "
              f"frac_steep={feats['dem_frac_steep']:.2f}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    n_feats = len(feats) - 1  # sans "airport"
    print(f"\n✓ {n_feats} features DEM avancées par aéroport → {OUT_JSON}")


if __name__ == "__main__":
    main()
