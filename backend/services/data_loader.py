"""
data_loader.py
--------------
Charge et met en cache les données du dataset test scorées par le modèle.

Au premier démarrage : feature engineering complet + scoring → sauvegarde dans
data/processed/test_scored.parquet pour éviter de recalculer à chaque restart.

Structure du cache mémoire :
    _alerts_cache[airport_lower] = [
        {
            alert_id, airport, start_date, end_date, n_flashes, duration_s,
            flashes: [{rank, date, lat, lon, flash_type, dist_km, amplitude, score,
                       prediction_triggered}, ...],
            prediction: {triggered_at_rank, triggered_at_date, confidence} | None
        },
        ...  # triés par start_date desc
    ]
"""

import importlib.util
import json
from datetime import datetime

import pandas as pd
import polars as pl

from backend.config import DATA, DATASET_TEST, ROOT, SCORED_CACHE, AIRPORTS
from backend.services.model_service import get_params, predict_proba

_alerts_cache: dict[str, list[dict]] = {}


# ── Import dynamique de compute_features.py ──────────────────────────────────

def _import_cf():
    spec = importlib.util.spec_from_file_location(
        "compute_features", ROOT / "scripts" / "compute_features.py"
    )
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


# ── Construction du parquet scoré ────────────────────────────────────────────

def _build_scored_parquet(cf) -> pl.DataFrame:
    print("[data_loader] Calcul des features et scores sur dataset_test...")

    meta = json.loads((DATA / "feature_cols.json").read_text())
    FEAT = meta["feature_cols"]
    params = get_params()

    test_raw = pl.read_csv(
        str(DATASET_TEST),
        schema_overrides={
            "airport_alert_id": pl.Float64,
            "is_last_lightning_cloud_ground": pl.Boolean,
        },
    ).with_columns(
        pl.col("date").str.to_datetime(time_unit="us", time_zone="UTC")
    )

    test_feats = cf.compute_features(test_raw)
    test_feats = cf.add_terrain_features(test_feats)

    ap_dfs = []
    for ap in ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]:
        subset = test_feats.filter(pl.col("airport") == ap)
        subset = cf.add_weather_features(subset, ap)
        ap_dfs.append(subset)

    all_cols: set[str] = set()
    for df in ap_dfs:
        all_cols.update(df.columns)
    aligned = []
    for df in ap_dfs:
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None).cast(pl.Float32).alias(c) for c in missing])
        aligned.append(df)
    test_feats = pl.concat(aligned)

    df_alerts = test_feats.filter(pl.col("airport_alert_id").is_not_null())
    missing_feat_cols = [c for c in FEAT if c not in df_alerts.columns]
    if missing_feat_cols:
        df_alerts = df_alerts.with_columns(
            [pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing_feat_cols]
        )

    X = df_alerts.select(FEAT).fill_nan(0).fill_null(0).to_numpy()
    scores = predict_proba(X)

    scored = df_alerts.select(
        ["airport", "airport_alert_id", "date", "lat", "lon", "icloud", "dist", "amplitude"]
    ).with_columns(pl.Series("score", scores))

    scored.write_parquet(str(SCORED_CACHE))
    print(f"[data_loader] Cache sauvegardé → {SCORED_CACHE} ({len(scored):,} éclairs)")
    return scored


# ── Construction du cache mémoire ────────────────────────────────────────────

def _build_memory_cache(scored: pl.DataFrame) -> None:
    params = get_params()
    K: int = params["k"]
    THR: float = params["base_threshold"]

    df = scored.sort(["airport", "airport_alert_id", "date"]).to_pandas()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    for airport_name, airport_df in df.groupby("airport"):
        airport_lower = airport_name.lower()
        alerts: list[dict] = []

        for alert_id, grp in airport_df.groupby("airport_alert_id"):
            grp = grp.sort_values("date").reset_index(drop=True)
            scores_arr = grp["score"].to_numpy()

            flashes: list[dict] = []
            prediction: dict | None = None

            for i, row in enumerate(grp.itertuples(index=False)):
                triggered = False
                if prediction is None and i >= K - 1:
                    window = scores_arr[i - K + 1 : i + 1]
                    if (window >= THR).all():
                        triggered = True
                        prediction = {
                            "triggered_at_rank": i + 1,
                            "triggered_at_date": row.date.isoformat(),
                            "confidence": float(window.min()),
                        }

                flashes.append({
                    "rank": i + 1,
                    "date": row.date.isoformat(),
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                    "flash_type": "IC" if int(row.icloud) == 1 else "CG",
                    "dist_km": float(row.dist),
                    "amplitude": float(row.amplitude),
                    "score": float(row.score),
                    "prediction_triggered": triggered,
                })

            alerts.append({
                "alert_id": str(int(alert_id)),
                "airport": airport_name,
                "start_date": grp["date"].iloc[0].isoformat(),
                "end_date": grp["date"].iloc[-1].isoformat(),
                "n_flashes": len(grp),
                "duration_s": (grp["date"].iloc[-1] - grp["date"].iloc[0]).total_seconds(),
                "flashes": flashes,
                "prediction": prediction,
            })

        alerts.sort(key=lambda a: a["start_date"], reverse=True)
        _alerts_cache[airport_lower] = alerts

    total = sum(len(v) for v in _alerts_cache.values())
    print(f"[data_loader] Cache mémoire prêt : {total} alertes / {len(_alerts_cache)} aéroports")


# ── Point d'entrée au démarrage ───────────────────────────────────────────────

async def initialize() -> None:
    cf = _import_cf()

    if SCORED_CACHE.exists():
        print("[data_loader] Chargement du cache depuis le disque...")
        scored = pl.read_parquet(str(SCORED_CACHE))
    else:
        scored = _build_scored_parquet(cf)

    _build_memory_cache(scored)


# ── Accesseurs ────────────────────────────────────────────────────────────────

def get_airports_list() -> list[dict]:
    return [
        {"id": k, "name": v["name"], "lat": v["lat"], "lon": v["lon"]}
        for k, v in AIRPORTS.items()
    ]


def get_alerts(airport: str) -> list[dict]:
    """Retourne la liste des alertes sans les flashes (pour le listing)."""
    return [
        {k: v for k, v in alert.items() if k != "flashes"}
        for alert in _alerts_cache.get(airport.lower(), [])
    ]


def get_alert(airport: str, alert_id: str) -> dict | None:
    """Retourne une alerte spécifique avec tous ses éclairs."""
    return next(
        (a for a in _alerts_cache.get(airport.lower(), []) if a["alert_id"] == alert_id),
        None,
    )


def get_default_alert(airport: str) -> dict | None:
    """Retourne une alerte aléatoire parmi celles avec ≥ 25 éclairs et prédiction déclenchée."""
    import random
    alerts = _alerts_cache.get(airport.lower(), [])
    if not alerts:
        return None
    candidates = [a for a in alerts if a["prediction"] is not None and a["n_flashes"] >= 25]
    if candidates:
        return random.choice(candidates)
    fallback = [a for a in alerts if a["n_flashes"] >= 25]
    if fallback:
        return random.choice(fallback)
    return random.choice(alerts)


def get_stats(airport: str) -> dict:
    from backend.config import MAX_GAP_MIN, MIN_DIST_KM

    alerts = _alerts_cache.get(airport.lower(), [])
    covered = [a for a in alerts if a["prediction"] is not None]

    total_gain_s = 0.0
    total_dangerous = 0
    total_missed_dangerous = 0

    for alert in covered:
        pred_date_iso = alert["prediction"]["triggered_at_date"]
        last_flash_date_iso = alert["end_date"]

        pred_ts = datetime.fromisoformat(pred_date_iso).timestamp()
        last_ts = datetime.fromisoformat(last_flash_date_iso).timestamp()

        baseline_end_ts = last_ts + MAX_GAP_MIN * 60
        gain_s = baseline_end_ts - pred_ts
        total_gain_s += gain_s

        for f in alert["flashes"]:
            if f["dist_km"] < MIN_DIST_KM:
                total_dangerous += 1
                if datetime.fromisoformat(f["date"]).timestamp() > pred_ts:
                    total_missed_dangerous += 1

    risk = total_missed_dangerous / max(total_dangerous, 1)

    return {
        "airport": airport.lower(),
        "total_alerts": len(alerts),
        "covered_alerts": len(covered),
        "coverage_rate": round(len(covered) / max(len(alerts), 1), 4),
        "total_gain_h": round(total_gain_s / 3600, 2),
        "risk": round(risk, 4),
    }
