import pathlib

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
DATASET_TEST = ROOT / "dataset_test" / "dataset_set.csv"
SCORED_CACHE = DATA / "test_scored.parquet"

AIRPORTS: dict[str, dict] = {
    "ajaccio":  {"name": "Ajaccio",  "lat": 41.9236, "lon": 8.8018},
    "bastia":   {"name": "Bastia",   "lat": 42.5527, "lon": 9.4837},
    "biarritz": {"name": "Biarritz", "lat": 43.4683, "lon": -1.5311},
    "nantes":   {"name": "Nantes",   "lat": 47.1531, "lon": -1.6108},
    "pise":     {"name": "Pise",     "lat": 43.6839, "lon": 10.3927},
}

MAX_GAP_MIN = 30
MIN_DIST_KM = 3
