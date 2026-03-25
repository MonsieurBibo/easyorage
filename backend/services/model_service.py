import numpy as np
import joblib
from backend.config import MODEL_DIR

_model = None
_params = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_DIR / "xgb_best.joblib")
    return _model


def get_params() -> dict:
    global _params
    if _params is None:
        _params = joblib.load(MODEL_DIR / "predict_params.joblib")
    return _params


def predict_proba(X: np.ndarray) -> np.ndarray:
    return get_model().predict_proba(X)[:, 1]
