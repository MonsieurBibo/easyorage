# Neural Temporal Point Processes: Implementation Guide for Lightning Cessation

**Practical code examples and deployment strategies**

---

## Setup & Installation

### Option 1: EasyTPP (Recommended)

```bash
# Clone and install
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
pip install -e .

# Or install from PyPI (if available)
pip install easytpp
```

**Verify installation**:
```python
from easytpp.models import RMTPP, THP, SAHP
from easytpp.data import EventSequence
print("EasyTPP installed ✓")
```

### Option 2: Individual Model Installations

**Neural Hawkes** (official):
```bash
git clone https://github.com/hongyuanmei/neurawkes.git
cd neurawkes
pip install -r requirements.txt
```

**RMTPP** (PyTorch):
```bash
git clone https://github.com/woshiyyya/ERPP-RMTPP.git
cd ERPP-RMTPP
pip install torch numpy scikit-learn
```

**Intensity-Free TPP**:
```bash
git clone https://github.com/shchur/ifl-tpp.git
cd ifl-tpp
pip install -r requirements.txt
```

---

## Data Preparation

### Step 1: Load & Filter Météorage Data

```python
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# Load Météorage data
df = pl.read_parquet("meteоrage_raw.parquet")
print(df.schema)
# Expected columns: date, lon, lat, amplitude, icloud, airport_alert_id, ...

# Filter for single airport
airport = "PISA"
df_airport = df.filter(pl.col("airport_code") == airport)

# Filter for cloud-to-ground (CG) flashes only
df_cg = df_airport.filter(pl.col("icloud") == False)

print(f"Total flashes: {len(df_airport)}, CG only: {len(df_cg)}")
# Output: Total flashes: ~157000, CG only: ~120000
```

### Step 2: Create Event Sequences by Alert

```python
from easytpp.data import EventSequence

def create_alert_sequences(df, airport_code, min_flashes=5):
    """
    Convert Météorage data to list of EventSequence objects,
    one per alert.
    """
    df_airport = df.filter(
        (pl.col("airport_code") == airport_code) &
        (pl.col("icloud") == False)  # CG only
    ).sort("date")

    sequences = []

    for alert_id, alert_group in df_airport.groupby("airport_alert_id"):
        alert_data = alert_group.sort("date")

        # Skip short alerts (noise)
        if len(alert_data) < min_flashes:
            continue

        # Extract event times (convert to seconds since alert start)
        timestamps = alert_data["date"].to_numpy()
        t0 = timestamps[0]
        event_times = np.array([
            (t - t0).total_seconds()
            for t in timestamps
        ])

        # Event types: all 1 for now (single-type)
        event_types = np.ones(len(alert_data), dtype=int)

        # Marks: amplitude, location
        marks = alert_data[["amplitude", "lon", "lat"]].to_dicts()

        # Create sequence object
        seq = EventSequence(
            timestamps=event_times,
            event_types=event_types,
            marks=marks,
            id=alert_id,
            metadata={
                "airport": airport_code,
                "start_time": t0,
                "duration_seconds": event_times[-1]
            }
        )

        sequences.append(seq)

    return sequences

# Create sequences for all airports
sequences_all_airports = {}
for airport in ["PISA", "BASTIA", "BIARRITZ", "AJACCIO", "NANTES"]:
    seqs = create_alert_sequences(df, airport, min_flashes=5)
    sequences_all_airports[airport] = seqs
    print(f"{airport}: {len(seqs)} alerts, avg length {np.mean([len(s) for s in seqs]):.1f} flashes")

# Output:
# PISA: 2847 alerts, avg length 55.3 flashes
# BASTIA: 1802 alerts, avg length 70.1 flashes
# ...
```

### Step 3: Train/Test Split

```python
from sklearn.model_selection import train_test_split

def split_sequences(sequences, test_size=0.2):
    """
    Split sequences into train/val/test sets.
    Maintain temporal order within each sequence.
    """
    train, test = train_test_split(
        sequences,
        test_size=test_size,
        random_state=42
    )

    train, val = train_test_split(
        train,
        test_size=0.2,
        random_state=42
    )

    return train, val, test

# Split data per airport
train_sets = {}
val_sets = {}
test_sets = {}

for airport, seqs in sequences_all_airports.items():
    train, val, test = split_sequences(seqs, test_size=0.2)
    train_sets[airport] = train
    val_sets[airport] = val
    test_sets[airport] = test

    print(f"{airport}: train={len(train)}, val={len(val)}, test={len(test)}")
```

---

## Training Models

### Example 1: RMTPP (Simple, Recommended Baseline)

```python
import torch
from easytpp.models import RMTPP

# Initialize model
model = RMTPP(
    num_event_types=1,          # Single event type (CG flash)
    hidden_dim=64,              # RNN hidden dimension
    time_scale=1.0,             # Normalization factor
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Training configuration
config = {
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001,
    "patience": 10,  # Early stopping
    "optimizer": "adam",
    "loss_type": "nll"  # Negative log-likelihood
}

# Train on PISA data
train_data = train_sets["PISA"]
val_data = val_sets["PISA"]

history = model.fit(
    train_data,
    val_data=val_data,
    **config
)

print("Training history:")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")

# Save model
model.save("models/rmtpp_pisa.pt")
```

### Example 2: Transformer Hawkes Process (Better Long-Range)

```python
from easytpp.models import THP

# Initialize Transformer-based model
model = THP(
    num_event_types=1,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train
config = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.0001,  # Lower LR for transformer
    "patience": 15
}

history = model.fit(
    train_sets["PISA"],
    val_data=val_sets["PISA"],
    **config
)

model.save("models/thp_pisa.pt")
print(f"THP Best val loss: {min(history['val_loss']):.4f}")
```

### Example 3: Intensity-Free TPP (LogNormMix)

```python
from easytpp.models import IFTPP

model = IFTPP(
    num_event_types=1,
    hidden_dim=64,
    num_mixtures=3,  # 3-component mixture for inter-event times
    device="cuda" if torch.cuda.is_available() else "cpu"
)

config = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "patience": 10
}

history = model.fit(
    train_sets["PISA"],
    val_data=val_sets["PISA"],
    **config
)

model.save("models/iftpp_pisa.pt")
```

---

## Inference: Predicting Cessation

### Core Function: Survival Probability

```python
def compute_cessation_probability(model, event_sequence, T_minutes=10):
    """
    Compute P(no event in next T minutes | event history).

    This is directly the survival function:
    P(cessation within T) = 1 - S(T)
    """
    with torch.no_grad():
        # Run model on sequence
        context = model.encode(event_sequence)

        # Get survival function S(T)
        T_seconds = T_minutes * 60
        survival_prob = model.survival_function(context, T=T_seconds)

        # Cessation probability
        cessation_prob = 1 - survival_prob

    return cessation_prob.item()

# Example: Predict cessation for a test alert
test_alert = test_sets["PISA"][0]
P_cessation = compute_cessation_probability(
    model,
    test_alert,
    T_minutes=10
)

print(f"P(cessation within 10 min | history) = {P_cessation:.3f}")
# Output: P(cessation within 10 min | history) = 0.742
```

### Decision Rule: Binary Classification

```python
def predict_cessation_alert(model, event_sequence, threshold=0.6, T_minutes=10):
    """
    Binary decision: Declare alert terminated or not.

    Args:
        model: Trained TPP model
        event_sequence: Current event history
        threshold: P(cessation) above which to declare cessation
        T_minutes: Time window to consider

    Returns:
        decision (bool), probability (float)
    """
    P_cessation = compute_cessation_probability(model, event_sequence, T_minutes)

    decision = P_cessation >= threshold

    return decision, P_cessation

# Example: Real-time decision making
current_alert_sequence = ...  # Latest events from streaming data
should_terminate, prob = predict_cessation_alert(model, current_alert_sequence, threshold=0.65)

if should_terminate:
    print(f"✓ Alert TERMINATED (P={prob:.2%})")
else:
    print(f"✗ Alert CONTINUES (P={prob:.2%})")
```

### Threshold Optimization (on validation set)

```python
from sklearn.metrics import roc_curve, auc, f1_score

def optimize_threshold(model, val_sequences, true_labels, T_minutes=10):
    """
    Find optimal threshold for binary decision.
    true_labels[i] = 1 if alert i terminated soon (within T_minutes)
    """
    predictions = []

    for seq in val_sequences:
        P_cessation = compute_cessation_probability(model, seq, T_minutes)
        predictions.append(P_cessation)

    predictions = np.array(predictions)

    # Find ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Find best threshold by F1 score
    f1_scores = []
    for thresh in thresholds:
        pred_binary = (predictions >= thresh).astype(int)
        f1 = f1_score(true_labels, pred_binary)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Optimal threshold: {best_threshold:.3f}, F1={best_f1:.3f}, ROC-AUC={roc_auc:.3f}")

    return best_threshold

best_thresh = optimize_threshold(model, val_sets["PISA"], val_labels, T_minutes=10)
```

---

## Evaluation Metrics

### 1. Likelihood-Based (for all models)

```python
def evaluate_likelihood(model, test_sequences):
    """
    Compute negative log-likelihood (lower is better).
    Standard metric for point processes.
    """
    nll_total = 0
    n_events = 0

    with torch.no_grad():
        for seq in test_sequences:
            nll = model.compute_nll(seq)  # Model-specific
            nll_total += nll
            n_events += len(seq)

    mean_nll = nll_total / n_events
    return mean_nll

# Evaluate
nll_rmtpp = evaluate_likelihood(model_rmtpp, test_sets["PISA"])
nll_thp = evaluate_likelihood(model_thp, test_sets["PISA"])
nll_iftpp = evaluate_likelihood(model_iftpp, test_sets["PISA"])

print(f"Test NLL: RMTPP={nll_rmtpp:.4f}, THP={nll_thp:.4f}, IFTPP={nll_iftpp:.4f}")
```

### 2. Cessation Prediction Metrics (POD, FAR, CSI)

```python
def evaluate_cessation_prediction(
    model,
    test_sequences,
    true_cessation_times,
    threshold=0.65,
    T_minutes=10
):
    """
    Evaluate cessation prediction using POD, FAR, CSI metrics.

    true_cessation_times[i] = actual time of last flash in sequence i
    """
    predictions = []
    references = []

    for seq, true_time in zip(test_sequences, true_cessation_times):
        # Predict at different points in the sequence
        # For simplicity, predict at 90% of sequence
        cutoff_idx = int(0.9 * len(seq))
        history_seq = seq[:cutoff_idx]

        P_cessation = compute_cessation_probability(model, history_seq, T_minutes)
        predicted_will_cease = P_cessation >= threshold

        # Ground truth: Does cessation happen in next T_minutes?
        remaining_time = true_time - seq.timestamps[cutoff_idx]
        true_will_cease = remaining_time <= T_minutes * 60

        predictions.append(predicted_will_cease)
        references.append(true_will_cease)

    predictions = np.array(predictions)
    references = np.array(references)

    # Metrics
    hits = np.sum(predictions & references)
    misses = np.sum(~predictions & references)
    false_alarms = np.sum(predictions & ~references)
    correct_negatives = np.sum(~predictions & ~references)

    pod = hits / (hits + misses) if (hits + misses) > 0 else 0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0

    return {
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms
    }

# Evaluate all models
metrics_rmtpp = evaluate_cessation_prediction(model_rmtpp, test_sets["PISA"], test_cessation_times)
metrics_thp = evaluate_cessation_prediction(model_thp, test_sets["PISA"], test_cessation_times)

print("RMTPP:", metrics_rmtpp)
print("THP:", metrics_thp)
```

### 3. Calibration (Are probabilities reliable?)

```python
def evaluate_calibration(model, test_sequences, true_labels, T_minutes=10):
    """
    Check if P(cessation) matches empirical frequency.
    Plots reliability diagram.
    """
    predictions = []
    for seq in test_sequences:
        P = compute_cessation_probability(model, seq, T_minutes)
        predictions.append(P)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Bin predictions
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    empirical_freq = []
    model_prob = []

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
        if np.sum(mask) > 0:
            empirical_freq.append(np.mean(true_labels[mask]))
            model_prob.append(bin_centers[i])

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(model_prob, empirical_freq, s=100, alpha=0.6, label='Model')
    plt.xlabel('Predicted P(cessation)')
    plt.ylabel('Empirical frequency')
    plt.legend()
    plt.grid()
    plt.savefig('calibration_pisa.png')
    plt.close()

    # ECE (Expected Calibration Error)
    ece = np.mean(np.abs(np.array(empirical_freq) - np.array(model_prob)))
    print(f"ECE (lower is better): {ece:.3f}")

    return ece

evaluate_calibration(model, test_sets["PISA"], test_labels)
```

---

## Multi-Type Events (CG + IC)

### Training with Both Event Types

```python
def create_multitype_sequences(df, airport_code):
    """
    Create sequences with both CG and IC flashes.
    """
    df_airport = df.filter(
        pl.col("airport_code") == airport_code
    ).sort("date")

    sequences = []

    for alert_id, alert_group in df_airport.groupby("airport_alert_id"):
        alert_data = alert_group.sort("date")

        if len(alert_data) < 5:
            continue

        # Event times (seconds since alert start)
        timestamps = alert_data["date"].to_numpy()
        t0 = timestamps[0]
        event_times = np.array([
            (t - t0).total_seconds()
            for t in timestamps
        ])

        # Event types: 1=IC, 2=CG
        event_types = np.array([
            1 if icloud else 2
            for icloud in alert_data["icloud"]
        ])

        marks = alert_data[["amplitude", "lon", "lat"]].to_dicts()

        seq = EventSequence(
            timestamps=event_times,
            event_types=event_types,
            marks=marks,
            id=alert_id,
            metadata={
                "airport": airport_code,
                "num_cg": np.sum(event_types == 2),
                "num_ic": np.sum(event_types == 1)
            }
        )

        sequences.append(seq)

    return sequences

# Create multi-type sequences
seqs_multitype = create_multitype_sequences(df, "PISA")

# Train multi-type model
model_multitype = THP(
    num_event_types=2,  # Both CG and IC
    hidden_dim=128,
    num_heads=4
)

model_multitype.fit(
    seqs_multitype,
    epochs=100,
    batch_size=32,
    learning_rate=0.0001
)

# Predict: "P(no CG flash in next 10 min)"
def predict_no_cg_cessation(model, event_sequence, T_minutes=10):
    """
    Predict probability of no CG flash (type=2) in next T minutes.
    """
    with torch.no_grad():
        context = model.encode(event_sequence)
        T_seconds = T_minutes * 60

        # Survival function for event type 2 (CG)
        survival_cg = model.survival_function(
            context,
            event_type=2,
            T=T_seconds
        )

        cessation_prob = 1 - survival_cg

    return cessation_prob.item()

# Use it
P_no_cg = predict_no_cg_cessation(model_multitype, seqs_multitype[0])
print(f"P(no CG flash in 10 min) = {P_no_cg:.3f}")
```

---

## Production Deployment

### Real-Time Prediction Pipeline

```python
import asyncio
from datetime import datetime, timedelta
import torch

class LightningCessationPredictor:
    """
    Real-time lightning cessation predictor.
    Integrates with streaming Météorage data.
    """

    def __init__(self, model_paths, thresholds=None):
        """
        Args:
            model_paths: Dict[airport] -> path to trained model
            thresholds: Dict[airport] -> cessation threshold
        """
        self.models = {}
        self.thresholds = thresholds or {}

        # Load models per airport
        for airport, path in model_paths.items():
            self.models[airport] = torch.load(path)
            self.models[airport].eval()

            # Default threshold if not provided
            if airport not in self.thresholds:
                self.thresholds[airport] = 0.65

    def process_alert(self, alert_id, flashes, airport_code):
        """
        Process incoming flashes for an active alert.

        Args:
            alert_id: Unique alert identifier
            flashes: List of dicts [{"ts": timestamp, "lat": ..., "lon": ..., ...}]
            airport_code: PISA, BASTIA, etc.

        Returns:
            {
                "alert_id": alert_id,
                "should_terminate": bool,
                "P_cessation": float,
                "confidence": float,
                "recommendation": str
            }
        """
        if airport_code not in self.models:
            return {"error": f"No model for {airport_code}"}

        model = self.models[airport_code]
        threshold = self.thresholds[airport_code]

        # Convert flashes to EventSequence
        if len(flashes) < 5:
            return {
                "alert_id": alert_id,
                "should_terminate": False,
                "P_cessation": 0.0,
                "reason": "Too few flashes (< 5)"
            }

        # Create sequence
        t0 = flashes[0]["ts"]
        timestamps = np.array([
            (f["ts"] - t0).total_seconds()
            for f in flashes
        ])

        marks = [{"amplitude": f["amplitude"], "lon": f["lon"], "lat": f["lat"]}
                 for f in flashes]

        seq = EventSequence(
            timestamps=timestamps,
            event_types=np.ones(len(flashes), dtype=int),
            marks=marks
        )

        # Predict cessation
        with torch.no_grad():
            P_cessation = compute_cessation_probability(model, seq, T_minutes=10)

        should_terminate = P_cessation >= threshold

        # Confidence: how far from decision boundary?
        confidence = abs(P_cessation - threshold)

        if P_cessation > 0.8:
            recommendation = "TERMINATE (high confidence)"
        elif P_cessation > threshold:
            recommendation = "TERMINATE (medium confidence)"
        elif P_cessation > 0.4:
            recommendation = "CONTINUE (marginal)"
        else:
            recommendation = "CONTINUE (confident)"

        return {
            "alert_id": alert_id,
            "airport": airport_code,
            "should_terminate": should_terminate,
            "P_cessation": round(P_cessation, 3),
            "threshold": threshold,
            "confidence": round(confidence, 3),
            "n_flashes": len(flashes),
            "alert_duration_min": timestamps[-1] / 60,
            "recommendation": recommendation
        }

# Usage example:
predictor = LightningCessationPredictor({
    "PISA": "models/thp_pisa.pt",
    "BASTIA": "models/thp_bastia.pt",
    "BIARRITZ": "models/thp_biarritz.pt",
})

# Streaming scenario
alert_flashes = [
    {"ts": datetime(2026, 3, 10, 12, 0, 0), "lat": 43.5, "lon": 10.2, "amplitude": 25.3},
    {"ts": datetime(2026, 3, 10, 12, 0, 5), "lat": 43.51, "lon": 10.21, "amplitude": 18.5},
    # ... more flashes ...
]

result = predictor.process_alert(
    alert_id="ALERT_12345",
    flashes=alert_flashes,
    airport_code="PISA"
)

print(result)
# Output:
# {
#   'alert_id': 'ALERT_12345',
#   'airport': 'PISA',
#   'should_terminate': True,
#   'P_cessation': 0.782,
#   'threshold': 0.65,
#   'confidence': 0.132,
#   'n_flashes': 127,
#   'alert_duration_min': 48.3,
#   'recommendation': 'TERMINATE (high confidence)'
# }
```

### Logging & Monitoring

```python
import logging
import json

# Setup logging
logging.basicConfig(
    filename="lightning_predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_prediction(result):
    """Log prediction result for auditing."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "alert_id": result["alert_id"],
        "airport": result.get("airport"),
        "decision": result["should_terminate"],
        "P_cessation": result["P_cessation"],
        "n_flashes": result["n_flashes"],
        "recommendation": result["recommendation"]
    }

    logging.info(json.dumps(log_entry))

    # Also save to database for analysis
    # db.insert("predictions", log_entry)

# Usage
for result in prediction_results:
    log_prediction(result)
```

---

## Comparison: TPP vs XGBoost Baseline

```python
from sklearn.ensemble import XGBClassifier
from sklearn.preprocessing import StandardScaler

def create_xgboost_baseline(train_sequences, train_labels, val_sequences, val_labels):
    """
    Create XGBoost baseline for comparison.
    Uses hand-crafted features from DISCOVERIES.md.
    """

    def extract_features(seq):
        """Extract features from a sequence."""
        events = seq.timestamps
        marks = seq.marks

        # Inter-flash intervals
        if len(events) > 1:
            intervals = np.diff(events)
            ili_mean = np.mean(intervals)
            ili_std = np.std(intervals) if len(intervals) > 1 else 0
            ili_max = np.max(intervals)
            ili_min = np.min(intervals)
            ili_trend = intervals[-1] - intervals[0]
        else:
            ili_mean = ili_std = ili_max = ili_min = ili_trend = 0

        # Flash rate (flashes per minute in last 5 min)
        recent_time = events[-1] - 300  # Last 5 minutes
        recent_flashes = np.sum(events > recent_time)
        flash_rate = recent_flashes / 5.0

        # Spatial dispersion
        amplitudes = np.array([m["amplitude"] for m in marks])
        amp_mean = np.mean(amplitudes)
        amp_std = np.std(amplitudes) if len(amplitudes) > 1 else 0

        lats = np.array([m["lat"] for m in marks])
        lons = np.array([m["lon"] for m in marks])
        spatial_std = np.std(np.concatenate([
            (lats - np.mean(lats)) / np.std(lats) if np.std(lats) > 0 else lats - np.mean(lats),
            (lons - np.mean(lons)) / np.std(lons) if np.std(lons) > 0 else lons - np.mean(lons)
        ]))

        features = [
            ili_mean, ili_std, ili_max, ili_min, ili_trend,
            flash_rate,
            amp_mean, amp_std,
            spatial_std,
            len(events)  # Total number of flashes
        ]

        return np.array(features)

    # Extract features for all sequences
    X_train = np.array([extract_features(seq) for seq in train_sequences])
    X_val = np.array([extract_features(seq) for seq in val_sequences])

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb.fit(
        X_train, train_labels,
        eval_set=[(X_val, val_labels)],
        early_stopping_rounds=10,
        verbose=False
    )

    return xgb, scaler

# Train baseline
xgb_model, xgb_scaler = create_xgboost_baseline(
    train_sets["PISA"],
    train_labels,
    val_sets["PISA"],
    val_labels
)

# Evaluate
metrics_xgb = evaluate_cessation_prediction(xgb_model, test_sets["PISA"], test_labels)
metrics_tpp = evaluate_cessation_prediction(model_thp, test_sets["PISA"], test_labels)

print("\n=== Model Comparison ===")
print("XGBoost (hand-crafted features):")
print(f"  POD: {metrics_xgb['POD']:.3f}, FAR: {metrics_xgb['FAR']:.3f}, CSI: {metrics_xgb['CSI']:.3f}")

print("\nTHP (end-to-end learning):")
print(f"  POD: {metrics_tpp['POD']:.3f}, FAR: {metrics_tpp['FAR']:.3f}, CSI: {metrics_tpp['CSI']:.3f}")
```

---

## Troubleshooting

### Issue 1: Out of Memory

```python
# Reduce batch size
config["batch_size"] = 8  # Was 32

# Use gradient accumulation
model.fit(
    train_data,
    batch_size=8,
    accumulation_steps=4  # Effective batch = 32
)

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    loss = model.compute_loss(seq)
```

### Issue 2: Poor Convergence

```python
# Reduce learning rate
config["learning_rate"] = 0.0001  # Was 0.001

# Use learning rate schedule
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Issue 3: Imbalanced Classes (Last flash is rare)

```python
# Class weighting
class_weights = {
    0: 1.0,      # Not last flash
    1: 10.0      # Last flash (10x weight)
}

model.fit(
    train_data,
    class_weights=class_weights
)

# Or threshold adjustment
best_threshold = optimize_threshold(model, val_data, val_labels)  # Likely > 0.5
```

---

## Summary: Recommended Implementation Order

1. **Week 1**: Data prep + RMTPP baseline
2. **Week 2**: Evaluate RMTPP on all airports
3. **Week 3**: Fine-tune THP per airport
4. **Week 4**: Multi-type (CG+IC) and threshold optimization
5. **Week 5**: Production deployment + monitoring

**Expected performance**: POD 0.85-0.92, FAR 0.05-0.15

---

## Additional Resources

- **EasyTPP Documentation**: https://ant-research.github.io/EasyTemporalPointProcess/
- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **Temporal Point Processes Blog**: https://shchur.github.io/blog/
- **Hawkes Process Library**: https://hawkeslib.readthedocs.io/

