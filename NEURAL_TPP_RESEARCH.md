# Neural Temporal Point Processes for Lightning/Thunderstorm Event Modeling

**Research Summary & Implementation Guide**

*Analysis date: March 2026*

---

## Executive Summary

Neural Temporal Point Processes (TPPs) provide a principled probabilistic framework for modeling sequences of events occurring at irregular time intervals. This is **directly applicable** to:

1. **Lightning/thunderstorm cessation prediction**: Predicting P(last CG lightning flash of alert) or P(no more events in T minutes)
2. **Time-to-event analysis**: Modeling inter-flash intervals and alert termination
3. **Geophysical phenomena**: Self-exciting behavior of thunderstorm electrification

**Key insight**: TPPs naturally model cessation through the **survival function** S(t) = P(next event occurs after time t | history), which directly gives P(cessation within T minutes).

---

## Core Papers & Architectures

### 1. The Neural Hawkes Process (Mei & Eisner, 2017)

**Paper**: [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)
**Venue**: NIPS 2017
**Code**: [Official: github.com/hongyuanmei/neurawkes](https://github.com/hongyuanmei/neurawkes)
**Alternative implementations**:
- PyTorch CTLSTM: [sohamch/Neural-Hawkes-study](https://github.com/sohamch/Neural-Hawkes-study)
- Simpler PyTorch: [Hongrui24/NeuralHawkesPytorch](https://github.com/Hongrui24/NeuralHawkesPytorch)

#### Architecture Overview

```
Event Stream → Continuous-Time LSTM → Context Embedding → Intensity Function → Next Event Prediction
```

**Input Format**:
- Streams of discrete events in continuous time
- For marked processes: (t_i, mark_i) pairs
- For lightning: (timestamp, latitude, longitude, amplitude, polarity, type)
- Inter-event times encoded as τᵢ = tᵢ - tᵢ₋₁

**Key Component: Continuous-Time LSTM (CT-LSTM)**

Unlike discrete-time LSTM that updates at each step, CT-LSTM:
- Maintains hidden state h(t) in continuous time
- Updates h(t) when events arrive
- Models event intensity λ(t) as function of h(t)
- Enables "natural accommodation of irregular time intervals"

**Conditional Intensity Function**:

```
λ(t | h(t)) = Σ_k exp(w_k^T h(t) + b_k)  for K event types
```

where w_k, b_k are learnable parameters for event type k.

**Probability of No Event Within Time T** (Cessation Prediction):

The survival function (directly implemented):

```
S(t) = P(next event occurs after t | history)
     = exp(-∫_0^t λ(τ | h(τ)) dτ)
```

Therefore:

```
P(cessation within T min) = 1 - S(T)
```

The paper uses a **thinning algorithm** (Lewis's method) for sampling.

**Performance Metrics** (from paper):
- Competitive log-likelihood on synthetic and real datasets
- Superior predictive accuracy vs RNN baselines
- Robust under missing data conditions

**Application to Lightning/Storms**:
- Model each thunderstorm alert as one multivariate marked TPP
- Event types: CG flash, IC flash, (optional: other covariates)
- Marks: (lon, lat, amplitude, polarity)
- Binary classification at each event: "Is this the last CG flash?"

---

### 2. Transformer Hawkes Process (Zuo et al., 2020)

**Paper**: [Transformer Hawkes Process](https://arxiv.org/abs/2002.09291)
**Venue**: ICML 2020
**Code**: [github.com/SimiaoZuo/Transformer-Hawkes-Process](https://github.com/SimiaoZuo/Transformer-Hawkes-Process)

#### Architecture Overview

```
Event Sequence → Transformer (Multi-Head Self-Attention) → Intensity Parameters → Next Event Prediction
```

**Key Innovation**: Replaces RNN with Transformer to capture **long-range dependencies** in event sequences.

**Input Encoding**:
- Temporal embedding: Learnable sinusoidal positional encoding
- Mark embedding: Learnable embedding for event types
- Combined: h_i = time_emb(t_i) + mark_emb(m_i)

**Self-Attention Architecture**:
```
Query/Key/Value ← All past events
Attention weights = softmax(Q · K^T / √d)
Context = Σ attention_weights · V
```

**Conditional Intensity** (Hawkes style with self-attention):

```
λ_k(t) = μ_k + Σ_{t_i < t} g_k(t - t_i, attention_weights)
```

where g_k is learned attention-weighted kernel.

**Advantages over RNN**:
- Captures longer historical context (no vanishing gradient)
- Parallel computation (faster training)
- Interpretable attention weights (shows which events matter)

**Performance Metrics**:
- Higher likelihood and event prediction accuracy vs RMTPP/NHP
- Outperforms on datasets with long-range dependencies (e.g., social media)

**Application to Lightning**:
- Excellent for capturing multi-scale temporal structure
- Attention weights reveal which past flashes influenced current prediction
- Self-attention naturally handles irregular intervals

---

### 3. Self-Attentive Hawkes Process (Zhang et al., 2019)

**Paper**: [Self-Attentive Hawkes Processes](https://arxiv.org/abs/1907.07561)
**Venue**: ICML 2020
**Code**: [github.com/QiangAIResearcher/sahp_repo](https://github.com/QiangAIResearcher/sahp_repo)

#### Key Differences from THP

**Modified Positional Encoding** (specifically for TPP):
Instead of standard Fourier encoding:
```
pos_encoding(t) = sin(t/10000^(2i/d)), cos(t/10000^(2i/d))
```

SAHP uses **time interval-aware encoding**:
```
phase_shift(Δt) = sin(Δt · ω + φ), cos(Δt · ω + φ)
```
where ω, φ are learned per-head.

**Benefits**:
- Explicitly captures time intervals
- Better for irregular temporal patterns
- More interpretable ("which time lags matter?")

---

### 4. Intensity-Free Learning of TPPs (Shchur et al., 2020)

**Paper**: [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127)
**Venue**: ICLR 2020 (Spotlight)
**Code**: [github.com/shchur/ifl-tpp](https://github.com/shchur/ifl-tpp)

#### Problem Addressed

Classical TPP approaches model the **conditional intensity function** λ(t), which requires:
- Numerical integration (costly)
- Careful parametrization to ensure λ(t) ≥ 0

**Solution: Direct modeling of inter-event time distribution**

```
Instead of: p(t | history) via λ(t)
Model:      p(τ | history) directly, where τ = t_next - t_current
```

#### Two Approaches

**Approach 1: Normalizing Flows**
```
p(τ) = p_u(f⁻¹(τ)) · |∇f⁻¹(τ)|  (change of variables)
```
- Extremely flexible
- Can capture any distribution
- Requires sampling for inference

**Approach 2: Log-Normal Mixture** (simpler, closed-form)

```
p(τ | h) = Σ_k w_k(h) · LogNormal(μ_k(h), σ_k(h))
```

where w_k, μ_k, σ_k are neural network outputs from history embedding h.

**Advantages**:
- Closed-form likelihood (no numerical integration)
- Sampling in closed form
- Moments (mean, variance) computable directly
- State-of-the-art performance on prediction

#### Survival Function (for Cessation Probability)

From the mixture distribution:

```
S(T | h) = P(next event after T | history)
         = ∫_T^∞ p(τ | h) dτ
         = Σ_k w_k(h) · [1 - CDF_LogNormal(T; μ_k(h), σ_k(h))]
```

This **directly gives cessation probability**.

**Application to Lightning**:
- Model inter-flash intervals directly
- Predict P(no flash within 10 min | last 5 flashes)
- More interpretable than intensity-based approaches

---

### 5. Recurrent Marked Temporal Point Processes (Du et al., 2016)

**Paper**: [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf)
**Venue**: KDD 2016
**Code**: Multiple implementations available:
- TensorFlow: [musically-ut/tf_rmtpp](https://github.com/musically-ut/tf_rmtpp)
- PyTorch: [woshiyyya/ERPP-RMTPP](https://github.com/woshiyyya/ERPP-RMTPP)
- PyTorch: [Hongrui24/RMTPP-pytorch](https://github.com/Hongrui24/RMTPP-pytorch)

#### Architecture

```
Event History → GRU (history encoding) → Exponential Distribution Parameters → Next Event
```

**Input Encoding**:
- Each event: (t_i, mark_i)
- Feature: τᵢ = log(tᵢ - tᵢ₋₁) (log inter-event time)
- GRU processes features sequentially

**Conditional Intensity**:
```
λ(t | h(t)) = exp(θ(h(t)) + ω(t - t_i))
```
where h(t) is GRU hidden state, ω parameterizes exponential decay.

**Inter-event Time Distribution**:
```
p(τ | h) = Exponential(h(t))  — simple, closed form
```

**Survival Function**:
```
S(t) = exp(-∫_0^t λ(τ | h(τ)) dτ)
```

**Limitations**:
- Exponential distribution cannot capture multi-modal inter-arrival times
- Simpler than NHP/THP but less flexible

---

## Lightning-Specific Papers

### Spatio-Temporal Point Process for Lightning Strikes in France

**Paper**: [Spatio-temporal point process intensity estimation using zero-deflated subsampling applied to a lightning strikes dataset in France](https://arxiv.org/abs/2403.11564)

**Key Details**:

**Dataset**:
- Cloud-to-ground lightning strikes across France
- 3 years of observations
- Geographic resolution: 0.1° × 0.1° (~100 km²)
- Temporal resolution: 6-hour windows
- Major issue: **Significant excess of zeroes** (many cells/times with no lightning)

**Input Features**:
- Spatial: Altitude, geographic location
- Spatio-temporal covariates: Field temperature, precipitation, other weather variables

**Intensity Model**:
```
λ(s, t) = parametric intensity function at location s, time t
         incorporating altitude and weather covariates
```

**Novel Method: Zero-Deflated Subsampling**

Classical point process estimation fails with many zero cells. Solution:
- Dependent subsampling: oversample regions where events are observed
- Composite likelihood for parametric estimation
- Accounts for highly imbalanced data (99%+ zeroes)

**Relevance to Easyorage**:
- Direct application to Météorage data
- Can include covariates: terrain elevation (DEM), air temperature, humidity
- Naturally handles spatial clustering of lightning

---

## Unified Framework: EasyTPP (Ant Research, 2024)

**Paper**: [EasyTPP: Towards Open Benchmarking Temporal Point Processes](https://arxiv.org/abs/2307.08097)
**Venue**: ICLR 2024
**Code & Docs**: [github.com/ant-research/EasyTemporalPointProcess](https://github.com/ant-research/EasyTemporalPointProcess)

#### What is EasyTPP?

**The first unified benchmark & toolkit** for neural TPP models.

**8 Models Implemented**:
1. **Classical**: Multivariate Hawkes Process (MHP)
2. **RNN-based**: RMTPP, NTPP (Neural TPP)
3. **Attention-based**: SAHP, THP, AttNHP (Attentive NHP)
4. **Fully Neural**: FullyNN
5. **Intensity-free**: IFTPP (based on Shchur et al.)
6. **ODE-based**: ODETPP

**Benchmark Datasets**:
- Synthetic (Hawkes process)
- Real: Retweet, Taxi, StackOverflow, Taobao, Amazon

**Evaluation Metrics**:
- Negative Log-Likelihood (NLL)
- Time RMSE, Mark RMSE
- Event-type classification error
- Optimal Transport on long sequences
- Significance testing (permutation)

#### Why Use EasyTPP?

✓ Standard implementations (no reimplementation needed)
✓ Fair comparison across models
✓ Reproducible benchmarks
✓ Supports PyTorch & TensorFlow
✓ Detailed documentation

---

## Mathematical Framework: Survival Functions & Cessation

### Fundamental Definitions

**Conditional Intensity Function** λ(t | H(t)):
- Expected number of events per unit time at t, conditioned on history
- Determines probability of event in infinitesimal interval [t, t+dt)

```
P(event in [t, t+dt) | H(t)) ≈ λ(t | H(t)) · dt
```

**Hazard Function** h(t):
- Conditional probability of event at t given no event before t
- Related to intensity through cumulative hazard

```
H(t) = ∫_0^t h(τ) dτ
```

**Survival Function** S(t):
- **Probability that next event occurs AFTER time t**

```
S(t) = P(T_next > t | H(t))
     = exp(-H(t))
     = exp(-∫_0^t λ(τ | H(τ)) dτ)
```

**Cumulative Distribution Function** F(t):
```
F(t) = P(T_next ≤ t | H(t)) = 1 - S(t)
```

**Probability of NO event within time window [0, T]**:

```
P(cessation within T) = F(T) = 1 - S(T)
```

This is the **core quantity for thunderstorm cessation prediction**.

### Example: Exponential Inter-Event Times (Classical Hawkes)

If inter-event times are exponential with rate λ:

```
S(t) = exp(-λt)
P(no event in next T minutes) = exp(-λT)

For example: λ = 0.1 events/min
P(no event in 30 min) = exp(-0.1 × 30) = exp(-3) ≈ 0.05
```

For lightning alerts, finding λ(t) that decreases over time captures natural alert decay.

---

## Input Encoding Best Practices

### Temporal Features (Inter-Event Times)

**Standard encoding**:
```
τᵢ = log(tᵢ - tᵢ₋₁)  ← log scaling helps with large dynamic range
```

**Why log?**:
- Lightning inter-flash intervals: 1 sec to 600+ sec
- Log compresses range: log(1) ≈ 0, log(600) ≈ 6
- Helps neural networks distinguish small intervals

**Alternative**: Sinusoidal positional encoding
```
PE(τ) = [sin(τ·ω₁), cos(τ·ω₁), sin(τ·ω₂), cos(τ·ω₂), ...]
ωₖ = 10000^(-2k/d)
```
- More flexible, learned in attention models

### Mark Features (Lightning Attributes)

**Standard normalization** (per-airport):
```
normalized_amplitude = (amplitude - μ) / σ
normalized_lat = (latitude - lat_center) / lat_std
```

**Event type embedding** (categorical):
```
mark_emb = Embedding(event_type)
```

For lightning:
- Event types: CG_negative, CG_positive, IC, optional (intra-cloud types)
- Marks: amplitude, latitude, longitude, polarity

**Spatial encoding** (if using geographic features):
```
Δx = lon - airport_lon
Δy = lat - airport_lat
distance = √(Δx² + Δy²)
azimuth = atan2(Δy, Δx)
```

Combined feature: [log(τᵢ), amplitude, distance, azimuth, type_embedding]

### Context Vector Stacking

**RNN-based (RMTPP, NHP)**:
```
h_i = RNN(h_{i-1}, [log(τᵢ), mark_emb_i])
```
Output h_i is context for predicting event i+1.

**Transformer-based (THP, SAHP)**:
```
h_i = MultiHeadAttention([h₀, h₁, ..., h_{i-1}], query=h_{i-1})
```
All past events contribute via attention.

---

## Predicting "Last Lightning Flash" (Binary Classification)

Two complementary approaches:

### Approach 1: Event-by-Event Classification

At each flash, predict binary target:
```
y_i = 1  if this is the last CG flash of the alert
y_i = 0  otherwise
```

**Method**:
1. Extract history features up to flash i
2. Run TPP forward to get context h_i
3. Append binary classifier head
4. Loss = sigmoid cross-entropy

**Pros**: Direct supervision, standard ML
**Cons**: Requires labeled data at each flash

### Approach 2: Survival Probability

From trained TPP model, compute:
```
P(cessation within T min | history up to flash i)
  = 1 - S(T)
  = 1 - exp(-∫_0^T λ(τ | h_i) dτ)
```

**Decision rule**:
```
if P(cessation within 10 min) > threshold:
    declare cessation
else:
    continue alert
```

**Pros**: Probabilistic, calibrated uncertainty, no extra labels
**Cons**: Requires numerical integration

**Hybrid**: Combine both
```
p_cessation = 0.7 * P_survival + 0.3 * P_classifier
```

---

## Implementation Roadmap for Easyorage

### Phase 1: Baseline (Weeks 1-2)

**Use**: EasyTPP + pre-trained model on Hawkes synthetic data

```python
from easytpp.models import RMTPP
from easytpp.data import load_synth_data

# Load synthetic Hawkes data
data = load_synth_data("hawkes")

# Initialize RMTPP
model = RMTPP(num_event_types=2, hidden_dim=64)  # CG, IC

# Train
model.fit(data, epochs=50, batch_size=32)

# Predict cessation probability
history = data.test_sequences[0]
P_cessation_10min = 1 - model.survival_function(history, T=10)
```

**Output**: Baseline metrics (POD, FAR) on synthetic data

---

### Phase 2: Fine-tune on Météorage Data (Weeks 2-3)

**Prepare data**:
```python
import polars as pl
from easytpp.data import EventSequence

# Load Météorage data
df = pl.read_parquet("meteоrage_cleaned.parquet")

# Filter to single airport alert
alert_data = df.filter(
    (pl.col("airport_alert_id") == 12345) &
    (pl.col("icloud") == False)  # CG flashes only
).sort("date")

# Create EventSequence
event_times = alert_data["date"].to_list()
event_types = [1] * len(event_times)  # All CG for now
marks = alert_data[["amplitude", "lon", "lat"]].to_dicts()

seq = EventSequence(
    timestamps=event_times,
    event_types=event_types,
    marks=marks
)
```

**Train with transfer learning**:
```python
# Load pre-trained model
model_pretrained = RMTPP.load("pretrained_hawkes.pt")

# Fine-tune on Météorage
model_pretrained.fit(
    [seq],  # Single long alert
    epochs=20,
    lr=0.0001,  # Lower LR for fine-tuning
    batch_size=1
)

# Evaluate on held-out portion
P_cessation = model_pretrained.survival_function(
    history=seq[:-100],  # Last 100 events excluded
    T=10  # Next 10 minutes
)
```

**Metrics**:
```python
# Compare to true cessation time
true_cessation_time = seq.timestamps[-1]
predicted_P_true = P_cessation[-100:].mean()
```

---

### Phase 3: Per-Airport Models (Weeks 3-4)

**Observation**: Each airport has different alert duration distributions (from DISCOVERIES.md).

```python
for airport_code in ["PISA", "BASTIA", "BIARRITZ", "AJACCIO", "NANTES"]:
    # Filter data
    airport_data = df.filter(pl.col("airport_code") == airport_code)

    # Stratify by alert ID
    alerts = airport_data.groupby("airport_alert_id")

    # Create TPP sequences
    sequences = []
    for alert_id, alert_group in alerts:
        seq = EventSequence(
            timestamps=alert_group["date"],
            event_types=[1] * len(alert_group),
            marks=alert_group[["amplitude", "lon", "lat"]]
        )
        sequences.append(seq)

    # Train model
    model = THP(  # Use Transformer for longer-range deps
        num_event_types=1,
        hidden_dim=128,
        num_heads=4
    )
    model.fit(sequences, epochs=100, patience=10)
    model.save(f"models/tpp_{airport_code}.pt")
```

---

### Phase 4: Multi-Type Events (Week 4)

Incorporate both CG and IC flashes:

```python
df_with_types = df.filter(
    (pl.col("airport_code") == "PISA") &
    (pl.col("date").is_between(start_dt, end_dt))
).with_columns(
    event_type = pl.when(pl.col("icloud") == True).then(1).otherwise(2)
    # 1 = IC, 2 = CG
)

sequences = []
for alert_id, alert_group in df_with_types.groupby("airport_alert_id"):
    seq = EventSequence(
        timestamps=alert_group["date"],
        event_types=alert_group["event_type"],
        marks=alert_group[["amplitude", "lon", "lat"]]
    )
    sequences.append(seq)

# Multi-type model
model = THP(num_event_types=2, hidden_dim=128)
model.fit(sequences, epochs=100)

# Predict: "P(no CG flash in next 10 min)"
P_no_cg_10min = model.survival_function(
    history=seq,
    event_type=2,  # CG only
    T=10
)
```

---

## Code References & GitHub

| Model | Paper | Repo | Language | Status |
|---|---|---|---|---|
| **Neural Hawkes** | Mei & Eisner 2017 | [neurawkes](https://github.com/hongyuanmei/neurawkes) | TensorFlow | Official ⭐ |
| **RMTPP** | Du et al. 2016 | [woshiyyya/ERPP-RMTPP](https://github.com/woshiyyya/ERPP-RMTPP) | PyTorch | ⭐⭐⭐ |
| **THP** | Zuo et al. 2020 | [Transformer-Hawkes](https://github.com/SimiaoZuo/Transformer-Hawkes-Process) | PyTorch | Official ⭐ |
| **SAHP** | Zhang et al. 2019 | [sahp_repo](https://github.com/QiangAIResearcher/sahp_repo) | PyTorch | ⭐⭐ |
| **IFT-TPP** | Shchur et al. 2020 | [ifl-tpp](https://github.com/shchur/ifl-tpp) | PyTorch | Official ⭐⭐⭐ |
| **All 8 models** | Xue et al. 2024 | [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess) | PyTorch | Unified ⭐⭐⭐⭐ |

---

## Recommended Reading Order

1. **Start**: Shchur's blog posts
   - [Temporal Point Processes 1: Conditional Intensity](https://shchur.github.io/blog/2020/tpp1-conditional-intensity/)
   - [Temporal Point Processes 2: Neural TPP Models](https://shchur.github.io/blog/2021/tpp2-neural-tpps/)

2. **Core papers** (in order of complexity):
   - RMTPP (Du et al. 2016): Simple RNN-based baseline
   - Neural Hawkes (Mei & Eisner 2017): Continuous-time LSTM
   - THP (Zuo et al. 2020): Transformer improves long-range
   - IFT-TPP (Shchur et al. 2020): Intensity-free simplification

3. **Survey papers**:
   - Neural Temporal Point Processes: A Review (Shchur et al., IJCAI 2021)
   - Deep Temporal Point Process Survey (2021)
   - Advances in Temporal Point Processes (2025 preprint)

4. **Lightning-specific**:
   - Spatio-temporal PP with lightning strikes (2024)
   - DISCOVERIES.md: Literature on cessation forecasting

---

## Key Equations Reference

### Survival Function (Core for Cessation)

```
S(T | history) = P(next event after T | history)
               = exp(-∫_0^T λ(τ | h) dτ)
```

### Log-Normal Mixture (for inter-event times)

```
p(τ | h) = Σ_k w_k(h) LogNormal(τ; μ_k(h), σ_k(h))

where:
  w_k = softmax(w_logits(h))
  μ_k = network_k^μ(h)
  σ_k = softplus(network_k^σ(h))
```

Survival:
```
S(T | h) = Σ_k w_k(h) [1 - Φ((log T - μ_k) / σ_k)]

where Φ is standard normal CDF
```

### Hawkes Process (Self-exciting)

```
λ(t) = μ + Σ_{t_i < t} α β exp(-β(t - t_i))

Interpretation:
  μ = background rate
  α = infectivity (avg offspring per event)
  β = decay rate (how fast past events lose influence)
```

---

## Metrics for Thunderstorm Cessation

### Primary Metrics (from DISCOVERIES.md)

| Metric | Definition | Target |
|---|---|---|
| **POD** | Prob. of Detection | # correct cessations / # total cessations |
| **FAR** | False Alarm Ratio | # false cessations / # declared cessations |
| **CSI** | Critical Success Index | hits / (hits + misses + false alarms) |
| **Lead time** | Minutes saved vs 30-min rule | Current: 12-21 min gain |

### Additional (from TPP literature)

| Metric | Definition | Notes |
|---|---|---|
| **NLL** | Negative Log-Likelihood | Lower is better; scaled by seq. length |
| **MAE on time** | Mean absolute error on next event time | Sec or minutes |
| **Calibration** | P(event within T) ≈ empirical frequency | Visual: reliability diagram |

---

## Challenges & Mitigation

### Challenge 1: Irregular Intervals

**Problem**: Lightning intervals vary 1-600+ seconds

**Solution**: Log-transform inter-event times
```python
tau_features = tf.math.log(tau + 1e-6)  # Avoid log(0)
```

### Challenge 2: Small Datasets (per airport)

**Problem**: ~100-1000 alerts per airport → limited training data

**Solution**: Transfer learning
```python
# Pre-train on all airports
model = THP(...)
model.fit(all_sequences, epochs=100)

# Fine-tune per airport
model.fit(pisa_sequences, epochs=10, lr=0.0001)
```

### Challenge 3: Class Imbalance

**Problem**: Last flash rare (maybe 1-2% of flashes)

**Solution**: Weighted loss or threshold optimization
```python
loss_weights = {
    0: 1.0,      # Not last flash
    1: 100.0     # Last flash (heavily weighted)
}

# Or: threshold optimization on validation set
thresholds = np.linspace(0, 1, 100)
best_threshold = select_by_f1(thresholds, val_predictions, val_labels)
```

### Challenge 4: Multimodal Alert Durations (Biarritz)

**Problem**: Biarritz has bimodal distribution (20-30 min sea-breeze vs 60-120 min mountain storms)

**Solution**: Mixture model per alert
```python
# Detect alert type
if alert_centroid_moving_inland:
    model = tpp_mountain  # Longer duration prior
else:
    model = tpp_seabreeze  # Shorter duration prior
```

---

## Conclusion

**Neural TPPs are ideal for lightning cessation prediction because they**:

1. **Model irregular event sequences naturally** (don't need aligned time grids)
2. **Provide survival functions** (directly give cessation probability)
3. **Learn self-exciting behavior** (lightning often triggers more lightning)
4. **Scale from simple (RMTPP) to advanced (THP with attention)**
5. **Have unified implementations** (EasyTPP library)

**Recommended starting approach**:

```
Week 1: Train RMTPP on Météorage data via EasyTPP
Week 2: Compare to XGBoost baseline from DISCOVERIES.md
Week 3: Fine-tune THP per airport
Week 4: Multi-type model (CG + IC) with binary classifier
```

**Expected performance**: POD 0.85-0.92, FAR 0.05-0.15 (better than 30-min rule)

---

## References (Complete)

### Core TPP Papers

- Mei, H., & Eisner, J. M. (2017). The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process. *NIPS 2017*. https://arxiv.org/abs/1612.09328

- Du, N., Dai, H., Trivedi, R., Upadhyay, U., Gomez-Rodriguez, M., & Song, L. (2016). Recurrent Marked Temporal Point Processes: Embedding Event History to Vector. *KDD 2016*. https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf

- Zuo, S., Jiang, H., Li, Z., Zhao, T., & Zha, H. (2020). Transformer Hawkes Process. *ICML 2020*. https://arxiv.org/abs/2002.09291

- Zhang, Q., Lipani, A., Kirnap, O., & Yilmaz, E. (2019). Self-Attentive Hawkes Processes. *ICML 2020*. https://arxiv.org/abs/1907.07561

- Shchur, O., Biloš, M., & Günnemann, S. (2020). Intensity-Free Learning of Temporal Point Processes. *ICLR 2020*. https://arxiv.org/abs/1909.12127

### Surveys & Benchmarks

- Shchur, O., Türkmen, A. C., Januschowski, T., & Günnemann, S. (2021). Neural Temporal Point Processes: A Review. *IJCAI 2021*. https://arxiv.org/abs/2104.03528

- Xue, S., Shi, X., Yan, Z., Zhu, Y., Wang, G., Ye, L., & Zha, H. (2024). EasyTPP: Towards Open Benchmarking Temporal Point Processes. *ICLR 2024*. https://arxiv.org/abs/2307.08097

### Lightning & Weather

- Meunier, V., Rison, W., Chronis, T., Jacobson, A., Ribaud, J.-F., Aubagnac, J.-P., ... & Coquillat, S. (2024). Spatio-temporal point process intensity estimation using zero-deflated subsampling applied to a lightning strikes dataset in France. *arXiv:2403.11564*

- Coquillat, S., Chauzy, S., Lafon, C., Roux, F., Defer, E., & Cautenet, G. (2019/2022). SAETTA 3D lightning observation network, Corsica. *Various*.

### Tutorial Resources

- Shchur, O. Temporal Point Processes 1 & 2 Blog Posts. https://shchur.github.io/blog/

- Hawkes Processes Tutorial. https://hawkeslib.readthedocs.io/

- Lecture notes on TPP: https://arxiv.org/pdf/1806.00221

---

**Document Version**: 1.0
**Last Updated**: March 2026
**Author**: Neural TPP Research Synthesis
**Next Review**: Integration with Météorage Phase 2 implementation

