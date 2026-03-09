# Neural Temporal Point Processes: Quick Reference Card

**For lightning cessation prediction • Easyorage DataBattle 2026**

---

## Paper Summary Table

| Paper | Authors | Year | Venue | Key Innovation | GitHub | Performance |
|---|---|---|---|---|---|---|
| **Neural Hawkes** | Mei & Eisner | 2017 | NIPS | Continuous-time LSTM | [neurawkes](https://github.com/hongyuanmei/neurawkes) | Competitive baseline |
| **RMTPP** | Du et al. | 2016 | KDD | RNN history encoding | [tf_rmtpp](https://github.com/woshiyyya/ERPP-RMTPP) | Simple, fast |
| **THP** | Zuo et al. | 2020 | ICML | Transformer + self-attention | [Transformer-Hawkes](https://github.com/SimiaoZuo/Transformer-Hawkes-Process) | Best long-range ⭐ |
| **SAHP** | Zhang et al. | 2019 | ICML | Time-aware attention | [sahp_repo](https://github.com/QiangAIResearcher/sahp_repo) | Interpretable |
| **IFT-TPP** | Shchur et al. | 2020 | ICLR | Log-normal mixture | [ifl-tpp](https://github.com/shchur/ifl-tpp) | Direct cessation ⭐ |
| **EasyTPP** | Xue et al. | 2024 | ICLR | Unified benchmark | [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess) | All 8 models ⭐⭐⭐ |

---

## Core Math Cheatsheet

### Probability of Cessation (Main Goal)

```
P(cessation within T min) = 1 - S(T)
                           = 1 - exp(-∫₀ᵀ λ(τ|h) dτ)
                           = F(T)  [CDF of next event time]
```

**In log-normal mixture** (simplest):

```
P(cessation) = Σₖ wₖ(h) [1 - Φ((ln T - μₖ)/σₖ)]
               ↑ mixture weight ↑  ↑ log-normal CDF ↑
```

### Conditional Intensity (Classical Hawkes)

```
λ(t) = μ + Σᵢ:tᵢ<t α β exp(-β(t - tᵢ))
       └─ background ──┬──┘ └─── self-exciting ───┘
```

### Hawkes Self-Excitement Parameter

```
α ∈ (0, 1): Controlled branching
α > 1: Exploding (unphysical for finite sequences)

For lightning: α ≈ 0.3-0.7 typical (each flash triggers 30-70% of next flashes)
```

---

## Architecture Comparison

### RMTPP (Baseline)

```
Input: [log(Δt), mark_embedding]
           ↓
        GRU Cell (RNN history)
           ↓
      Output Context h
           ↓
    λ(t) = exp(h + decay)
    τ ~ Exponential(λ)
```

**Pros**: Fast, simple, proven
**Cons**: Single mode (exponential), limited history

---

### THP (Recommended)

```
Input: All events (t₁, m₁), ..., (tₙ, mₙ)
           ↓
    Sinusoidal Encoding
           ↓
Multi-Head Self-Attention (4+ heads)
    ↑ Captures long-range ↓
    ← Parallel computation
           ↓
Intensity Parameters (Hawkes-style)
           ↓
P(next event at time T)
```

**Pros**: Long-range dependencies, interpretable attention, faster training
**Cons**: Slightly more complex to implement

---

### IFT-TPP (For Direct Cessation)

```
Input: Event history h
           ↓
   GRU Encoder → h_context
           ↓
  Output: w₁, μ₁, σ₁, ..., wₖ, μₖ, σₖ
          └─ Mixture parameters ─┘
           ↓
τ ~ Σₖ wₖ LogNormal(μₖ, σₖ)
           ↓
Survival: S(T) = Σₖ wₖ [1 - Φ((ln T - μₖ)/σₖ)]
           ↓
P(cessation) = 1 - S(T) ← **Direct target**
```

**Pros**: Direct cessation prediction, closed-form moments
**Cons**: Slightly less flexible than flows

---

## Feature Engineering

### Minimal Features (if using end-to-end)

```python
features = [
    np.log(inter_flash_interval + 1e-6),
    amplitude_kA,
    distance_from_airport_km,
    azimuth_deg
]
```

### Hand-Crafted Features (for XGBoost comparison)

```python
features = [
    # Temporal
    np.mean(intervals),           # ILI mean
    np.std(intervals),            # ILI std
    np.max(intervals),            # ILI max
    intervals[-1] - intervals[0], # ILI trend (increasing?)

    # Rate
    recent_flash_count / 5,       # Flash rate (flashes/min) last 5 min

    # Spatial
    np.std(amplitudes),
    spatial_dispersion_km,

    # Meta
    total_flash_count,
    duration_min
]
```

**Best single feature**: Inter-flash interval (ILI) → mean/std/trend
**Reason**: Directly captures "has alert stopped accelerating?"

---

## Implementation Checklist

### Phase 1: Baseline (Week 1)

- [ ] Load Météorage data
- [ ] Convert to `EventSequence` objects (per-alert)
- [ ] Train RMTPP on synthetic Hawkes data (EasyTPP)
- [ ] Evaluate: NLL on test set
- [ ] Baseline metrics: POD/FAR on synthetic

### Phase 2: Fine-tune (Week 2)

- [ ] Train THP on all Météorage alerts mixed
- [ ] Compute cessation probability P(T=10 min)
- [ ] Optimize threshold on validation set
- [ ] Metrics: POD/FAR/CSI on holdout test
- [ ] Compare to XGBoost baseline

### Phase 3: Per-Airport (Week 3)

- [ ] Build separate model for each airport
- [ ] Account for duration distribution differences (Biarritz bimodal!)
- [ ] Per-airport thresholds

### Phase 4: Production (Week 4)

- [ ] Real-time pipeline (streaming flashes)
- [ ] Decision rule: P(cessation) > threshold → terminate
- [ ] Logging & monitoring
- [ ] Uncertainty quantification

---

## Key Hyperparameters

### RMTPP / THP

```python
config = {
    "hidden_dim": 64,           # 64-128, larger = more capacity
    "num_heads": 4,             # For THP only; 4-8 typical
    "num_layers": 2,            # 1-3
    "dropout": 0.1,             # 0.0-0.3
    "epochs": 100,
    "batch_size": 32,           # 16-64
    "learning_rate": 0.001,     # 0.0001-0.01
    "patience": 10,             # Early stopping
}
```

### Log-Normal Mixture (IFT-TPP)

```python
config = {
    "num_mixtures": 3,          # Components; 2-5 typical
    "hidden_dim": 64,
    "epochs": 100,
    "learning_rate": 0.001,
}
```

---

## Evaluation Metrics

### Likelihood-Based (Reported in papers)

```
NLL = -log p(data)  ← Lower is better
      Averaged over all events in all sequences
```

### Cessation-Specific (What we care about)

```
POD   = TP / (TP + FN)           [Recall, Sensitivity]
FAR   = FP / (TP + FP)           [False Alarm Ratio]
CSI   = TP / (TP + FP + FN)      [Critical Success Index]

Typical targets:
  POD ≥ 0.85  (catch 85%+ of cessations)
  FAR ≤ 0.15  (avoid premature termination)
```

### Advanced

```
Calibration: Are probabilities correct?
  → Plot P(predicted) vs P(empirical) → should be diagonal

Lead time: Minutes saved vs 30-min rule
  → If declare cessation at T, actual cessation at T+10
  → Save 30-10 = 20 minutes ← Goal
```

---

## Common Pitfalls & Fixes

| Problem | Symptom | Fix |
|---|---|---|
| **Vanishing gradient** | Loss plateaus early | Reduce LR, add gradient clipping |
| **Out of memory** | GPU crash | Reduce batch size, use smaller hidden_dim |
| **Overfitting** | Train loss low, val loss high | Add dropout (0.2-0.3), early stopping |
| **Class imbalance** | Always predicts "continues" | Class weighting or threshold tuning |
| **Poor calibration** | P(0.9) but empirical ≈ 0.6 | Train longer or use different loss |
| **Sequence length mismatch** | Padding/truncation issues | Use variable-length RNNs (PyTorch default) |

---

## Quick Start Code

### Install

```bash
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
pip install -e .
```

### Train

```python
from easytpp.models import THP
from easytpp.data import EventSequence
import numpy as np

# Load your data
sequences = [...]  # List of EventSequence objects

# Train
model = THP(num_event_types=1, hidden_dim=128, num_heads=4)
model.fit(sequences, epochs=100, batch_size=32, learning_rate=0.001)

# Predict cessation
P_cessation = 1 - model.survival_function(sequence, T=600)  # 600 sec = 10 min
print(f"P(cessation in 10 min) = {P_cessation:.3f}")

# Decision
if P_cessation > 0.65:
    print("✓ TERMINATE ALERT")
else:
    print("✗ CONTINUE ALERT")
```

---

## Paper Ranking (for this problem)

### Tier 1: Must Read

1. **"Intensity-Free Learning of TPPs"** (Shchur et al., ICLR 2020)
   - *Why*: Direct cessation probability, closed-form survival
   - *Link*: https://arxiv.org/abs/1909.12127

2. **"Transformer Hawkes Process"** (Zuo et al., ICML 2020)
   - *Why*: Long-range dependencies in alert evolution
   - *Link*: https://arxiv.org/abs/2002.09291

### Tier 2: Important Context

3. **"Neural Temporal Point Processes: A Review"** (Shchur et al., IJCAI 2021)
   - *Why*: Comprehensive overview of architectures
   - *Link*: https://arxiv.org/abs/2104.03528

4. **"The Neural Hawkes Process"** (Mei & Eisner, NIPS 2017)
   - *Why*: Foundational continuous-time approach
   - *Link*: https://arxiv.org/abs/1612.09328

### Tier 3: If Time

5. **"EasyTPP"** (Xue et al., ICLR 2024)
   - *Why*: Practical benchmarking framework
   - *Link*: https://arxiv.org/abs/2307.08097

6. **"Spatio-temporal PP for Lightning"** (Meunier et al., arXiv 2024)
   - *Why*: Direct application to Météorage data
   - *Link*: https://arxiv.org/abs/2403.11564

---

## Expected Performance

### Baseline (30-min rule)
```
POD = 1.0  (always detects cessation after 30 min)
FAR = ~0.3 (30% of alerts already ended before 30 min)
Lead time = 0 min
```

### XGBoost (hand-crafted)
```
POD ≈ 0.88
FAR ≈ 0.08
Lead time ≈ +15 min
```

### Neural TPP (THP/IFT-TPP)
```
POD ≈ 0.90
FAR ≈ 0.06
Lead time ≈ +18 min  ← Target
```

---

## Resources

| Resource | Link | Type |
|---|---|---|
| EasyTPP Docs | https://ant-research.github.io/EasyTemporalPointProcess/ | Implementation |
| Shchur's Blog | https://shchur.github.io/blog/ | Tutorial |
| Original Papers | See references in NEURAL_TPP_RESEARCH.md | Theory |
| HawkesLib | https://hawkeslib.readthedocs.io/ | Simpler baseline |

---

## Questions to Answer While Reading

1. **How does my model handle irregular time intervals?**
   - Answer: Via log-transform of Δt or sinusoidal encoding

2. **How do I predict "last event"?**
   - Answer: Use survival function S(T) → P(cessation) = 1 - S(T)

3. **What's the difference between intensity-based and intensity-free?**
   - Answer: Intensity-based: λ(t) → need numerical integration; Intensity-free: p(τ) directly → closed-form

4. **Why is attention better than RNN?**
   - Answer: RNNs forget distant past; attention allows any event to influence any future event

5. **Can I use pre-trained models?**
   - Answer: Yes! Pre-train on all airports, fine-tune per airport (transfer learning)

---

**Last Updated**: March 2026
**Status**: Ready for Phase 1 implementation
**Expected Effort**: 3-4 weeks to production

