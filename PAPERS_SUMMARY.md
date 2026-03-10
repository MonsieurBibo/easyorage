# Quick Reference: Top 10 Papers for EasyOrage (2023-2025)

## Ranked by Relevance to Lightning Cessation Forecasting

### 🥇 Tier 1: Highest Relevance (Score 5/5)

#### **1. FlashBench: Hybrid Physics-ML Lightning Nowcasting (2023)**
- **URL**: https://arxiv.org/abs/2305.10064
- **Authors**: Singh et al.
- **Task**: Real-time lightning nowcasting (0-3h)
- **Key Metric**: POD=0.73, FAR=0.26, ETS=0.559
- **Why**: Lightning-specific, hybrid physics-ML, operational metrics match ours
- **Data**: WWLLN (analogous to Météorage)
- **Deployment**: Google Earth Engine (cloud-ready)
- **Action**: Study framework for Phase 2 hybrid model

#### **2. DeepLight: Lightning Prediction with Uncertainty (2025)**
- **URL**: https://arxiv.org/abs/2508.07428
- **Authors**: Arifin et al.
- **Task**: Lightning prediction 1-6h with asymmetric uncertainty
- **Key Metric**: ETS improvement +30% (1h), +18% (3h), +8% (6h)
- **Why**: MB-ConvLSTM + Hazy Loss (asymmetric cost), no NWP required
- **Data**: Radar + historical lightning (replaceable with Météorage)
- **Action**: Benchmark against XGBoost; adapt architecture for Phase 2

#### **3. Deep Diffusion Model for Satellite Thunderstorm (DDMS, 2024)**
- **URL**: https://www.pnas.org/doi/10.1073/pnas.2517520122 | Code: https://github.com/Applied-IAS/DDMS
- **Authors**: Dai, Li, Fang, et al.
- **Task**: 4-hour thunderstorm nowcasting (satellite-only)
- **Key Metric**: 20M km² coverage, 4 km × 15 min resolution, outperforms NowcastNet
- **Why**: Satellite-only (no radar required), longest lead time, publicly available code
- **Data**: FengYun-4A brightness temperature
- **Action**: Adapt diffusion architecture to 1D Météorage sequences for Phase 3

---

### 🥈 Tier 2: High Relevance (Score 4/5)

#### **4. Thunderstorm Nowcasting - Multi-Hazard Deep Learning (2023)**
- **URL**: https://agipubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022GL101626 | Code: https://zenodo.org/records/7157986
- **Authors**: Leinonen, Hamann, Germann, Sideris
- **Task**: Joint nowcasting of lightning, hail, heavy precipitation
- **Key Metric**: POD/FAR not explicitly stated; radar shown as dominant (70%+)
- **Why**: Multi-hazard fusion strategy, DEM-aware, data source importance quantified
- **Data**: Radar, lightning, satellite, NWP, DEM
- **Action**: Borrow multi-source fusion weighting strategy; adapt loss functions

#### **5. Skilful Nowcasting of Extreme Precipitation with NowcastNet (2023)**
- **URL**: https://www.nature.com/articles/s41586-023-06184-4
- **Authors**: Nature publication (DeepMind-adjacent)
- **Task**: Extreme precipitation nowcasting (2048×2048 km, 3h lead)
- **Key Metric**: Multiscale accuracy, ensemble-capable
- **Why**: Hybrid physics-data network (deterministic + stochastic), proven on continental scale
- **Data**: High-res radar (NEXRAD, CINRAD)
- **Action**: Study hybrid architecture for storm decay phase modeling

#### **6. Aerosol-Informed Lightning Nowcasting (2023)**
- **URL**: https://www.nature.com/articles/s41612-023-00451-x
- **Authors**: NOAA/NCAR
- **Task**: Hourly lightning nowcasts (continuous spatial fields)
- **Key Metric**: Satellite + LightGBM; metrics not fully detailed
- **Why**: Feature engineering approach (aerosols→microphysics); satellite-based
- **Data**: GLM (satellite) + MERRA2 (aerosols) + ECMWF
- **Action**: Study feature engineering strategy; benchmark LightGBM vs XGBoost

---

### 🥉 Tier 3: Moderate Relevance (Score 3-4/5)

#### **7. Hybrid AI-Enhanced Lightning Flash Prediction (2024)**
- **URL**: https://www.nature.com/articles/s41467-024-44697-2
- **Authors**: Nature Communications
- **Task**: Medium-range lightning prediction (2-day lead)
- **Key Metric**: AI enhances ECMWF deterministic model significantly
- **Why**: Demonstrates AI improves physics-based predictions
- **Limitation**: Long lead (2 days), not nowcasting
- **Action**: Consider for Phase 3+ (medium-range hybrid enhancement)

#### **8. Precipitation Nowcasting with Diffusion Transformer & Causal Attention (2024)**
- **URL**: https://arxiv.org/abs/2410.13314
- **Authors**: Multiple (2024)
- **Task**: Extreme precipitation nowcasting with causal attention
- **Key Metric**: CSI +15% (heavy) / +8% (moderate)
- **Why**: Causal attention captures temporal ordering without future leakage
- **Data**: Radar
- **Action**: Attention mechanisms useful for feature importance visualization

#### **9. Skilful Precipitation Nowcasting with DGMR (2021)**
- **URL**: https://www.nature.com/articles/s41586-021-03854-z | Code: https://github.com/openclimatefix/skillful_nowcasting
- **Authors**: DeepMind & UK Met Office
- **Task**: GAN-based precipitation nowcasting (90 min lead)
- **Key Metric**: Ranked #1 by 58 meteorologists (89% preference)
- **Why**: Generative approach balances intensity/extent; extensive code
- **Limitation**: Radar-only, focused on precipitation
- **Action**: Reference for ensemble generation methods (Phase 3+)

#### **10. Physics-Informed Machine Learning for Weather (Survey, 2024)**
- **URL**: https://arxiv.org/html/2403.18864v1
- **Title**: Interpretable Machine Learning for Weather and Climate Prediction: A Survey
- **Why**: Comprehensive review of physics-ML hybrid approaches
- **Action**: Deep dive for Phase 3 (theoretical foundations for hybrid model)

---

## Implementation Roadmap for EasyOrage

### Phase 1 (Week 1-2): Baseline
- Implement empirical rules (ILI + exponential decay)
- Target: POD ~0.85-0.90
- Reference: Stano et al. 2010 (cited in Discoveries.md)

### Phase 2 (Week 3-4): ML Enhancement
**Primary**: XGBoost + physics-ML fusion (#1, #4, #6)
- Feature engineering: ILI, flash-rate decay, IC/CG evolution, DEM
- Per-airport stratification

**Secondary**: DeepLight-style ConvLSTM (#2)
- MB-ConvLSTM architecture
- Hazy Loss for asymmetric cost

### Phase 3 (Week 5-8, if resources)
**Option A**: DDMS diffusion model adaptation (#3)
- Generate plausible future flash sequences
- Extract cessation time from generated trajectories

**Option B**: Survival analysis (Cox PH) — novel contribution
- Time-to-event framework
- Interpretable hazard ratios
- Uncharted territory in lightning literature

**Option C**: Ensemble with uncertainty quantification
- Bootstrap or Bayesian approaches
- Probabilistic output (not just POD/FAR)

---

## Critical URLs (Bookmarks)

### Code Repositories
- DDMS (diffusion for satellite): https://github.com/Applied-IAS/DDMS
- DGMR (GAN for radar): https://github.com/openclimatefix/skillful_nowcasting
- NowcastNet (multi-modal): Contact via Nature publication
- Leinonen multi-hazard: https://zenodo.org/records/7157986

### arXiv Papers (Direct Access)
- FlashBench: https://arxiv.org/abs/2305.10064
- DDMS: https://arxiv.org/abs/2404.10512
- DeepLight: https://arxiv.org/abs/2508.07428
- Leinonen: https://arxiv.org/abs/2211.01001

### Nature Publications
- NowcastNet: https://www.nature.com/articles/s41586-023-06184-4
- DGMR: https://www.nature.com/articles/s41586-021-03854-z
- PNAS DDMS: https://www.pnas.org/doi/10.1073/pnas.2517520122

---

## Key Metrics & Definitions

| Metric | Definition | Typical Range | Target for Cessation |
|---|---|---|---|
| **POD** | Probability of Detection = TP/(TP+FN) | 0.70-0.96 | >0.85 |
| **FAR** | False Alarm Ratio = FP/(FP+TP) | 0.08-0.40 | <0.15 |
| **ETS** | Equitable Threat Score | 0.40-0.70 | >0.55 |
| **Lead Time** | Minutes gained vs. baseline (30 min) | 5-30 min | >15 min |
| **F1 Score** | Harmonic mean precision/recall | 0.50-0.80 | >0.65 @ 5 min |
| **C-Index** | Concordance (survival analysis) | 0.70-0.90 | >0.80 |

---

## Research Gaps (Opportunities for Publication)

1. ✅ **Météorage-specific ML nowcasting** (no published work)
2. ✅ **Survival analysis for cessation** (Cox PH, Weibull — uncharted)
3. ✅ **GNNs on Météorage station topology** (unexplored)
4. ✅ **Diffusion models for 1D temporal sequences** (generalization of DDMS)

Any of these = publishable contribution (high-impact venue potential)

---

**Last Updated**: March 10, 2026
**Scope**: 20+ papers analyzed from 2023-2025
**Format**: Quick-reference index with full analysis in LITERATURE_REVIEW_2023-2025.md
