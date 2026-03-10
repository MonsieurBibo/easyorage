# Lightning Cessation Prediction: Quick Reference Guide

## Key Findings Summary

### 1. CAPE Thresholds (Published Evidence)
- **Lightning Threshold**: 400-625 J/kg (Cheng et al., 2021, JGR)
  - Below 625 J/kg: Lightning suppressed (especially tropical oceans)
  - Implies: When CAPE drops below ~400, lightning likely ceases
- **Development Threshold**: 1000 J/kg (usually sufficient for strong storms)
- **CRITICAL GAP**: No published thresholds specifically for *cessation timing*

### 2. CAPE Depletion & Cessation (Research Status)
- **Direct Evidence**: NONE found in literature
- **Theoretical Support**: STRONG (entrainment dilutes updraft, reduces charge separation)
- **Action**: Engineer features empirically; test CAPE_depletion_rate vs. cessation timing
- **Expected Signal**: -200 to -400 J/kg/hr depletion → cessation within 1-2 hours (hypothesis)

### 3. Best Paper on ERA5 for Lightning (2025)
**"Identifying Lightning Processes in ERA5 Soundings with Deep Learning"** (GMD 2025)
- Used ~670 ERA5 features (cloud ice/snow, mass-field, wind variables)
- Found cloud ice/snow in mid/upper troposphere most predictive
- Implication: Raw ERA5 data may be better than pre-computed indices (CAPE, etc.)

### 4. Leinonen et al. (2022, NHESS) Key Findings
- Nowcasting (60 min lead time), not cessation prediction
- **106 features** from: radar (most important) > satellite > NWP > lightning
- Radar reflectivity patterns at 0°C, -10°C, -20°C levels critical
- Graupel/ice signatures and weak echo regions best predictors

### 5. Feature Importance Rankings (ML Studies)
1. CAPE + Wind Shear combined (CAPESHEAR index) - CRITICAL
2. Radar reflectivity temporal evolution
3. Upper-level winds (850, 500 hPa)
4. Lapse rate / vertical structure
5. Cloud ice/snow content (from DL study)
6. K-Index, Lifted Index (useful but CAPE superior)
7. CIN (better for initiation than cessation)

### 6. No Papers Found On:
- Open-Meteo + ERA5 for alert cessation (novel area)
- CAPE depletion rate vs. cessation timing (empirical validation needed)
- Alert-specific cessation features (novel research)

---

## Recommended Feature Engineering Strategy

### Must-Have Features (21 total)

#### Temporal Changes (CRITICAL - Start Here)
1. **CAPE_depletion_rate** = ΔCAPE / alert_duration_hours
2. **CAPE_delta** = CAPE_end - CAPE_start
3. **CAPE_pct_change** = (CAPE_delta / CAPE_start) * 100
4. **CAPE_mean_during_alert** = mean(CAPE over alert window)
5. **CAPE_std_during_alert** = std(CAPE over alert window)

#### Point-in-Time (Initialization)
6. **CAPE_at_alert_start**
7. **CIN_at_alert_start**
8. **LI_at_alert_start**

#### Climatological (Context)
9. **CAPE_anomaly** = CAPE_now - CAPE_climatological_mean
10. **CAPE_anomaly_percentile** = percentile rank of CAPE

#### Interactions (Published Importance)
11. **CAPESHEAR** = CAPE × bulk_wind_shear (CRITICAL - shown in hail ML)
12. **CAPE_CIN_ratio** = CAPE / (|CIN| + 50)

#### Categorical
13. **CAPE_regime** = categorical (very_low/low/moderate/high/extreme)
14. **season** = DJF/MAM/JJA/SON

#### Additional Stability Metrics
15. **CIN_delta** = CIN_end - CIN_start
16. **LI_trend** = (LI_end - LI_start) / duration_hours
17. **CAPE_min_during_alert**
18. **CAPE_max_during_alert**
19. **atmosphere_regime** = categorical (favorable/suppressed/explosive/stable)
20. **instability_composite** = (CAPE/1000) + (10 + LI)
21. **rapid_CAPE_depletion** = binary (1 if rate < -200 J/kg/hr)

### Implementation Pseudocode
```python
# For each alert:
cape_start = ERA5_CAPE[alert_start_time]
cape_end = ERA5_CAPE[alert_cessation_time]
cape_ts = ERA5_CAPE[alert_start_time : alert_cessation_time]

# Temporal changes
CAPE_delta = cape_end - cape_start
CAPE_depletion_rate = CAPE_delta / alert_duration_hours
CAPE_pct_change = (CAPE_delta / max(cape_start, 50)) * 100

# Climatology (precomputed)
CAPE_clim = climatology[month][day][lat][lon]
CAPE_anomaly = cape_start - CAPE_clim
CAPE_percentile = percentile_rank(cape_start, historical_dist)

# Interactions
wind_shear = compute_bulk_wind_shear(ERA5_u, ERA5_v)
CAPESHEAR = cape_start * wind_shear
CAPE_CIN_ratio = cape_start / (abs(CIN_start) + 50)

# Store features -> ML model
```

---

## Feature Importance Predictions

After training, expect this ranking:

| Rank | Feature | Importance | Confidence |
|------|---------|-----------|------------|
| 1 | CAPE_depletion_rate | HIGH (40-50%) | VERY HIGH |
| 2 | CAPE_delta | HIGH (30-40%) | VERY HIGH |
| 3 | CAPESHEAR | HIGH (20-30%) | HIGH |
| 4 | CAPE_at_alert_start | MEDIUM (10-20%) | MEDIUM |
| 5 | CAPE_mean_during_alert | MEDIUM (5-15%) | MEDIUM |
| 6 | CAPE_anomaly | MEDIUM (5-15%) | MEDIUM |
| 7 | CIN_at_alert_start | MEDIUM (5-10%) | MEDIUM |
| 8 | LI_at_alert_start | LOW-MEDIUM (2-8%) | LOW |

**If actual rankings differ significantly**: Check for data quality issues, temporal lags (t-1, t-2 features), or alert duration bias.

---

## Validation Checklist

- [ ] CAPE data spans ±6 hours around alert start/end
- [ ] ERA5 climatology built from 20+ years (1980-2024+)
- [ ] Missing values handled (forward/backward fill for CAPE)
- [ ] Collinear features removed (|r| > 0.95 threshold)
- [ ] Features normalized (standardization for neural nets, optional for trees)
- [ ] Categorical features one-hot encoded
- [ ] Train/test split stratified by season and alert duration
- [ ] Cross-validation on held-out alerts
- [ ] SHAP analysis for feature interpretability

---

## Expected Model Performance

### Baseline (If Predicting Cessation Timing)
- **Naive**: Always predict median cessation time ~ 2-4 hours
- **With CAPE Features**: Expected 20-40% improvement (R² or RMSE)

### Baseline (If Binary Classification: Ceased vs. Ongoing)
- **Naive**: Always predict majority class
- **With CAPE Features**: Expected 70-85% accuracy, AUC > 0.80

*Adjust based on actual alert dataset balance and duration distribution.*

---

## Research Gaps to Investigate

1. **Empirical CAPE Cessation Threshold**
   - Question: At what CAPE level does lightning cease 90% of time?
   - Method: Scatter plot CAPE vs. time_to_cessation
   - Expected answer: Likely 200-400 J/kg depending on latitude/season

2. **CAPE Depletion Rate Predictability**
   - Question: Can we predict cessation 1-3 hours in advance from CAPE trends?
   - Method: Lagged regression of CAPE_depletion_rate on cessation_time
   - Hypothesis: Depletion rates >300 J/kg/hr → cessation within 2 hours

3. **Seasonal Variation in CAPESHEAR Effect**
   - Question: Does CAPESHEAR impact storm duration equally year-round?
   - Method: Stratified analysis by season; interaction term: season × CAPESHEAR
   - Hypothesis: Effect stronger in spring/summer, weaker in winter

4. **Wind Shear Role in Cessation**
   - Question: Does high shear → organized storms → slower cessation?
   - Method: Separate high/low shear storms; compare cessation times
   - Hypothesis: High shear storms 30-50% longer duration

---

## Key References (Annotated)

### Must Read
1. **Cheng et al. (2021) - CAPE Threshold for Lightning**
   - [https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JD035621](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JD035621)
   - Core finding: 625 J/kg threshold over tropical ocean
   - Action: Check if applies to your geographic region

2. **Leinonen et al. (2022) - Nowcasting with ML**
   - [https://nhess.copernicus.org/articles/22/577/2022/](https://nhess.copernicus.org/articles/22/577/2022/)
   - Core finding: 106 features; radar > satellite > NWP
   - Action: Review feature importance rankings; adapt for cessation

3. **ERA5 + Deep Learning (2025) - GMD**
   - [https://gmd.copernicus.org/articles/18/1141/2025/](https://gmd.copernicus.org/articles/18/1141/2025/)
   - Core finding: Cloud ice/snow most predictive; raw features > indices
   - Action: Consider adding cloud microphysics if ERA5 extended available

### Nice to Have
4. **Hail Prediction ML (2019) - Era5 + Radar**
   - [https://www.sciencedirect.com/science/article/abs/pii/S0169809519300900](https://www.sciencedirect.com/science/article/abs/pii/S0169809519300900)
   - Key: CAPESHEAR, SHIP, HSI most important
   - Action: Use for wind shear proxy if CAPESHEAR unavailable

5. **Radar Cessation Guidance (2019) - WAF**
   - [https://journals.ametsoc.org/view/journals/wefo/34/3/waf-d-18-0144_1.xml](https://journals.ametsoc.org/view/journals/wefo/34/3/waf-d-18-0144_1.xml)
   - Key: Graupel presence, reflectivity at 0/-10/-20°C level best
   - Action: Compare CAPE features vs. radar-only baseline

---

## Implementation Timeline

### Week 1: Data Prep
- Download ERA5 hourly CAPE, CIN, LI (±10 years historical)
- Build climatology (20+ year mean for each calendar day)
- Extract features for all alerts

### Week 2: EDA
- Scatter: CAPE_depletion_rate vs. cessation_time
- Correlation matrix: All features vs. target
- Distribution: Compare long vs. short alerts

### Week 3: Baseline Model
- Random Forest with 5 basic features (CAPE_delta, CAPE_start, etc.)
- 10-fold CV; record baseline R² or AUC
- SHAP analysis for top 3 features

### Week 4: Full Model
- Train with all 21 features
- Hyperparameter tuning
- Validation on held-out set
- Compare to baseline

### Week 5: Analysis & Publication
- Feature importance ranking
- Temporal window optimization (which lookback window best?)
- Write-up: "CAPE Depletion as Predictor of Lightning Cessation"

---

## Questions to Ask Your Data

1. **"What fraction of alerts have CAPE dropping > 50% by cessation?"**
   - Answer guides CAPE_pct_change importance

2. **"Are long alerts (>4 hrs) associated with higher CAPESHEAR?"**
   - Answer: If yes, CAPESHEAR critical; if no, wind shear less important

3. **"Does CAPE_depletion_rate differ between daytime and nighttime alerts?"**
   - Answer: May need separate models; or include time-of-day feature

4. **"Are climatological anomalies (CAPE_anomaly) more predictive than absolute CAPE?"**
   - Answer: If yes, confirms published findings; if no, anomalies less useful

5. **"Does alert duration bias the model?"**
   - Answer: Stratify by duration bins; check feature importance stability

---

**Document Version**: 1.0  
**Generated**: March 10, 2026  
**Last Updated**: March 10, 2026  
**Status**: Ready for implementation

