# Feature Engineering Specification for Lightning Cessation Prediction

## Overview

This document provides detailed specifications for engineered features derived from ERA5 atmospheric indices (CAPE, CIN, Lifted Index) to predict thunderstorm cessation timing.

## Dataset Context

- **Input Data**: ERA5 hourly reanalysis (CAPE, CIN, LI)
- **Target Variable**: Time until lightning cessation (hours) or binary (ceased/ongoing)
- **Time Window**: Alert start to cessation event
- **Spatial Resolution**: Point-based (or interpolated to alert location)

---

## Feature Categories and Specifications

### 1. Point-in-Time Features (Alert Start)

These capture the instantaneous atmospheric state when the alert is issued.

#### 1.1 CAPE at Alert Start
```
Feature Name: CAPE_at_alert_start
Type: Numeric (continuous)
Unit: J/kg
Range: [0, 10000]
Computation: CAPE value at alert_time (t0)
Missing Value: Forward/backward fill from nearest available hourly
Physical Interpretation: Updraft strength available at alert onset
ML Importance: Medium (initialization condition)
```

#### 1.2 CIN at Alert Start
```
Feature Name: CIN_at_alert_start
Type: Numeric (continuous)
Unit: J/kg
Range: [-1000, 0] (should be non-positive)
Computation: CIN value at alert_time (t0)
Missing Value: Forward/backward fill
Physical Interpretation: Energy barrier preventing convection initiation
ML Importance: Medium (contextualizes CAPE)
```

#### 1.3 Lifted Index at Alert Start
```
Feature Name: LI_at_alert_start
Type: Numeric (continuous)
Unit: K (Kelvin)
Range: [-10, +5]
Computation: LI value at alert_time (t0)
Missing Value: Forward/backward fill
Physical Interpretation: Parcel instability measure
ML Importance: Medium (correlated with CAPE, may capture tail behavior)
```

#### 1.4 K-Index at Alert Start (if available)
```
Feature Name: K_index_at_alert_start
Type: Numeric (continuous)
Unit: K
Range: [0, 100]
Computation: (T850 - T500) + Td850 - (T700 - Td700)
Missing Value: Compute from ERA5 T, Td at 850, 700, 500 hPa
Physical Interpretation: Air mass instability
ML Importance: Low-Medium (often correlated with CAPE)
```

---

### 2. Temporal Change Features (Most Critical)

These capture atmospheric evolution during the alert window. **Most important for cessation prediction.**

#### 2.1 CAPE Change Over Alert Duration
```
Feature Name: CAPE_delta
Type: Numeric (continuous)
Unit: J/kg
Range: [-4000, +2000]
Computation: CAPE_at_alert_end - CAPE_at_alert_start
         = CAPE(t0 + duration) - CAPE(t0)
Time Window: [t0, t_cessation] or [t0, t_cessation+1h]
Missing Value: Skip if end time missing; use last available hour
Physical Interpretation: CAPE depletion magnitude
Expected Pattern: Negative CAPE_delta → cessation likely
ML Importance: CRITICAL (directly measures storm evolution)
Key Insight: Research gap - this relationship not empirically validated yet
```

#### 2.2 CAPE Depletion Rate
```
Feature Name: CAPE_depletion_rate
Type: Numeric (continuous)
Unit: J/kg per hour
Range: [-1000, +100]
Computation: CAPE_delta / max(alert_duration_hours, 1)
Physical Interpretation: Rate of CAPE reduction per hour
Expected Pattern: Larger negative rate → faster cessation
ML Importance: CRITICAL (normalizes by alert duration)
Note: Handles varying alert durations (30 min to 6 hours)
```

#### 2.3 CAPE Relative Change Percentage
```
Feature Name: CAPE_pct_change
Type: Numeric (continuous)
Unit: Percent (%)
Range: [-100, +200]
Computation: (CAPE_delta / max(CAPE_at_alert_start, 50)) * 100
Floor: If CAPE_at_alert_start < 50, use 50 to avoid division issues
Physical Interpretation: Percentage depletion relative to initial value
Expected Pattern: -80% CAPE change → strong cessation signal
ML Importance: High (dimensionless, comparable across storms)
```

#### 2.4 CAPE Minimum During Alert
```
Feature Name: CAPE_min_during_alert
Type: Numeric (continuous)
Unit: J/kg
Range: [0, 6000]
Computation: min(CAPE[t0:t_cessation])
Temporal Resolution: Hourly
Physical Interpretation: Most unstable conditions available to storm
Expected Pattern: Low min CAPE → rapid cessation
ML Importance: Medium-High
```

#### 2.5 CAPE Maximum During Alert
```
Feature Name: CAPE_max_during_alert
Type: Numeric (continuous)
Unit: J/kg
Range: [0, 8000]
Computation: max(CAPE[t0:t_cessation])
Physical Interpretation: Peak instability during event
ML Importance: Medium (captures storm intensity potential)
```

#### 2.6 CAPE Mean During Alert
```
Feature Name: CAPE_mean_during_alert
Type: Numeric (continuous)
Unit: J/kg
Range: [0, 4000]
Computation: mean(CAPE[t0:t_cessation])
Temporal Resolution: Hourly
Physical Interpretation: Average available energy during alert
ML Importance: Medium (summary statistic)
```

#### 2.7 CAPE Standard Deviation During Alert
```
Feature Name: CAPE_std_during_alert
Type: Numeric (continuous)
Unit: J/kg
Range: [0, 2000]
Computation: std(CAPE[t0:t_cessation])
Physical Interpretation: Variability/stability of CAPE during alert
Expected Pattern: High std → fluctuating instability → uncertain cessation
ML Importance: Medium (captures fluctuations)
```

#### 2.8 CIN Evolution
```
Feature Name: CIN_delta
Type: Numeric (continuous)
Unit: J/kg
Range: [-500, +500]
Computation: CIN_at_alert_end - CIN_at_alert_start
Physical Interpretation: Change in convective inhibition
Expected Pattern: CIN increasing (becoming more inhibitory) → cessation
ML Importance: Medium (captures environment stabilization)
```

#### 2.9 Lifted Index Trend
```
Feature Name: LI_trend
Type: Numeric (continuous)
Unit: K per hour
Range: [-5, +5]
Computation: (LI_at_alert_end - LI_at_alert_start) / alert_duration_hours
Physical Interpretation: Rate of atmospheric stabilization
Expected Pattern: Positive LI_trend (warming/stabilization) → cessation
ML Importance: Medium
```

---

### 3. Climatological Anomaly Features

These normalize indices by historical baselines, capturing "how unusual" current conditions are.

#### 3.1 CAPE Anomaly (Absolute)
```
Feature Name: CAPE_anomaly
Type: Numeric (continuous)
Unit: J/kg
Range: [-2000, +3000]
Computation: CAPE_at_alert_start - CAPE_climatological_mean

Climatology Computation:
  1. Extract ERA5 CAPE for same (month, day) ±15 days for all years
  2. Compute mean: CAPE_clim(month, day, lat, lon)
  3. Anomaly = CAPE(t0) - CAPE_clim

Temporal Resolution: Calendar day (with ±15 day smoothing)
Spatial Resolution: Match to alert location (0.25° ERA5 grid)
Historical Period: 1980-2024 (45 years minimum)

Physical Interpretation: Deviation from typical instability for date/location
Expected Pattern: Positive anomaly (more unstable than normal) → longer duration
ML Importance: HIGH (captures relative context)
```

#### 3.2 CAPE Anomaly (Percentile)
```
Feature Name: CAPE_anomaly_percentile
Type: Numeric (continuous, rank-based)
Unit: Percentile [0, 100]
Range: [0, 100]
Computation: 
  percentile_rank(CAPE_at_alert_start, 
                  historical_CAPE_distribution(month, day, location))

Definition: What percentile does current CAPE fall into?
  0 = lowest 1% historical CAPE
  50 = median CAPE for this date/location
  100 = highest 1% historical CAPE

Physical Interpretation: Relative position in historical distribution
Expected Pattern: High percentile → unusually unstable → potentially longer duration
ML Importance: HIGH (captures tail behavior, less skewed than raw anomaly)
Advantage: Robust to outliers, interpretable
```

#### 3.3 CIN Anomaly
```
Feature Name: CIN_anomaly
Type: Numeric (continuous)
Unit: J/kg
Range: [-500, +500]
Computation: CIN_at_alert_start - CIN_climatological_mean(month, day, location)

Physical Interpretation: Deviation in inhibition from typical
Expected Pattern: Positive CIN_anomaly (more inhibition than normal) → shorter duration
ML Importance: Medium
```

#### 3.4 Seasonal Index (Categorical Representation)
```
Feature Name: season
Type: Categorical
Categories: ['DJF', 'MAM', 'JJA', 'SON']  or numeric [0-3]
Computation: Derived from alert date month
  DJF: Dec, Jan, Feb (Winter)
  MAM: Mar, Apr, May (Spring)
  JJA: Jun, Jul, Aug (Summer)
  SON: Sep, Oct, Nov (Autumn)

Use: As interaction term with CAPE features (storm behavior season-dependent)
ML Importance: Medium (may be captured by climatology, but explicit helps)
```

---

### 4. Ratio and Interaction Features

These combine multiple indices to capture multi-dimensional atmospheric state.

#### 4.1 CAPE-to-CIN Ratio
```
Feature Name: CAPE_CIN_ratio
Type: Numeric (continuous)
Unit: Dimensionless ratio
Range: [0, 100]
Computation: CAPE_at_alert_start / (abs(CIN_at_alert_start) + 50)
Numerator: Instability
Denominator: Inhibition (add 50 J/kg offset to avoid division by zero near CIN=0)

Physical Interpretation: Balance of instability vs. inhibition
Expected Pattern: High ratio → favorable for sustained convection → longer duration
ML Importance: Medium-High (captures relative contributions)
```

#### 4.2 CAPESHEAR Index
```
Feature Name: CAPESHEAR
Type: Numeric (continuous)
Unit: J/kg × (m/s) = unconventional
Range: [0, 20000]
Computation: CAPE_at_alert_start × bulk_wind_shear

Bulk Wind Shear:
  SFC_to_6km_shear = sqrt(
    (u(600hPa) - u(SFC))^2 + 
    (v(600hPa) - v(SFC))^2
  )

Data Source: ERA5 u, v components at surface and 600 hPa
Alternative: If not available, use T500-T850 as proxy for shear-proxying

Physical Interpretation: Combined effect of instability + organization
Published Importance: HIGH (from 2019 hail ML prediction paper)
Stormy Behavior: High CAPESHEAR = organized, rotating, long-lived storms
Expected Pattern: Higher CAPESHEAR → longer alert duration
ML Importance: CRITICAL (explicitly recommended in literature)
```

#### 4.3 Instability Index (Custom)
```
Feature Name: instability_composite
Type: Numeric (continuous)
Unit: Composite (normalized)
Range: [0, 10] (after normalization)
Computation: (CAPE_at_alert_start / 1000) + (10 + LI_at_alert_start)

Interpretation:
  - CAPE term: Normalized to J/kg units
  - LI term: Shifted so negative (unstable) values contribute positively
  
Physical Interpretation: Multi-index stability measure
ML Importance: Low-Medium (usually better to keep indices separate for SHAP analysis)
```

#### 4.4 CAPE Depletion Acceleration
```
Feature Name: CAPE_depletion_acceleration
Type: Numeric (continuous)
Unit: J/kg per hour^2
Range: [-500, +500]
Computation: (CAPE_depletion_rate(t=t_end) - CAPE_depletion_rate(t=t_start)) 
           / alert_duration_hours

Data Requirements: Need at least 3 hourly CAPE values
Physical Interpretation: Is depletion speeding up or slowing?
Expected Pattern: Positive acceleration (depletion increasing) → cessation imminent
ML Importance: Low (requires more data points, redundant with depletion rate)
Recommendation: Include only if alert duration > 3 hours consistently
```

---

### 5. Binary/Categorical Features

These encode regime or threshold-based conditions.

#### 5.1 CAPE Regime at Alert Start
```
Feature Name: CAPE_regime
Type: Categorical/Ordinal
Categories: ['very_low', 'low', 'moderate', 'high', 'extreme']
Thresholds (operational NOAA/SPC): 
  very_low: CAPE < 500 J/kg
  low: 500 ≤ CAPE < 1000
  moderate: 1000 ≤ CAPE < 2000
  high: 2000 ≤ CAPE < 3500
  extreme: CAPE ≥ 3500

Numeric Encoding: 0, 1, 2, 3, 4
Physical Interpretation: Operational weather classification
ML Use: Can be one-hot encoded or ordinal depending on algorithm
Importance: Medium (ordinal nature important for trees)
```

#### 5.2 Crossed CAPE-CIN Regime
```
Feature Name: atmosphere_regime
Type: Categorical
Categories: Combination of CAPE × CIN bins
Thresholds:
  - High CAPE, Low |CIN| = "Favorable" → long duration expected
  - Low CAPE, High |CIN| = "Suppressed" → rapid cessation
  - High CAPE, High |CIN| = "Explosive" → unpredictable cessation
  - Low CAPE, Low |CIN| = "Stable" → no convection

Encoding: One-hot encode into separate binary features
ML Importance: Medium (captures non-linear interactions)
```

#### 5.3 Depletion Threshold Binary
```
Feature Name: rapid_CAPE_depletion
Type: Binary (0/1)
Threshold: CAPE_depletion_rate < -200 J/kg/hr
Meaning: Is CAPE being depleted rapidly?
Expected: 1 → cessation likely within 1-2 hours
ML Importance: Low-Medium (summary of depletion rate)
```

---

## Feature Engineering Implementation Pseudocode

```python
def engineer_alert_features(alert_row, era5_hourly_data, era5_climatology):
    """
    Generate all 25+ features for single alert.
    
    Inputs:
      alert_row: {alert_id, location_lat, location_lon, 
                  alert_start_time, alert_end_time, cessation_time}
      era5_hourly_data: DataFrame with hourly CAPE, CIN, LI, u, v
      era5_climatology: Dict with climatological means/percentiles
    
    Returns:
      feature_dict: Dict with all computed features
    """
    
    # Time windows
    t_start = alert_row['alert_start_time']
    t_end = alert_row['alert_end_time']
    t_cessation = alert_row['cessation_time']
    duration_hours = (t_end - t_start).total_seconds() / 3600
    
    # Extract relevant ERA5 data
    cape_ts = era5_hourly_data[
        (era5_hourly_data['time'] >= t_start) & 
        (era5_hourly_data['time'] <= t_cessation)
    ]['CAPE']
    cin_ts = era5_hourly_data[...]['CIN']
    li_ts = era5_hourly_data[...]['LI']
    
    # === CATEGORY 1: Point-in-time ===
    features = {
        'CAPE_at_alert_start': cape_ts.iloc[0],
        'CIN_at_alert_start': cin_ts.iloc[0],
        'LI_at_alert_start': li_ts.iloc[0],
    }
    
    # === CATEGORY 2: Temporal changes ===
    cape_start = cape_ts.iloc[0]
    cape_end = cape_ts.iloc[-1]
    cape_delta = cape_end - cape_start
    
    features.update({
        'CAPE_delta': cape_delta,
        'CAPE_depletion_rate': cape_delta / max(duration_hours, 1),
        'CAPE_pct_change': (cape_delta / max(cape_start, 50)) * 100,
        'CAPE_min_during_alert': cape_ts.min(),
        'CAPE_max_during_alert': cape_ts.max(),
        'CAPE_mean_during_alert': cape_ts.mean(),
        'CAPE_std_during_alert': cape_ts.std(),
        'CIN_delta': cin_ts.iloc[-1] - cin_ts.iloc[0],
        'LI_trend': (li_ts.iloc[-1] - li_ts.iloc[0]) / max(duration_hours, 1),
    })
    
    # === CATEGORY 3: Climatological anomalies ===
    cape_clim_mean = era5_climatology.get_mean(
        month=t_start.month, day=t_start.day, 
        lat=alert_row['location_lat'], 
        lon=alert_row['location_lon']
    )
    cape_anomaly = cape_start - cape_clim_mean
    
    features.update({
        'CAPE_anomaly': cape_anomaly,
        'CAPE_anomaly_percentile': era5_climatology.get_percentile(
            cape_start, month=t_start.month, day=t_start.day
        ),
    })
    
    # === CATEGORY 4: Ratios and interactions ===
    cin_start = cin_ts.iloc[0]
    features.update({
        'CAPE_CIN_ratio': cape_start / (abs(cin_start) + 50),
        'CAPESHEAR': cape_start * compute_bulk_wind_shear(...),
    })
    
    # === CATEGORY 5: Categorical ===
    features.update({
        'CAPE_regime': classify_cape_regime(cape_start),
        'season': get_season(t_start.month),
    })
    
    return features
```

---

## Missing Data Handling Strategy

| Feature | Strategy |
|---------|----------|
| CAPE_at_alert_start | Forward/backward fill within ±3 hours |
| CAPE_delta | Require both start and end; skip if only one available |
| CAPE_mean_during_alert | Interpolate hourly CAPE, then compute mean |
| CAPE_anomaly | Use nearest available historical date if exact date unavailable |
| CAPESHEAR | Interpolate u/v components if sparse |
| All features | Mark as missing if >50% of time window has gaps |

---

## Feature Normalization

### Before Training
```python
from sklearn.preprocessing import StandardScaler

# Numeric features
numeric_features = [
    'CAPE_at_alert_start', 'CAPE_delta', 'CAPE_depletion_rate',
    'CAPE_anomaly', 'CAPESHEAR', 'CIN_delta', ...
]

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X[numeric_features])

# Categorical features: one-hot encode
categorical_features = ['CAPE_regime', 'season', 'atmosphere_regime']
X_encoded = pd.get_dummies(X[categorical_features])

# Combine
X_final = pd.concat([
    pd.DataFrame(X_normalized, columns=numeric_features),
    X_encoded
], axis=1)
```

---

## Expected Feature Correlations

```
High Positive Correlations (watch for collinearity):
  - CAPE_at_alert_start <-> CAPE_max_during_alert (r ~ 0.85)
  - CAPE_delta <-> CAPE_depletion_rate (r ~ 0.95, skip one)
  - CAPE_anomaly <-> CAPE_at_alert_start (depends on seasonality)
  - CAPESHEAR <-> CAPE_at_alert_start (r ~ 0.70)

Low Correlations (good diversity):
  - CAPE_depletion_rate <-> CAPE_at_alert_start (r ~ -0.3)
  - CIN_delta <-> CAPE_delta (r ~ -0.1 to +0.3)
  - LI_trend <-> CAPE_depletion_rate (r ~ 0.5)

Collinearity Handling:
  1. Compute correlation matrix
  2. Drop one feature if |r| > 0.95
  3. Preferred drops: absolute values over rates/derivatives
```

---

## Feature Importance Validation

After training, expected feature importance rankings:

1. **CAPE_depletion_rate** - MOST IMPORTANT (directly predicts cessation)
2. **CAPE_delta** - VERY IMPORTANT (magnitude of change)
3. **CAPESHEAR** - IMPORTANT (organization/duration)
4. **CAPE_at_alert_start** - IMPORTANT (initialization)
5. **CAPE_mean_during_alert** - IMPORTANT (sustained instability)
6. **CAPE_anomaly** - IMPORTANT (relative context)
7. **CIN_at_alert_start** - MEDIUM (contextualizing)
8. **LI_at_alert_start** - MEDIUM-LOW (often redundant with CAPE)

If feature importances significantly differ from this, investigate:
- Data quality issues
- Temporal lag (features need t-1, t-2 lookback?)
- Interaction effects not captured
- Alert duration bias (long vs. short alerts behave differently)

---

## Document Version History

| Date | Author | Changes |
|------|--------|---------|
| 2026-03-10 | Research Summary | Initial specification based on literature review |

