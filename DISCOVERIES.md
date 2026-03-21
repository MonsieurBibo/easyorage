# DISCOVERIES — EasyOrage DataBattle 2026

Journal des résultats et décisions clés.
Détail de la littérature → `docs/research/literature.md`
Spec features → `docs/specs/features.md`

---

## Données

- 5 aéroports : Ajaccio, Bastia, Biarritz, Nantes, Pise — rayon 30km, 2016-2025
- ~507k éclairs CG, target : `is_last_lightning_cloud_ground`
- **Warning** : `airport_alert_id` pas unique globalement → grouper par `(airport, airport_alert_id)`
- Distribution alertes : 14% à 1 éclair, 43% ≤ 5 éclairs, médiane = 8 éclairs

## Architecture pipeline

```
data/raw/           → data brutes Météorage
scripts/
  compute_features.py         → 102 features (59 lightning + 26 terrain + 17 météo)
  train_optuna_gain.py        → XGBoost, optimise Gain G directement
  train_optuna_silence.py     → XGBoost + points virtuels de silence (en cours)
  generate_predictions.py     → évaluation + predictions.csv
  eval_silence_strategy.py    → comparaison stratégies (flash-only vs silence-aware)
models/
  xgb_best.joblib             → meilleur modèle actuel
  predict_params.joblib       → K, base_threshold optimaux
```

## Résultats clés

### Modèle unifié vs par-aéroport (2026-03-10)
Un seul XGBoost sur tous les airports > modèles séparés.
Nantes profite le plus : +5.3% AUC (38% alertes triviales, 164 alertes train seulement).

### Ablation study (2026-03-10) — 94 features
| Groupe | AUC moyen |
|---|---|
| Lightning only (51) | **0.907** |
| + Terrain (77) | 0.909 (+0.002) |
| + Météo ERA5 (94) | 0.908 (météo nuit à Nantes) |
→ Features lightning dominent. Météo : gain nul voire négatif.

### Top features (XGBoost gain)
1. `fr_log_slope` — décroissance log flash rate (Schultz 2009)
2. `rolling_ili_5`, `rolling_ili_3`
3. `flash_rate_3` — taux très récent
4. `ili_s`, `lightning_rank`

### Meilleur modèle — XGBoost Optuna Gain (2026-03-24)
Optimise directement Gain G (pas AUC). 60 trials, 102 features, K=2 consécutifs.

| | Eval local | Test officiel |
|---|---|---|
| **Gain** | **72h** | **41.7h** |
| **Risk** | 0.0162 ✓ | 0.0034 ✓ |
| θ optimal | 0.85 | 0.85 |

AUC = 0.9281. Params : n_estimators=561, max_depth=5, lr=0.022, K=2, threshold=0.36

### Stratégie silence-aware — TESTÉE, PAS MIEUX (2026-03-24)
Idée : générer des points virtuels toutes les minutes pendant les silences, réentraîner.
Training augmenté : 154k points (vs 42k réels), pos_weight=1.3 (vs 19.3).
60 trials Optuna → Gain eval=128.7h (K=3, threshold=0.41).

| Stratégie | Eval | Test |
|---|---|---|
| Flash-only (xgb_best) | 72h | **41.7h** ← meilleur |
| Silence K=3 (xgb_silence) | 128.7h | 33.0h |
| Silence K=1 (xgb_silence) | 90.2h | 0h (R > 0.02) |

**Conclusion** : surentraînement massif sur données virtuelles. Modèle silence ne généralise pas.
→ `models/xgb_silence.joblib` conservé pour référence mais pas utilisé pour les soumissions.

### GRU bidirectionnel — ⚠️ Bug leakage (2026-03-16)
AUC=1.0 trivial : la passe backward voit les éclairs futurs.
Fix : `bidirectional=False` dans `scripts/train_gru.py`. Non corrigé.

## Décisions architecturales

| Décision | Choix | Raison |
|---|---|---|
| Split | 80/20 temporel par alerte | Pas de fuite temporelle |
| Modèle | Unifié tous airports | Transfer learning, +2% AUC moyen |
| Percentiles ILI | Causaux (running intra-alerte) | Pas de lookahead, fonctionne sur test |
| Stratégie inférence | K-consécutifs | Évite les faux positifs isolés |
| Météo ERA5 | Incluse mais impact nul | Ablation montre 0 gain |
| CNN orographie | ❌ Rejeté | 5 aéroports = mémorisation, pas apprentissage |

## Survival Analysis (2026-03-11)
| Modèle | C-index | AUC |
|---|---|---|
| Cox PH (18 feat.) | 0.7388 | 0.9080 |
| XGBoost AFT | 0.7414 | 0.8880 |
| XGBoost clf | 0.7198 | **0.9205** |
→ XGBoost classification reste meilleur pour l'objectif binaire.

## Pièges techniques
- `uv run --script` (pas `uv run python`) pour les scripts inline deps
- XGBoost AFT : bounds en secondes brutes, pas log(t) — XGBoost applique log en interne
- Polars `rolling_sum_by` incompatible avec `.over()` → rolling count-based uniquement
- `sigma_level` : `.replace(0.0, 1.0)` sur Polars Expr — vérifier version si bug
- `generate_predictions.py` : charger `predict_params.joblib` pour K et threshold — sinon defaults sous-optimaux
- Silence retraining : augmentation data virtuelle → surentraîne fort (eval/test gap ×3.8 vs ×1.7 pour flash-only)
