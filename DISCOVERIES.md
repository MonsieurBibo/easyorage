# Discoveries - EasyOrage (DataBattle 2026 Météorage)

## Contexte

Prédire la fin d'un orage autour d'aéroports européens à partir de données d'éclairs (Météorage).
Objectif : modèle probabiliste estimant la fin réelle d'un orage, pour réduire le temps d'attente
par rapport à la règle classique des 30 minutes.

## Données

- **Source** : Météorage, 10 ans (2016-2025), rayon 30km autour de chaque aéroport
- **Volume** : ~507k éclairs, 13 colonnes
- **Aéroports** : Pise (157k), Bastia (126k), Biarritz (115k), Ajaccio (73k), Nantes (37k)
- **Bron** : listé dans le doc mais absent des données (à clarifier)
- **Colonnes clés** :
  - `date`, `lon`, `lat` : horodatage UTC et position
  - `amplitude` : polarité + intensité max du courant (kA)
  - `maxis` : erreur de localisation théorique (km)
  - `icloud` : True = intra-nuage, False = nuage-sol
  - `dist`, `azimuth` : distance et direction par rapport à l'aéroport
  - `airport_alert_id` : identifiant de l'alerte (seulement <20km)
  - `is_last_lightning_cloud_ground` : target potentielle (seulement <20km)
- **Warning Pise** : système d'enregistrement différent en 2016 pour éclairs intra-nuage

## Revue de littérature

### Approches principales (cessation forecasting)

1. **Seuils radar** (approche dominante) : réflectivité à des isothermes (-5, -10, -15°C).
   Quand le seuil n'est plus dépassé → timer → cessation déclarée.
   - Mosier et al. 2015 : 75 algorithmes testés, POD 0.96-1.0, FAR 0.32-0.38, lead time 12-21 min
   - Preston & Fuelberg 2022 : ZH >= 40 dBZ à -5°C + 15 min wait

2. **GLM / Régression logistique** (Shafer & Fuelberg 2019) :
   Modèle probabiliste à chaque éclair : "est-ce le dernier ?". ~99% correct.

3. **VAHIRR** (Schultz et al. 2010) : approche physique, VAHIRR <= 10 dBZ-km → risque négligeable.

4. **Réseaux de capteurs de champ électrique** (Bitzer, FSU) :
   Field mills + radar → advisory finie ~10 min après cessation réelle (vs 30 min baseline).

5. **Deep Learning** (AFIT) : CRNN sur données field mill → TPR 77.6%, FPR 8.3%.

### Features les plus prédictives (littérature)

| Feature | Description |
|---|---|
| Réflectivité radar à isothermes | ZH à -5, -10, -15, -20°C |
| Présence de grésil (graupel) | Proxy du mécanisme de charge |
| **Intervalle inter-éclairs** | **Meilleur prédicteur unique** (Stano et al. 2010) |
| Taux d'éclairs et sa tendance | Décroissance du flash rate → signal de cessation |
| Ratio IC:CG | Évolue au cours du cycle de vie de l'orage |
| Distance/concentration spatiale | Dispersion des éclairs = orage en fin de vie |

### Gap identifié : pas d'analyse de survie dans la littérature

Aucune étude publiée n'applique formellement du survival analysis (Cox, Kaplan-Meier)
au problème de cessation. C'est une opportunité claire : le problème est naturellement
un problème de time-to-event avec covariables temporelles.

### Baseline universelle : la règle des 30 minutes

Attendre 30 min après le dernier éclair. Utilisée par Météorage, NOAA, aéroports.
Les meilleurs modèles réduisent à ~10 min de wait post-cessation.

### Métriques d'évaluation classiques

- **POD** (Probability of Detection) : fraction des cessations correctement identifiées
- **FAR** (False Alarm Ratio) : fraction des cessations déclarées trop tôt (danger !)
- **CSI** (Critical Success Index) = hits / (hits + misses + false alarms)
- **Lead time** : minutes gagnées vs baseline 30 min
- **Coût asymétrique** : fausse alerte (cessation prématurée) >> alerte manquée

### Références clés

- Stano, Fuelberg & Roeder 2010 - Empirical cessation schemes, KSC
- Mosier, Schultz, Carey & Petersen 2015 - Polarimetric radar algorithms
- Shafer & Fuelberg 2019 - GLM probabiliste sur radar dual-pol
- Preston & Fuelberg 2022 - Radar + LMA thresholds, Washington D.C.
- Bitzer ~2016 (FSU thesis) - Field mill + radar combined
- Schultz et al. 2010 - VAHIRR physically-based

## Idées pour notre approche (sans radar)

Nous n'avons **pas de données radar**, donc les approches dominantes (seuils de réflectivité)
ne sont pas directement applicables. Nos features devront être construites uniquement à partir
des caractéristiques spatio-temporelles des éclairs :

- **Temporelles** : intervalle inter-éclairs, flash rate (rolling window), accélération/décélération
- **Spatiales** : dispersion spatiale, dérive du centroïde, concentration vs éparpillement
- **Physiques** : ratio IC/CG, amplitude moyenne, évolution de l'amplitude
- **Géométriques** : direction de déplacement, vitesse de propagation
- **Par aéroport** : features spécifiques au lieu (climatologie locale)

Approches envisagées :
1. **Survival analysis** (Cox PH, modèles accélérés) - gap dans la littérature
2. **Classification séquentielle** : à chaque éclair, P(dernier éclair CG de l'alerte)
3. **Gradient boosting** (XGBoost/LightGBM) avec features engineered
4. **Analyse par aéroport** : clustering des types d'orages par lieu

## Stack & outils de visualisation

- **marimo** : notebook réactif (DAG), polars, altair, mo.sql (DuckDB)
- **openlayers** (`openlayers` package) : widget marimo pour cartes interactives
  - `VectorLayer` / `VectorSource` : points d'éclairs sur carte
  - `HeatmapLayer(weight=["get", "field"])` : heatmap pondérée (amplitude, densité)
  - `MapWidget` + `add_tooltip()` : carte interactive avec infos au survol
  - Couplable avec `mo.ui.slider` pour animation temporelle / filtrage
  - Ref : https://github.com/marimo-team/gallery-examples/blob/main/notebooks/geo/earthquake.py
