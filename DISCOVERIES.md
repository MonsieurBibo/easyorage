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

## Modèles envisageables (revue approfondie)

### Modèles statistiques / ML classique

| Modèle | POD attendu | FAR attendu | Notes |
|---|---|---|---|
| **Règles empiriques ILI** | 92-95% | 1-3% | ILI seul → 92% POD. Baseline rapide. |
| **Régression logistique** par éclair | ~AUROC 0.92-0.96 | - | Shafer & Fuelberg 2019 : ~99% (avec radar) |
| **XGBoost / LightGBM** | ~POD 0.88-0.92 | ~FAR 0.05-0.10 | Meilleur compromis perf/interprétabilité |
| **Cox PH / Weibull AFT** | C-index 0.75-0.85 | - | Gap littérature — jamais appliqué à ce pb |
| **HMM** (4 états : growth/mature/decline/terminal) | 75-85% état correct | - | Interprétable, capture les phases |

### Deep learning

| Modèle | POD attendu | Notes |
|---|---|---|
| **LSTM / GRU** | 70-80% | Baseline DL, 1 semaine à implémenter |
| **Temporal Fusion Transformer (TFT)** | 80-85% | Attention interprétable, recommandé DL |
| **Neural Temporal Point Process** (Hawkes) | 80-90% | Adapté aux séquences irrégulières, très interprétable |
| ConvLSTM | 84-96%* | *Surtout prouvé sur données radar/spatiales |

Référence TFT : arXiv:1912.09363

### Modèles physiques (sans radar)

Basés sur 3 mécanismes physiques du vieillissement d'un orage :
1. **Épuisement microphysique** : updraft s'affaiblit → collisions graupel-glace cessent → flash rate décroît exponentiellement
2. **Érosion des charges** : stratification verticale s'effondre → ratio IC/CG décroît depuis le pic (3-8) vers <1
3. **Récupération du champ électrique** : déplétion des charges → intervalles inter-éclairs s'allongent (IFI > 120 s = phase de déclin)

| Approche | POD | FAR | Notes |
|---|---|---|---|
| **Fit exponentiel du flash rate** | 0.85-0.95 | 0.15-0.25 | Meilleur indicateur physique unique |
| Proxy field mill (IFI rolling median) | 0.80-0.90 | 0.20-0.30 | IFI > 120 s → déclin |
| Évolution ratio IC/CG | 0.70-0.80 | 0.25-0.35 | Pic → déclin = signal de cessation |
| Tracking centroïde (exit du rayon 20km) | 0.85-0.95 | 0.15-0.25 | Orage qui s'éloigne |
| Évolution amplitude / maxis | 0.65-0.80 | 0.25-0.40 | Signal secondaire, à combiner |

Ref : Schultz et al. 2010 (VAHIRR), Stano et al. 2010 (ILI empirique)

### Orographie — impact critique par aéroport

**Durée typique des alertes par aéroport (implication pour le modèle) :**

| Aéroport | Contexte orographique | Durée alerte typique | Complexité |
|---|---|---|---|
| **Nantes** | Plaine atlantique, flat | 15-60 min | Faible → modèle simple fiable |
| **Bastia** | Côte montagne Corse | 20-90 min | Moyenne |
| **Biarritz** | Côte Atlantique + Pyrénées | **BIMODALE** 20-30 min (brise de mer) vs 60-120 min (orages de montagne) | Haute → stratification nécessaire |
| **Ajaccio** | Intérieur montagneux Corse | 30-120 min (piégeage en vallée) | Haute |
| **Pise** | Plaine du Pô + Apennins | 30-180+ min (dynamique triple-point) | Très haute |

**Conséquences pour le modèle :**
- **Modèles par aéroport** obligatoires (pas un seul modèle global)
- Biarritz : stratifier par régime (brise de mer vs frontal/montagne)
- Features DEM à inclure : `elevation_range`, `TRI` (Terrain Ruggedness Index), `distance_to_ridge`, `channeling_index`

Refs : Coquillat et al. 2019/2022 (SAETTA 3D lightning, Corse — directement nos aéroports),
Leinonen et al. 2022 (seul modèle ML publié avec features DEM explicites)

### Notre approche retenue

**Phase 1 (baseline rapide)** : règles ILI + fit exponentiel flash rate → POD ~92%, lead time +15-20 min

**Phase 2 (modèle ML)** : XGBoost par aéroport avec features engineered :
- Temporelles : ILI (mean/std/max/trend), flash rate rolling (5/10/30 min), décroissance
- Spatiales : dispersion spatiale, dérive centroïde, distance au bord du rayon
- Physiques : ratio IC/CG glissant, fit exponentiel τ, fraction amplitude > 50 kA
- DEM (statiques par aéroport) : TRI, distance au relief dominant, indice de canalisation

**Phase 3 (optionnel)** : Cox PH ou TFT si le temps le permet

### Idées pour notre approche (sans radar)

Nous n'avons **pas de données radar**, donc les approches dominantes (seuils de réflectivité)
ne sont pas directement applicables. Nos features devront être construites uniquement à partir
des caractéristiques spatio-temporelles des éclairs :

- **Temporelles** : intervalle inter-éclairs, flash rate (rolling window), accélération/décélération
- **Spatiales** : dispersion spatiale, dérive du centroïde, concentration vs éparpillement
- **Physiques** : ratio IC/CG, amplitude moyenne, évolution de l'amplitude
- **Géométriques** : direction de déplacement, vitesse de propagation
- **Par aéroport** : features orographiques DEM + climatologie locale

## Stack & outils de visualisation

- **marimo** : notebook réactif (DAG), polars, altair, mo.sql (DuckDB)
- **openlayers** (`openlayers` package) : widget marimo pour cartes interactives
  - `VectorLayer` / `VectorSource` : points d'éclairs sur carte
  - `HeatmapLayer(weight=["get", "field"])` : heatmap pondérée (amplitude, densité)
  - `MapWidget` + `add_tooltip()` : carte interactive avec infos au survol
  - Couplable avec `mo.ui.slider` pour animation temporelle / filtrage
  - Ref : https://github.com/marimo-team/gallery-examples/blob/main/notebooks/geo/earthquake.py

---

# Analyse approfondie de papiers clés — Mars 2026

## PARTIE 1 : DeepLight (arXiv:2508.07428)

**Référence** : "Lightning Prediction under Uncertainty: DeepLight with Hazy Loss", Md Sultanul Arifin et al., arXiv:2508.07428, Août 2025

### Architecture MB-ConvLSTM exacte

L'architecture Multi-Branch ConvLSTM utilise **4 branches parallèles** avec des kernels de convolution de tailles 3×3, 5×5, 7×7 et 11×11. Chaque branche applique une activation ReLU indépendamment.

Mécanisme de combinaison :
1. Les 4 sorties de branches sont **concaténées**
2. Une convolution 1×1 fusionne (fusion) les représentations concaténées
3. Le résultat fusionné est **découpé channel-wise en 4 partitions** pour les gates LSTM : forget gate, input gate, candidate gate, output gate

Les branches traitent l'**input courant X_t** et l'**état caché précédent H_{t-1}** en parallèle via des multi-branch blocks séparés. L'idée est que les noyaux multi-échelles capturent simultanément des structures spatiales à différentes résolutions (locales 3×3 et globales 11×11).

### Hazy Loss — définition mathématique complète

La Hazy Loss est construite en 3 étapes :

**Étape 1 — Gaussian Blur 3D (spatio-temporel) :**
```
[K_m]_{x1,x2,x3} = 1/((2π)^(3/2) σ1 σ2 σ3) × exp(-(x1²/(2σ1²) + x2²/(2σ2²) + x3²/(2σ3²)))
L^blur = GaussianBlur(L)   [normalisé par timestep]
```
Le flou gaussien 3D (spatial + temporel) crée une carte de "haze" autour de chaque éclair réel. Cela encode l'incertitude de localisation spatio-temporelle.

**Étape 2 — Importance weighting (neighborhood-aware) :**
```
P = (1 - L^blur) ∘ L̂  +  L^blur ∘ (1 - L̂)
```
P est une matrice de poids d'importance. Quand `L^blur` est élevé (voisinage d'un vrai éclair), la pénalité sur les **faux négatifs** dans ce voisinage est réduite, et réciproquement pour les faux positifs loin d'éclairs.

**Étape 3 — Loss finale :**
```
B = -[L ∘ log(L̂) + (1-L) ∘ log(1-L̂)]   # Binary Cross-Entropy pixel-wise
Loss_Hazy = (1/(h·N·N)) × (P · B)          # h = horizon, N×N = grille
Total_Loss = Loss_WBCE + Loss_Hazy
```

**Pourquoi neighborhood-aware ?** Les éclairs ont une incertitude spatio-temporelle inhérente. Prédire un éclair à 4 km du vrai emplacement ne devrait pas être pénalisé autant que prédire en zone totalement vide. Le flou gaussien encode la "zone d'influence" de chaque éclair, et la pondération P réduit la pénalité dans ce voisinage.

### Features d'entrée — radar + éclairs (pas éclairs seuls)

DeepLight utilise **trois sources obligatoires** :
- **Réflectivité radar NEXRAD** : valeurs en dBZ, plafonnées à [0, 65 dBZ]
- **Propriétés nuageuses (satellite GOES/ABI)** : hauteur du sommet nuageux (pieds), pression au sommet (hPa), épaisseur optique nuageuse
- **Éclairs historiques (GLM)** : occurrence de flash (binaire), fréquence de flash, énergie de flash

**Le modèle n'est PAS utilisable sans radar dans sa forme actuelle.**

### Type de problème : occurrence, pas cessation

DeepLight prédit **si un éclair va se produire** (probabilité d'occurrence par cellule de grille). Ce n'est pas un problème de cessation. La sortie L̂_t ∈ [0,1] est une probabilité d'occurrence.

### Données
- **Région** : Texas/Oklahoma centré sur Dallas (30.2°N–35.93°N, 93.52°W–100.3°W)
- **Grille** : 159×159 cellules à 4 km×4 km
- **Résolution temporelle** : observations horaires
- **Période** : Avril-Juillet 2021-2023
- **Split** : Train 2021-2022, Val Avr-Mai 2023, Test Juin-Juil 2023

### Ce qui est adaptable sans radar pour EasyOrage

| Composant DeepLight | Adaptabilité sans radar |
|---|---|
| MB-ConvLSTM (4 branches multi-échelle) | **Adaptable** : utiliser sur séries 1D temporelles d'éclairs (flash rate, ILI, amplitude) |
| Hazy Loss (neighborhood-aware) | **Directement adaptable** : remplacer la dimension spatiale par la dimension temporelle → tolérance sur la prédiction du "dernier éclair" |
| Fusion multi-branch + 1×1 conv | **Adaptable** : multi-branch sur différentes fenêtres temporelles (5 min, 10 min, 30 min) |
| Données radar | **Non applicable** : remplacer par ILI stats, IC/CG ratio, amplitude rolling |

**Adaptation Hazy Loss pour cessation** : La Hazy Loss est particulièrement pertinente pour notre problème. Le "dernier éclair" a une incertitude temporelle — prédire la cessation 3 min en avance ne devrait pas être pénalisé autant que 20 min. On peut appliquer un flou gaussien 1D sur l'axe temporel autour de la vraie cessation.

---

## PARTIE 2 : Survival Analysis pour thunderstorm cessation

### Conclusion principale : gap confirmé dans la littérature

Après recherche exhaustive avec plusieurs requêtes :
- "survival analysis thunderstorm duration prediction"
- "Cox hazard model storm lifetime lightning"
- "time-to-event thunderstorm cessation"
- "Weibull storm duration model"
- "thunderstorm cessation time-to-event survival analysis lightning warning"

**Aucun papier publié n'applique formellement du survival analysis (Cox PH, Weibull AFT, Kaplan-Meier) au problème de cessation d'orage ou de durée de session d'éclairs.** Ce gap est confirmé.

### Ce que la littérature fait à la place

**Wilhelm 2023 (QJRMS)** — "Statistical relevance of meteorological ambient conditions and cell attributes for nowcasting the life cycle of convective storms"
- **Méthode** : Analyse de corrélation statistique (Kendall τ) entre attributs de cellules convectives et variables ambiantes
- **Prédicteur principal de durée de vie** : aire horizontale initiale de la cellule (corrélation modérée)
- **Variables ambiantes** : vent moyen en moyenne troposphère et cisaillement vertical (meilleurs discriminants long/court cycle de vie)
- **Pas de survival analysis** : modèle purement descriptif/corrélationnel

**HESS 2025 — "Modelling convective cell life cycles with a copula-based approach"**
- **Méthode** : Vine copulas (D-vine 4D + C-vine 3D) pour modéliser les distributions jointes des propriétés du cycle de vie
- **Variables** : axe majeur/mineur peak, intensité max, durée (D_L), taux de croissance et de déclin
- **Distribution de durée** : Distribution Gamma (meilleure selon AIC)
- **Dissipation** : modélisée via taux de déclin (R_decay = ratio valeur peak / valeur finale), hypothèse linéaire
- **Pas de hazard function** : approche copule purement distributionnelle, sans modélisation conditionnelle de la survie
- **Features** : extraites de radar (27 731 cycles de vie de cellules)

### Apports du domaine adjacent (cyclones tropicaux)

Les modèles paramétrique de cyclones tropicaux (STORM, etc.) incluent un "decay model" mais :
- Ciblés sur cyclones (durée de jours/semaines vs minutes pour orages)
- Approche statistique paramétrique classique, sans survival analysis formel

### Opportunité scientifique pour EasyOrage

Le cadre survival analysis est **naturellement adapté** au problème :
- **Event** : dernier éclair dans le rayon 20 km de l'aéroport
- **Time** : temps restant avant cet event, conditionnel aux éclairs déjà observés
- **Covariables time-varying** : flash rate, ILI, IC/CG ratio, amplitude — toutes évoluent pendant la session
- **Censure** : sessions qui n'ont pas encore cessé (pas de problème ici avec données historiques, mais en temps réel oui)

**Framework recommandé** :
1. **Cox PH avec covariables time-varying** : `h(t|X(t)) = h0(t) × exp(β·X(t))`
   - X(t) = [flash_rate(t), ILI_mean(t), IC_CG_ratio(t), amplitude_trend(t), ...]
   - Avantage : interprétable, pas d'hypothèse sur la distribution de survie
2. **Weibull AFT** : `log(T) = μ + σ·ε + β·X`
   - Permet de prédire directement le temps restant
   - Utile pour communication "temps restant estimé : X minutes"
3. **DeepHit / DeepSurv** : survival analysis avec neural network
   - Capturent les non-linéarités
   - Peuvent utiliser des covariables time-varying complexes

**Métriques d'évaluation survival** :
- C-index (Harrell's concordance) : mesure la discrimination (analogue AUROC)
- Integrated Brier Score : calibration des probabilités de survie
- Expected lead time gain vs baseline 30 min

### Résultats expérimentaux (2026-03-11)

Reformulation survival : `duration = ILI_{i+1}` (temps jusqu'au prochain éclair), `event = 1` si pas
le dernier (observé), `event = 0` si dernier (censuré à 1800s = 30 min).

**Kaplan-Meier (baseline)** : P(T > 1800s) = 4.9% → 4.9% des flashes sont les derniers (cohérent).

| Modèle | C-index | AUC last flash | Features |
|--------|---------|----------------|----------|
| Cox PH (penalizer=0.1) | **0.7388** | 0.9080 | 18 lightning |
| XGBoost AFT (normal, σ=1.2) | **0.7414** | 0.8880 | 102 toutes |
| XGBoost classifieur | 0.7198 | **0.9205** | 102 toutes |

**Interprétation** :
- XGBoost AFT a le meilleur C-index (0.7414) : meilleur classement des durées de survie
- XGBoost classifieur a le meilleur AUC (0.9205) : optimisé directement pour l'objectif binaire
- Cox PH avec seulement 18 features = très compétitif (C-index 0.7388), interprétable

**Conclusion** : le classificateur XGBoost IS un modèle de survie implicite (il prédit P(T>1800s)).
La survival analysis formelle ne bat pas le classificateur mais offre :
1. Un C-index légèrement meilleur (0.7414 vs 0.7198)
2. La prédiction du temps restant estimé (pas seulement binaire)
3. Une interprétabilité des features via hazard ratios (Cox PH)

**Bug découvert (XGBoost AFT)** : les `label_lower_bound` / `label_upper_bound` doivent être en
**échelle originale (secondes)**, pas en log-scale. Passer `np.log(duration)` → tous les
gradients = 0 → preds = NaN. XGBoost applique le log en interne.

---

## PARTIE 3 : Leinonen 2022/2023 — Architecture et features DEM

### Deux papiers connexes du même auteur

**Papier A** : "Seamless lightning nowcasting with recurrent-convolutional deep learning" (arXiv:2203.10114, AIES 2022)
**Papier B** : "Thunderstorm nowcasting with deep learning: a multi-hazard data fusion model" (arXiv:2211.01001, GRL 2023)

Le papier B étend le papier A à la prédiction multi-aléas (foudre + grêle + précipitations).

### Architecture exacte (Papier A — détails les plus complets)

**Type** : Recurrent-convolutional avec cellules **ConvGRU** (Convolutional Gated Recurrent Units) dont les convolutions sont **remplacées par des blocs résiduels**.

**Structure encoder-forecaster** :
- **6 timesteps passés** (30 minutes d'historique à résolution 5 min)
- **12 timesteps futurs** prédits (60 minutes, résolution 5 min)
- **Deux branches de résolution** dans l'encodeur :
  - Branche haute résolution (1 km) : radar, foudre, DEM, satellite HRV
  - Branche basse résolution (4 km) : canaux SEVIRI (sauf HRV)

**Loss function** : Focal loss avec γ=2

### Features DEM — intégration exacte

Les données DEM proviennent de l'**ASTER Global DEM** et sont incorporées comme **3 canaux statiques** dans la branche haute résolution (1 km) :

| Feature DEM | Description | Normalisation |
|---|---|---|
| Élévation | Altitude en mètres | Mise à l'échelle μ=1 |
| Dérivée Est-Ouest | Gradient horizontal EO | Normalisé (μ≈0, σ≈1) |
| Dérivée Nord-Sud | Gradient horizontal NS | Normalisé (μ≈0, σ≈1) |

**Mécanisme d'intégration** : Les 3 canaux DEM statiques sont **concaténés directement** avec les données temporelles (radar, foudre) dans la branche d'entrée haute résolution. Ils passent par le même encodeur ConvGRU que les données dynamiques — il n'y a pas de branche séparée pour les features statiques.

**Implication pour EasyOrage** : Cette approche (concaténation de features statiques par aéroport avec séquences temporelles) est directement applicable. Nos features DEM statiques par aéroport (TRI, distance au relief, channeling index) peuvent être concaténées aux séquences de flash rate.

### Features foudre utilisées

| Feature | Description | Résolution |
|---|---|---|
| Densité d'éclairs | Nombre de flashes par cellule, par 5 min | 1 km |
| Densité pondérée par le courant | Flash density pondérée par le courant de pic | 1 km |
| Occurrence passée (cible de test) | 1 si éclair dans 8 km / 10 min | 1 km |

La target finale est : **pixel = 1 si éclair dans 8 km dans les 10 prochaines minutes**.

### Feature list complète (Papier A, Tables 3-4 de l'appendix)

| Source | Features |
|---|---|
| Foudre | Densité de flash (5 min), densité pondérée courant |
| Radar (MeteoSwiss) | RZC, CZC, LZC, EZC-20, EZC-45, HZC |
| Satellite SEVIRI | 12 canaux (visible + infrarouge) + HRV |
| Satellite NWCSAF | Phase nuageuse, T sommet, hauteur sommet, épaisseur optique |
| NWP (COSMO) | CAPE, CIN, HZEROCL, LCL, MCONV, OMEGA, SLI, type de sol, T-2M, T-SO |
| DEM (ASTER) | Élévation, dérivée EO, dérivée NS |
| Auxiliaire | Angle zénithal solaire |

**Total : ~40+ features** incluant données multi-source.

### Composante "cessation" dans le modèle Leinonen

**Il n'y a pas de composante cessation explicite.** Le modèle est un **nowcasting d'occurrence** (probabilité qu'un éclair se produise dans les 60 prochaines minutes).

Toutefois, le papier B (Leinonen 2023, GRL) mentionne que le modèle "reconnaît et prédit le mouvement, la croissance ET le déclin des cellules orageuses". Un exemple cas d'usage montre que "le modèle est capable de reconnaître une séquence d'entrée décroissante et de prédire correctement des probabilités décroissantes". C'est une **cessation implicite apprise** — le modèle apprend que les éclairs décroissent mais il ne prédit pas explicitement un "dernier éclair" ou un "temps avant cessation".

**Pour EasyOrage** : Cette distinction est cruciale. Nous avons besoin d'une **formulation explicite de cessation**, pas seulement d'occurrence décroissante. Un modèle d'occurrence peut donner P(éclair dans 5 min) → 0.1, mais ne peut pas dire "l'orage a 90% de chances de cesser dans les 15 prochaines minutes".

### Pertinence Leinonen pour notre projet (score mis à jour)

| Aspect | Utilisable ? | Comment |
|---|---|---|
| Architecture ConvGRU encoder-forecaster | Partiellement | Adapter en 1D temporel sur séquences d'éclairs |
| Intégration DEM statique (concaténation) | **Directement** | 3 gradients DEM → concaténer à nos séquences |
| Features foudre (densité, courant-pondéré) | **Directement** | Nos colonnes `amplitude`, `icloud` permettent de calculer exactement ces proxies |
| Focal loss | **Directement** | Adaptée aux événements rares (cessation = classe positive rare) |
| Approche occurrence vs cessation | **Non** | Reformuler comme survival analysis ou régression de durée |

---

# État de l'art — Lightning Nowcasting & Thunderstorm Lifecycle (2022-2025)

## Résumé exécutif

L'analyse de 20+ publications récentes (2023-2025) révèle une **accélération du deep learning** pour la prédiction d'éclairs et la nowcasting d'orages. Les approches dominantes passent des simples seuils radar à des **modèles génératifs (GANs, diffusion)** capables de prédictions jusqu'à **4 heures** avec résolution spatiale 4 km. Aucune étude majeure n'utilise Météorage directement, mais des opportunités existent pour l'intégration dans des modèles hybrides.

---

## Paper 1: Aerosol-Informed Lightning Nowcasting with Satellite Data (2023)

**Référence** : npj Climate and Atmospheric Science, 2023
**URL** : [Lightning nowcasting with aerosol-informed machine learning and satellite-enriched dataset](https://www.nature.com/articles/s41612-023-00451-x)

**Auteurs** : Équipe NOAA/NCAR
**Tâche** : Nowcasting probabiliste de la foudre horaire continue

### Méthodologie
- **Données d'entrée** :
  - Satellite GLM (Geostationary Lightning Mapper)
  - Données d'aérosols MERRA2
  - Variables atmosphériques ECMWF
- **Modèle** : LightGBM bien optimisé
- **Région** : CONUS (États-Unis continentaux), saison estivale
- **Résolution** : Nowcasts spatialement continus

### Performance
- Génère des nowcasts d'éclairs horaires sur grilles CONUS
- Capture les mécanismes de foudre via inclusion des features d'aérosols
- Enrichissement satellite GLM améliore la qualité

### Code disponible
- Approche LightGBM publiée ; dépôt code non mentionné directement

### Pertinence pour EasyOrage
**Score : 4/5** — Approche satellite + ML, mais pas centré sur le problème de "cessation". Utilise GLM plutôt que Météorage.

---

## Paper 2: Hybrid AI-Enhanced Lightning Flash Prediction (2024)

**Référence** : Nature Communications, 2024
**URL** : [Hybrid AI-enhanced lightning flash prediction in the medium-range forecast horizon](https://www.nature.com/articles/s41467-024-44697-2)

**Tâche** : Prédiction d'éclairs 2 jours à l'avance

### Méthodologie
- **Données d'entrée** : Prévisions ECMWF (European Centre for Medium-range Weather Forecasts) à 2 jours
- **Modèle** : IA hybride trouvant le mapping optimal de features météorologiques en occurrences d'éclairs
- **Approche** : Mapping intelligent des prévisions ECMWF → occurrence éclair

### Performance
- **Capacité de prédiction significativement plus élevée** que l'algorithme entièrement déterministe du modèle ECMWF
- Démontre que l'IA peut optimiser les paramétrisations physiques pour les prévisions d'éclair

### Pertinence pour EasyOrage
**Score : 3/5** — Excellente pour prédictions long terme (2 jours), mais nous travaillons sur nowcasting court terme (minutes à heures).

---

## Paper 3: Thunderstorm Nowcasting with Deep Learning - Multi-Hazard Model (2023)

**Référence** : Geophysical Research Letters, 2023
**Auteurs** : J. Leinonen et al.
**URL** : [Thunderstorm Nowcasting With Deep Learning: A Multi‐Hazard Data Fusion Model](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022GL101626)
**arXiv** : [arXiv:2211.01001](https://arxiv.org/abs/2211.01001)

**Tâche** : Nowcasting probabiliste multi-aléas (foudre, grêle, précipitations)

### Méthodologie
- **Données** :
  - Radar météo (réflectivité, polarimétrie)
  - Détection d'éclair (détails non précisés)
  - Imagerie satellite visible/infrarouge
  - Prévisions NWP
  - Modèles numériques de terrain (DEM)
- **Modèle** : Architecture unifiée capable de prédire 3 aléas simultanément
- **Résolution** : Grille 1 km, résolution temporelle 5 min
- **Lead times** : Jusqu'à 60 minutes

### Performance
- Prédictions probabilistes en 2D des 3 aléas
- Explainability AI révèle que **le radar est la source de données la plus importante** (>70% contribution)
- Modèle capable de détecter et prédire le mouvement d'orages
- Prédit l'évolution des orages (intensification vs affaiblissement)

### Code & pré-entraînement
- **Pré-modèles disponibles** : Zenodo records/7157986

### Pertinence pour EasyOrage
**Score : 4/5** — Excellent pour "intégration multi-source". Nous pourrions adapter pour capteurs éclair uniquement (sans radar). Démontre le gain d'approches fusionnées.

---

## Paper 4: NowcastNet - Deep Learning for Extreme Precipitation (2023)

**Référence** : Nature, 2023
**URL** : [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4)

**Tâche** : Nowcasting de précipitations extrêmes (grille 2048×2048 km)

### Méthodologie
- **Architecture** : Réseau hybride physique-données
  - Réseau d'évolution déterministe (advection) basé sur schémas physiques
  - Réseau génératif stochastique (apprentissage données)
  - Conditionnement bidirectionnel pour capturer la chaotique
- **Données** : Observations radar USA + Chine
- **Régime** : Précipitations extrêmes

### Performance
- Nowcasts multischelle avec motifs 1-10 km reproduits fidèlement
- Lead times jusqu'à **3 heures** sur grilles très grandes
- Prévisibilité physiquement plausible (conserve masse)
- **Ensemble nowcasting** capable (via stochasticité)

### Code disponible
- Architecture publiquement disponible ; implémentations open source existent

### Pertinence pour EasyOrage
**Score : 4/5** — Approche état-de-l'art pour nowcasting convectif. Concept hybride physique-ML directement applicable aux prédictions de cessation (modéliser l'évolution du cycle de vie).

---

## Paper 5: Deep Diffusion Model for Satellite Thunderstorm (DDMS) (2024)

**Référence** : PNAS, 2024
**URL** : [Four-hour thunderstorm nowcasting using a deep diffusion model for satellite data](https://www.pnas.org/doi/10.1073/pnas.2517520122)
**arXiv** : [arXiv:2404.10512](https://arxiv.org/abs/2404.10512)

**Auteurs** : Kuai Dai, Xutao Li, Junying Fang, et al.
**Tâche** : Nowcasting d'orages convectifs jusqu'à **4 heures**

### Méthodologie
- **Modèle** : Diffusion profonde (analogous à DDPM/imagen)
- **Données** : Satellite FengYun-4A (température de brillance)
- **Approche** : Modèles de diffusion paramétrés pour capturer l'évolution spatio-temporelle compliquée des nuages convectifs
- **Résolution** : 4 km spatiale, 15 min temporelle

### Performance
- **Coverage** : ~20 millions km² (très large)
- **Accuracy** : Supérieur aux approches persistance et PySTEPS
- **Comparaison** : Dépasse PredRNN-v2 et NowcastNet en évaluation quantitative multischelle
- Capture fidèlement la croissance et dissipation convective

### Code disponible
- **Oui** : [GitHub - Applied-IAS/DDMS](https://github.com/Applied-IAS/DDMS)
- Poids pré-entraînés fournis pour satellite nowcasting + détection convection

### Pertinence pour EasyOrage
**Score : 4/5** — **Très pertinent** : diffusion modèle pour orages sur données satellites (pas radar). Lead times longs (4h) = signal du cycle de vie visible. Approche transférable aux données Météorage brutes.

---

## Paper 6: FlashBench — Analyse approfondie (arXiv:2305.10064)

**Référence** : Singh et al., arXiv:2305.10064, Mai 2023
**URL** : [FlashBench: A lightning nowcasting framework based on hybrid deep learning and physics-based dynamical models](https://arxiv.org/abs/2305.10064)

**Auteurs** : Manmeet Singh, S. Vaisakh, Dipjyoti Mudiar, Debojyoti Chakraborty, V. Gopalakrishnan, Bhupendra Singh, Shikha Singh, Rakesh Ghosh, Rajib Chattopadhyay, Bipin Kumar, S. Pawar, S. Rao (12 auteurs, IITM Pune, Inde)
**Tâche** : Nowcasting d'éclair en temps quasi-réel

---

### 1. Architecture du modèle — Partie physique

**Modèle de base** : WRF (Weather Research and Forecasting) — modèle non-hydrostatique, 3D compressible, terrain-following, cloud-resolving.

**Configuration WRF dans FlashBench** :
- 4 domaines imbriqués (nested) : d01 (27 km), d02 (9 km), d03 (3 km), d04 (1 km)
- Domaine intérieur d04 centré sur le Maharashtra (Inde de l'Ouest)
- Paramétrisations de foudre WRF testées : PR92 (Price & Rind 1992), WRF-ELEC, et plusieurs autres schémas

**Schéma de paramétrisations lightning WRF** :
- PR92 est la meilleure paramétrisation sur cette région selon les auteurs (validée préalablement)
- WRF-ELEC ne couvre pas toutes les zones d'éclairs observées dans certains cas
- Ces schémas diagnostiquent le nombre de flashes à partir de variables dynamiques (updraft, contenu en glace, CAPE)

**Sortie physique** : Champs spatiaux simulés de densité d'éclairs (flash density, unit/km²/heure) pour les 6 prochaines heures.

---

### 2. Architecture du modèle — Partie ML

**Inspiration directe** : LightNet (Geng et al., KDD 2019) — architecture duale avec deux encodeurs, un module de fusion, et un décodeur.

**Innovation FlashBench vs LightNet** : LightNet utilise uniquement les champs simulés WRF comme entrée ML. FlashBench ajoute une deuxième source : **les champs observés avec leur développement temporel** (time-dependent development). Cela corrige les erreurs systématiques des simulations WRF en les "calibrant" avec les observations récentes.

**Architecture probable** (inférée de la description et de la parenté LightNet) :
```
[Encodeur WRF]  ──────────────────────────────────────────┐
  ConvLSTM sur champs WRF simulés (6h de prévision NWP)   │
                                                           ├→ [Module de Fusion] → [Décodeur spatiotemporel] → Prévision
[Encodeur Observations]  ─────────────────────────────────┘
  ConvLSTM sur champs observés récents (TRMM LIS / réseau sol)
  Capture la dynamique temporelle des observations
```

**Détail critique** : L'intégration des observations avec leur "time-dependent development" signifie que le modèle ML ne reçoit pas juste un instantané observé, mais une séquence temporelle (trend, trajectoire) qui permet de corriger la dérive WRF.

**Déploiement** : Système cloud temps réel sur Google Earth Engine pour l'Inde de l'Ouest.

---

### 3. Features d'entrée

**Source 1 — Champs WRF simulés** (passent dans l'encodeur WRF) :
- Densité de foudre simulée par les schémas de paramétrisations
- Probablement : CAPE, contenu en eau glacée, updraft, réflectivité simulée (variables WRF standard alimentant PR92)

**Source 2 — Observations récentes avec développement temporel** (passent dans l'encodeur observations) :
- Données de foudre observées : **TRMM LIS** (satellite Tropical Rainfall Measuring Mission Lightning Imaging Sensor)
- Développement temporel = séquence de plusieurs pas de temps passés (tendance, évolution)
- Pas de radar, pas de données sol explicitement mentionnées

**Note importante** : FlashBench n'utilise PAS de données Météorage ni de réseau sol européen. Il utilise TRMM LIS (satellite) pour les observations d'éclairs.

---

### 4. Target — Que prédit le modèle ?

**Variable prédite** : Présence/intensité de foudre (flash density) dans le domaine de prévision.
**Type de problème** : Nowcasting d'**occurrence** d'éclairs — PAS un problème de cessation.
**Lead time** : Prévision jusqu'à 6 heures (analogue à LightNet qui atteint 6h).
**Résolution spatiale** : Grille 1 km (domaine d04 WRF).
**Résolution temporelle** : Non précisée explicitement, probablement 30 min ou 1h (cohérent avec WRF).

---

### 5. Données d'entraînement

| Aspect | Détail |
|---|---|
| **Source observations éclairs** | TRMM LIS (satellite) — capteur optique, détecte les éclairs par émission lumineuse |
| **Source NWP/physique** | WRF avec schémas de paramétrisations (PR92, WRF-ELEC) |
| **Zone géographique** | Inde de l'Ouest / Maharashtra (domaine tropical) |
| **Période** | Non spécifiée dans les sources accessibles (étude de cas unique sur un événement orageux) |
| **Nature des données** | Cas d'étude sur une tempête (pas de dataset statistiquement large comme Météorage 10 ans) |

**Limitation critique** : L'évaluation repose sur des **cas d'étude** et non sur un test statistique large. Cela fragilise la généralisation.

---

### 6. Performances détaillées

| Métrique | FlashBench (ML hybride) | Modèle dynamique WRF seul | Amélioration |
|---|---|---|---|
| **POD** | **0.73134** | Plus faible | Meilleur |
| **FAR** | **0.25757** | Plus élevé | Meilleur |
| **ETS** | **0.55907** | Plus faible | Meilleur |

**Résultats sur plusieurs intervalles de temps** : FlashBench surpasse WRF à tous les intervalles temporels testés (greater POD and ETS, lower FAR).

**Interprétation** : Un ETS de 0.56 est excellent pour un problème de prédiction d'éclairs (un ETS > 0.3 est considéré comme "utile", > 0.5 comme "bon"). La FAR de 0.26 signifie 26% de fausses alarmes — acceptable pour la météorologie opérationnelle.

**Comparaison avec l'état de l'art** :
- Seamless lightning nowcasting (Leinonen 2022, MeteoSwiss) : ETS ~0.45, mais 60 min seulement
- FlashNet (Italy, Nat. Comm. 2024) : recall 86-88% (0-24h), mais problème différent (binaire occurrence par maille)
- **FlashBench est compétitif pour des lead times de 6h, domaine tropical**

---

### 7. Code disponible ?

**Non.** Aucun repository GitHub trouvé pour FlashBench. La page GitHub de l'auteur principal (manmeet3591.github.io) n'indique pas de code public pour ce projet. Le système de déploiement (Google Earth Engine) est opérationnel mais non open-source.

LightNet (le modèle parent dont FlashBench s'inspire) a un repository Keras : https://github.com/gyla1993/LightNet

---

### 8. Ce qui est transférable à notre contexte (données Météorage seules, pas de NWP, aéroports européens)

| Composant FlashBench | Transférable ? | Adaptation pour EasyOrage |
|---|---|---|
| **Encodeur WRF (partie physique)** | **NON** — nécessite WRF opérationnel | Remplacer par modèle physique analytique : fit exponentiel flash rate + ILI rolling |
| **Encodeur observations temporelles** | **OUI — directement** | Encoder séquences Météorage (flash rate 5 min, ILI, IC/CG, amplitude) sur fenêtre 30-60 min |
| **Module de fusion dual** | **OUI — adapter** | Fusionner [features physiques analytiques] + [features ML sur séquences Météorage] |
| **Idée de "calibration observations sur physique"** | **OUI — clé** | Utiliser les observations récentes pour corriger/calibrer le modèle de déclin physique |
| **Déploiement temps réel** | **NON** — GEE spécifique | Déploiement Python standard suffisant |
| **Prédiction occurrence vs cessation** | **NON** — nous voulons cessation | Reformuler target : P(dernier éclair dans les T prochaines minutes) |
| **TRMM LIS comme source** | **NON** — nous avons Météorage | Météorage est **meilleur** : résolution 250 m, 10 ans, CG + IC |

**Adaptation recommandée** :
```
[Composant physique analytique]  ───────────────────────────────────┐
  - Fit exponentiel du flash rate (τ = constante de déclin)          │
  - ILI rolling (inter-lightning interval, médiane 10-30 min)        ├→ [Fusion] → P(cessation dans T min)
  - Ratio IC/CG glissant (décline vers cessation)                    │
                                                                      │
[Composant ML sur séquences Météorage]  ─────────────────────────────┘
  - XGBoost ou ConvGRU sur séquence temporelle 60 min
  - Features : amplitude stats, dispersion spatiale, dérive centroïde
  - Features DEM statiques par aéroport
```

---

## Paper 6b : Papiers complémentaires sur la cessation — nouvellement identifiés

### Lightning Cessation Guidance — Polarimetric Radar + LMA (2019, 2022)

**Référence 1** : Shafer & Fuelberg 2019 — "Using Radar-Derived Parameters to Develop Probabilistic Guidance for Lightning Cessation within Isolated Convection near Cape Canaveral"
**URL** : https://journals.ametsoc.org/view/journals/wefo/34/3/waf-d-18-0144_1.xml

**Approche** : Généralized Linear Model (GLM) — **régression logistique** — à chaque éclair : "est-ce le dernier flash de la tempête ?"

**Features** :
- Réflectivité maximale (ZH) à 0°C
- Présence de grésil (graupel) aux niveaux -5°C, -10°C, -15°C, -20°C
- Réflectivité composite (maximum de colonne)
- Données : radar dual-polarisation (1-min intervals)

**Résultat** : ~99% de précision probabiliste pour identifier le dernier éclair — **mais avec données radar**.

**Référence 2** : Drugan & Preston 2022 — "Lightning Cessation Guidance Using Polarimetric Radar Data and LMA in the Washington, D.C. Area" (Atmosphere, MDPI)

**Meilleurs algorithmes** :
1. ZH ≥ 40 dBZ à -5°C — si seuil non atteint → attendre 15 min → all-clear
2. ZH ≥ 35 dBZ à -10°C — même logique
3. Présence de graupel à -15°C — même logique

**Pertinence pour EasyOrage** : Ces approches requirent des données radar que nous n'avons pas. Mais elles confirment la **physique sous-jacente** : le graupel et la réflectivité aux isothermes sont des proxies de la charge électrique dans le nuage. Sans radar, nos proxies équivalents sont :
- ILI (inter-lightning interval) croissant → charge s'épuise
- Ratio IC/CG décroissant → stratification verticale s'effondre
- Amplitude décroissante → courants de décharge plus faibles

---

### Lightning Nowcasting Using Solely Lightning Data (MDPI Atmosphere 2023)

**Référence** : Mansouri, Mostajabi, Tong, Rubinstein & Rachidi — Atmosphere 14(12):1713, Nov 2023
**URL** : https://www.mdpi.com/2073-4433/14/12/1713

**Approche** : Réseau résiduel U-Net entraîné **uniquement sur données d'éclairs satellites** (GLM/TRMM LIS). Prédit l'occurrence future d'éclairs sans NWP, sans radar.

**Architecture** : Residual U-Net — exploite l'inductive bias que les phénomènes atmosphériques évoluent de façon continue (skip connections = évolution de l'état précédent).

**Input** : Séquences temporelles d'images d'éclairs GLM uniquement.

**Target** : Occurrence d'éclairs dans les 15-60 prochaines minutes.

**Avantage clé** : "Pas de dépendance aux NWP, qui sont intrinsèquement lents due à leur nature séquentielle" → compatible temps réel.

**Pertinence pour EasyOrage** : **Score 5/5** — C'est le papier le plus proche de notre cas d'usage :
- Données d'éclairs uniquement (notre contrainte)
- Architecture transférable aux données Météorage (remplacer pixels GLM par grilles de densité d'éclairs Météorage)
- Pas de NWP, pas de radar
- Mais : prédit **occurrence** pas **cessation** — reformulation nécessaire

---

### Seamless Lightning Nowcasting — Résultats ablation (MeteoSwiss 2022)

Confirmation importante (déjà en DISCOVERIES) : les données EUCLID/Météorage sont **directement utilisées** dans ce papier MeteoSwiss. Résultat ablation : "l'omission des données NWP peut être bien compensée par l'information dans les données observationnelles sur la période de nowcast". Cela valide notre approche sans NWP.

---

### FlashNet (Nature Communications 2024) — Réseau italienéclairs

**URL** : https://pmc.ncbi.nlm.nih.gov/articles/PMC10853497/

**Données éclairs** : Réseau LAMPINET (Vaisala IMPACT ESP, 15 capteurs, Italie) — 90% DE pour courants >50 kA, localisation 500 m. Comparable à Météorage pour la France.

**Métriques clés** :
- Recall 86-88% (0-24h), 76-82% (24-48h) vs ECMWF HRES 72-74%
- AUC PR 0.93 sur test set 2021
- F1 score 0.36-0.38 (vs HRES 0.15-0.19)

**Limitations** : Prédit occurrence binaire par maille 10km toutes les 3h — pas de nowcasting haute résolution temporelle, pas de cessation.

**Code disponible** : Oui — TensorFlow, pré-entraîné disponible (~23 GB dataset sample).

---

## Paper 7: DeepLight - Lightning Prediction with Uncertainty (2025)

**Référence** : arXiv, Août 2025
**URL** : [Lightning Prediction under Uncertainty: DeepLight with Hazy Loss](https://arxiv.org/abs/2508.07428)

**Auteurs** : Md Sultanul Arifin et al.
**Tâche** : Prédiction d'occurrence d'éclair avec modélisation d'incertitude

### Méthodologie
- **Modèle** : Architecture multi-branch ConvLSTM variant (MB-ConvLSTM)
- **Données** : Données d'observation météorologique réelles & diversifiées
  - Réflectivité radar
  - Propriétés nuageuses
  - Occurrences d'éclair historiques
- **Loss function** : "Hazy Loss" = loss neighborhood-aware custom
- **Key Feature** : Ne dépend **pas de NWP** ; basé sur observations brutes

### Performance
- **Amélioration ETS** :
  - +30% à 1 heure
  - +18-22% à 3 heures
  - +8-13% à 6 heures
- Surpasse état-de-l'art sur horizons variés

### Code disponible
- Publié sur arXiv ; implémentation likely disponible via auteurs

### Pertinence pour EasyOrage
**Score : 5/5** — **Critique** :
- Architecture ConvLSTM multibranche = approche adaptable aux données temporelles d'éclair
- Loss asymétrique (Hazy) capturable comme coût d'alerte manquée vs fausse alerte
- Pas de dépendance NWP = compatible données Météorage brutes
- Horizons de 1-6h couvrent notre gamme (5 min à 60 min) → transfert learning possible

---

## Paper 8: Four-hour Thunderstorm Nowcasting + Probabilistic Lightning (2024-2025)

### Sous-étude A: Deep Learning for Short-Range Lightning Risk (2025)

**Tâche** : Probabilistic forecasting très court terme (5-60 min)

**Méthodologie**
- Méthodologie : Deep learning dédié à l'aviationSafety (5-60 min lead)
- **Performance**
  - F1 score : 0.65 à 5 min → 0.5 à 30 min
  - ECE (Expected Calibration Error) : <10%
  - Modèle probabiliste bien-calibré

### Pertinence
**Score : 5/5** — Horizon exact pour aéroports. Métriques F1 + calibration = critérique pour alertes aéroport.

---

## Paper 9: DGMR - Deep Generative Model of Radar (DeepMind 2021)

**Référence** : Nature, 2021
**URL** : [Skilful precipitation nowcasting using deep generative models of radar](https://www.nature.com/articles/s41586-021-03854-z)

**Auteurs** : DeepMind & UK Met Office
**Tâche** : Nowcasting génératif de précipitations (lead 2 heures)

### Méthodologie
- **Architecture** : Conditional GAN
  - Générateur : 4 frames radar → 18 frames futures
  - 2 Discriminateurs : 1 spatiale (frames), 1 temporelle (séquences)
- **Données** : Radar à haute résolution (UK/USA)

### Performance
- **Évaluation expert** (58 météorologistes) : DGMR 1er rang en accuracy/usefulness 89% des cas
- Équilibre intensité/étendue mieux que PySTEPS ou UNet déterministe
- Génération full-resolution <2 sec/frame (V100 GPU)

### Code & implémentations
- **Oui** : [GitHub - openclimatefix/skillful_nowcasting](https://github.com/openclimatefix/skillful_nowcasting)
- Implémentations PyTorch populaires

### Pertinence pour EasyOrage
**Score : 3/5** — Landmark paper pour nowcasting génératif, mais radar-centric. Concepts GAN/diffusion applicables si nous avions entrées multi-modales.

---

## Paper 10: Transformer-Based Nowcasting with Attention (2024)

### 10A: Diffusion Transformer with Causal Attention (2024)

**Tâche** : Nowcasting de précipitations extrêmes

**Méthodologie**
- **Architecture** : Diffusion + Transformer + Causal Attention
- **Innovation** : Attention causale capture dépendances long-range tout en respectant causalité temporelle
- **Performance**
  - **CSI amélioré** : +15% (heavy precip) / +8% (modéré)
  - État-de-l'art atteint

**Pertinence** : **Score 3/5** — Généraliste sur précipitations. Architecture attention utile pour feature engineering temporel sur séquences d'éclairs.

### 10B: Spatiotemporal Feature Fusion Transformer (2024)

**URL** : [Spatiotemporal Feature Fusion Transformer for Precipitation Nowcasting](https://www.mdpi.com/2072-4292/16/14/2685)

**Méthodologie**
- Encoder/forecaster Transformers avec modules attention temporels indépendants
- Global feedforward spatial-temporal
- Architecture multi-scale

**Pertinence** : **Score 3/5** — Multi-scale feature fusion idée applicable.

---

## Synthèse comparative des architectures

| Architecture | Données requises | Lead time max | POD/CSI attendu | Interprétabilité | GPU requis | Recommandé pour |
|---|---|---|---|---|---|---|
| **Seuils radar** (baseline) | Radar | 15-30 min | 0.96/- | ★★★★★ | Non | Baseline opérationnelle |
| **LightGBM** | Satel + features | 1-2h | 0.85-0.92/- | ★★★★ | Non | Features méticuleux |
| **XGBoost** | Mixte | 1-2h | 0.88-0.92/0.70 | ★★★★ | Non | Production réaliste |
| **ConvLSTM** | Spatio-temporel | 1h | 0.84-0.96 | ★★ | Oui (petit) | Séries temporelles spatiales |
| **NowcastNet** | Radar / Satellite | 3h | 0.85+/0.75+ | ★★★ | Oui | Convection long lead |
| **DGMR (GAN)** | Radar | 2h | 0.80-0.95/0.70 | ★★ | Oui | Ensemble nowcasting |
| **DDMS (Diffusion)** | Satellite | 4h | 0.85+/0.75+ | ★★★ | Oui | Très long lead (4h) |
| **DeepLight** | Radar + obs | 6h | 0.90+/0.75+ | ★★ | Oui | Horizons variés |
| **Cox PH** | Temporel | Continu | C-index 0.80+ | ★★★★★ | Non | Survival analysis → **GAP LITTÉRATURE** |
| **TFT** (Fusion Temporelle) | Univarié/multi | 1h | 0.82-0.88/- | ★★★ | Oui (petit) | Attention interprétable |

---

## 🔴 Gap Littérature Identifié : Méthodologies pour données Météorage brutes

**Observation clé** : Les ~20 papers analysés se divisent en :
1. **Radar-centrique** (60%) : DGMR, NowcastNet, ConvLSTM standards → Non applicables directement
2. **Satellite-centric** (20%) : DDMS, GLM nowcasting → Transfert possible (données 1D temporelles)
3. **Hybrid ML+Physics** (15%) : FlashBench, NowcastNet-hybrid → **Notre créneau idéal**
4. **Field mill + radar** (5%) : AFIT CRNN → Données électrique uniquement (pas radar)

**Aucune étude majeure n'aborde**
- ✅ Données **Météorage seules** (réseau de capteurs multi-sites distribués)
- ✅ **Cessation forecasting** spécifiquement (c.f. Stano 2010, mais pré-DL)
- ✅ **Survival analysis** formelle (Cox PH, Weibull) pour time-to-event

**Opportunité scientifique** : Un modèle hybride physique-ML sur données Météorage pour cessation = **Publication à fort impact potentiel**.

---

## Datasets & Benchmarks Mentionnés

| Dataset | Couverture | Résolution | Taille | Accès | Note |
|---|---|---|---|---|---|
| **GLM** (NOAA) | CONUS | 10 km grid, real-time | ~1M flashes/jour | Public | Geostationary Lightning Mapper |
| **WWLLN** | Global | ~10 km | ~40k flashes/jour | Public | World Wide Lightning Location Network |
| **Météorage** | France/Benelux | ~250 m localisation | ~10M flashes/an | Privé | **NOS DONNÉES** |
| **FengYun-4A** | Asie-Pacifique | 4 km satellite | Continu | Public | Satellite chinois (données DDMS) |
| **NEXRAD** | USA | 1 km radar | Continu | Public | Radar à réflectivité doppler |
| **UK Met Office radar** | UK | 2 km | Continu | Restreint | Données DGMR |
| **ERA5 soundings** | Global | ~30 km reanalyse | Continu | Public | ECMWF réanalyse |

---

## Recommandations pour EasyOrage

### Court terme (Phase 1-2, 2-3 semaines)

1. **Baseline hybrid physique-ML** inspirée de **FlashBench**
   - Composant physique : fit exponentiel flash-rate + ILI + IC/CG ratio trends
   - Composant ML : XGBoost/LightGBM sur features engineered
   - Cible : POD 0.85-0.92, FAR 0.10-0.15, +15-20 min lead vs 30 min baseline

2. **Feature engineering** guidé par Leinonen 2023 (multi-hazard fusion)
   - Même si pas de radar, inclure proxy field-mill-like : ILI stats, flash rate acceleration
   - Features orographiques (DEM/TRI) par aéroport

3. **Stratification par aéroport** (implication Biarritz/Pise = dynamiques très différentes)

### Moyen terme (Semaines 4-8)

4. **ConvLSTM/DeepLight-inspired** pour capturer trajectoires temporelles
   - Adaptation de MB-ConvLSTM pour séquences 1D d'éclairs multisites
   - Loss asymétrique (alertes manquées >> fausses alertes)

5. **Ensemble probabiliste** (inspiré DGMR/DDMS) pour calibration
   - Prédictions d'intervalle vs point estimates

### Long terme (Optionnel, post-DataBattle)

6. **Cox Proportional Hazards** ou **Weibull AFT** pour formulation survival analysis
   - **Uncharted territory** en nowcasting d'éclair → publication potentielle
   - Interprétabilité physique maximale

---

## Références complètes (2023-2025)

1. [Lightning nowcasting with aerosol-informed machine learning and satellite-enriched dataset](https://www.nature.com/articles/s41612-023-00451-x) — npj Climate, 2023
2. [Hybrid AI-enhanced lightning flash prediction in the medium-range forecast horizon](https://www.nature.com/articles/s41467-024-44697-2) — Nature Comm., 2024
3. [Thunderstorm Nowcasting With Deep Learning: A Multi‐Hazard Data Fusion Model](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022GL101626) — GRL, 2023 ([arXiv](https://arxiv.org/abs/2211.01001))
4. [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) — Nature, 2023
5. [Four-hour thunderstorm nowcasting using a deep diffusion model for satellite data](https://www.pnas.org/doi/10.1073/pnas.2517520122) — PNAS, 2024 ([arXiv](https://arxiv.org/abs/2404.10512), [Code](https://github.com/Applied-IAS/DDMS))
6. [FlashBench: A lightning nowcasting framework](https://arxiv.org/abs/2305.10064) — arXiv, 2023
7. [Lightning Prediction under Uncertainty: DeepLight with Hazy Loss](https://arxiv.org/abs/2508.07428) — arXiv, 2025
8. [Skilful precipitation nowcasting using deep generative models of radar](https://www.nature.com/articles/s41586-021-03854-z) — Nature, 2021 ([Code](https://github.com/openclimatefix/skillful_nowcasting))
9. [Precipitation Nowcasting Using Diffusion Transformer with Causal Attention](https://arxiv.org/abs/2410.13314) — arXiv, 2024
10. [Spatiotemporal Feature Fusion Transformer for Precipitation Nowcasting](https://www.mdpi.com/2072-4292/16/14/2685) — Remote Sensing, 2024

---

---

# Session de recherche approfondie — 2026-03-10

## Résumé exécutif des 4 agents de recherche

### Agent 1 — Papiers fondateurs (cessation classique)

**Stano, Fuelberg & Roeder 2010** (JGR 115, D09205)
- Dataset : 116 orages, KSC/CCAFS Floride, réseau LDAR total lightning (IC+CG)
- 5 schémas testés — seule la **Méthode des Percentiles (MIFI)** passe les critères sécurité
- **Distribution empirique du MIFI** (Maximum Inter-Flash Interval) :
  - P50 = 4.2 min | P75 = 7.5 min | P95 = 15 min | P99.5 = 25 min
- **Feature #1 sans radar** : flash interval + maximum interval sur la vie du storm
- Fit log-linéaire des ILI successifs : R² = 75 %
- Opérationnel à KSC depuis l'été 2008 → +5 à +10 min de gain vs règle 30 min
- Applicable directement à Météorage — 100% lightning-only

**Drugan & Preston 2022** (Atmosphere MDPI, 13(7):1111) — Dégradation hors Floride
- Washington D.C. (environnement continental) : mêmes seuils physiques mais délai 15 min vs 10 min → **dépendance géographique forte**, justifie nos modèles par aéroport
- Meilleurs algos : ZH ≥ 40 dBZ à -5°C (15 min), grésil à -15°C (15 min) — tout radar

**Crum et al. 2019** (WAF 34(3), DOI:10.1175/WAF-D-18-0144)
- Même papier que "Patton & Fuelberg" — 184 orages isolés, KSC
- GLM probabiliste par flash : "est-ce le dernier ?" → ~99% de précision avec radar
- Meilleures features : réflectivité **composite** et à **0°C** (pas -10°C comme attendu)

### Agent 2 — FlashBench et papiers similaires

**Papier le plus proche de notre cas : Mansouri et al. 2023** (MDPI Atmosphere 14(12):1713)
- Lightning nowcasting avec **données d'éclairs satellites uniquement** (GLM/TRMM LIS)
- Architecture : Residual U-Net sur séquences temporelles d'images d'éclairs
- Skip connections = encode l'évolution continue de l'état précédent
- Directement transférable : remplacer pixels GLM par grilles de densité Météorage

**Leinonen 2022 ablation** : "omission des données NWP bien compensée par les données observationnelles"
→ Valide notre approche sans NWP / sans radar

**Lightning nowcasting VLF-only (ScienceDirect 2025)** :
- Réseau VLF/LLN uniquement (comparable Météorage) + GMM + ME-RNN
- Preuve que les approches sol-only sont viables

### Agent 3 — DeepLight, Survival Analysis, Leinonen

**DeepLight MB-ConvLSTM** — adaptation 1D recommandée :
- 4 branches kernels 3×3, 5×5, 7×7, 11×11 → adapter en 1D : 4 fenêtres 5/10/15/30 min
- Hazy Loss en 1D : flou gaussien temporel autour du "dernier éclair" → tolérance aux prédictions légèrement en avance/retard

**Survival Analysis — gap confirmé** :
- Aucun papier n'applique Cox PH / Weibull à la cessation d'orage
- HESS 2025 : la durée des alertes suit une **distribution Gamma** → prior pour Weibull AFT
- Wilhelm 2023 : meilleur prédicteur de durée = **aire initiale** + vent moyen troposphérique

**Leinonen DEM** — 3 canaux statiques (élévation, gradient E-O, gradient N-S) concaténés directement avec les données temporelles → pas d'encodeur séparé nécessaire

### Agent 4 — Features lightning-only (état de l'art complet)

**Lightning Jump Algorithm (Schultz 2009)** — appliqué à la détection du déclin :
- DFRDT = FR_2min(t) - FR_2min(t-1) (dérivée du flash rate)
- sigma_level = DFRDT / std(DFRDT[-5:]) → sigma < -1 soutenu = dissipation
- Flash rate peut chuter de 80% dans les 8 minutes précédant la cessation
- Décroissance du FR précède la cessation de **5–15 min** dans 80% des cas

**Spatial dispersion (Bruning & MacGorman 2013)** :
- Corrélation anti-corrélation flash size / flash rate : r = -0.87
- En dissipation : flashes grands et rares ; en phase active : petits et fréquents
- Features : convex hull area, flash density (flashes/km²), centroid speed

**Polarité et amplitude (Yang et al. 2018)** :
- CG+ fraction : 3.52% (phase active) → 21.25% (dissipation) — signal fort
- Amplitude std : augmente en dissipation (mélange CG- convectif + CG+ enclume)
- Seuil opérationnel : pos_CG_fraction > 0.2 + FR < 5 flashes/5min → dissipation

---

## Features v3 (nouvelles, ajoutées à compute_features.py)

| Feature | Justification | Source |
|---|---|---|
| `rolling_ili_max_5/10` | Pause max = meilleur prédicteur (MIFI) | Stano 2010 |
| `ili_log` | Log(ILI+1) linéarise la relation exponentielle | Stano 2010 |
| `flash_rate_5/10` | Flash rate local (déclin récent vs global) | Schultz 2009 |
| `flash_rate_ratio` | FR_local / FR_global < 1 = ralentissement | Stano 2010 |
| `fr_vs_max_ratio` | FR_actuel / FR_peak cumulatif (décroissance depuis pic) | Stano 2010 |
| `positive_cg_frac` | Fraction CG+ augmente en dissipation | Yang 2018 |
| `high_amp_frac` | Fraction > 50 kA proxy CG+ enclume | Yang 2018 |
| `azimuth_spread_10` | Dispersion angulaire des éclairs | Bruning 2013 |
| `dist_from_edge` | Éloignement du bord 20 km | Drugan 2022 |
| `lat_drift/lon_drift` | Direction de déplacement du centroïde | Novel |
| `centroid_speed_km` | Vitesse de déplacement de l'orage | Leinonen 2022 |
| Terrain SRTM (8 features) | Orographie = contexte pour durée d'alerte | Leinonen 2023 |
| Météo ERA5 (17 features) | CAPE, LI, etc. si disponible | Littérature générale |

**Total features v3 : ~70 features** (38 lightning + 8 terrain + 17 météo + nouvelles)

---

## Features implémentées — v3 (session 2026-03-10)

**71 features totales** (46 lightning + 8 terrain SRTM + 17 météo ERA5)

Ajouts par rapport à v2 :
- `sigma_level` : DFR / std(DFR[-5:]) — taux de décroissance normalisé. Mean=-0.329 (last flash) vs +0.087 (other). Signal présent mais faible pour XGBoost (gain≈0), sera plus utile pour LSTM/GRU.
- `spatial_bbox_km2` : bounding box lat/lon des 10 derniers éclairs × 111² → surface en km². Gain XGBoost = 41.5 (rank ~25).
- `flash_density_km2` : n_cg_so_far / (bbox_km2 + 1) — concentration spatiale.
- `lat_range_10`, `lon_range_10` : composantes de la bounding box (intermédiaires).

**Météo ERA5 complète (5/5 aéroports)** — Pise enfin disponible.

**Performances baseline XGBoost (paramètres par défaut, 71 features)** :

| Aéroport | AUC | AP |
|---|---|---|
| Ajaccio | 0.915 | 0.392 |
| Bastia | 0.931 | 0.270 |
| Biarritz | 0.894 | 0.339 |
| Nantes | 0.792 | 0.206 |
| Pise | 0.910 | 0.240 |
| **Moyenne** | **0.888** | **0.289** |

Note : Nantes faible (0.792) — ~200 alertes seulement, distribution différente.

**Top features XGBoost (gain, all airports)** :
1. `lightning_rank` (rang dans l'alerte) — 1229
2. `n_cg_so_far` (nb cumulatif de CG) — 1164
3. `rolling_ili_max_5` — 410
4. `rolling_ili_max_10` — 367
5. `flash_rate_5` — 323
6. `dist_spread` — 270
Confirme Stano 2010 : ILI max est le prédicteur le plus fort.

## Modèle unifié (tous aéroports) — résultats session 2026-03-10

**Découverte clé** : entraîner un seul modèle sur tous les aéroports est systématiquement
meilleur que les modèles par aéroport, grâce au transfer learning implicite.

XGBoost Optuna 30 trials, 73 features, entraîné sur ~42k éclairs (tous airports) :

| Aéroport | Per-airport (défaut) | Unifié (Optuna) | Δ AUC |
|---|---|---|---|
| Ajaccio | 0.913 | **0.928** | +0.015 |
| Bastia | 0.933 | **0.942** | +0.009 |
| Biarritz | 0.895 | **0.914** | +0.019 |
| Nantes | 0.797 | **0.850** | **+0.053** |
| Pise | 0.908 | **0.919** | +0.011 |
| **Moyenne** | 0.889 | **0.910** | **+0.021** |

Nantes bénéficie le plus (+5.3%) : 38% d'alertes triviales + seulement 164 alertes train.
Les features terrain SRTM + météo ERA5 permettent au modèle de distinguer les aéroports.

## Ablation study — contribution de chaque groupe de features (2026-03-10)

Modèle : XGBoost default, entraîné sur tous les aéroports unifiés, évalué par aéroport.
Dataset : 94 features (51 lightning + 26 terrain + 17 météo ERA5).

| Groupe | AUC moyen | Aja | Bas | Bia | Nan | Pis |
|---|---|---|---|---|---|---|
| Lightning only (51) | **0.907** | 0.925 | 0.942 | 0.908 | 0.844 | 0.918 |
| Lightning + Terrain (77) | 0.909 | 0.927 | 0.944 | 0.906 | 0.849 | 0.918 |
| Lightning + Terrain + Weather (94) | 0.908 | 0.928 | 0.945 | 0.910 | 0.842 | 0.917 |
| All (94) | 0.908 | 0.928 | 0.945 | 0.910 | 0.842 | 0.917 |

**Conclusion** : les features lightning dominent (~0.907). Terrain apporte +0.002. Météo apporte quasiment rien (+0.001), voire nuit à Nantes (-0.007 avec météo). Probablement car ERA5 ne capture pas les conditions synoptiques fines à l'échelle d'un orage.

**Implication** : se concentrer sur les features lightning pour la compétition. Terrain reste utile pour la discrimination inter-aéroport.

## Features v5 (ajoutées en session 2026-03-10, suite)

| Feature | Justification | Gain XGBoost (default, 100 features) |
|---|---|---|
| `rolling_ili_min_5` | Plus courte pause parmi les 5 derniers — burst récent si petit | ~15 |
| `flash_rate_3` | Taux très récent sur 3 éclairs | 141 (#4) |
| `fr_log_slope_3` | Décroissance log 3→5 éclairs (fenêtre fine) | ~10 |
| `ili_cv_5` | CV de l'ILI sur 5 — régularité de la décroissance | ~5 |
| `ic_cg_ratio` | Ratio IC:CG cumulé (cycle de vie) | ~8 |
| `n_ic_so_far` | Count cumulatif des éclairs intra-nuage | ~5 |
| `ili_vs_alert_max` | ILI courant / max ILI de l'alerte (→1 = nouveau record) | 29 (#14) |

Total features v5 : **100** (57 lightning + 26 terrain + 17 météo)

## Features encore manquantes

| Feature | Méthode | Priorité |
|---|---|---|
| CNN sur heightmap | features orographiques extraites du DEM .npy | Basse (phase finale) |

## Notebook 04_sequential.py — Architecture GRU

Créé en session 2026-03-10. Architecture :
- **Input** : 25 flash features + 5 static features → 30 dim
- **GRU bidirectionnel** : 2 couches × 128 hidden × 2 (bidir) = 513k params
- **Entraînement** : BCEWithLogitsLoss + pos_weight≈40, AdamW lr=1e-3, CosineAnnealingLR
- **Early stopping** : patience=10 epochs sur AUC eval

Justifications :
- GRU > LSTM pour alertes courtes (médiane ~15 éclairs)
- Bidir permet d'apprendre les patterns début/milieu en train
- En inférence temps-réel, version unidir serait nécessaire
- Modèle unique pour tous les aéroports (transfer learning implicite)

---

## Références nouvelles identifiées

- Stano et al. 2010 : JGR 115, D09205, doi:10.1029/2009JD013034
- Drugan & Preston 2022 : Atmosphere 13(7):1111, doi:10.3390/atmos13071111 [open access](https://www.mdpi.com/2073-4433/13/7/1111)
- Mansouri et al. 2023 : Atmosphere 14(12):1713 [open access](https://www.mdpi.com/2073-4433/14/12/1713)
- Schultz et al. 2009 : JAMC 48(12), doi:10.1175/2009JAMC2237.1
- Yang et al. 2018 : JGR Atmospheres, doi:10.1029/2017JD026759
- Bruning & MacGorman 2013 : JAS 70(12)
- HESS 2025 : doi:10.5194/hess-29-1-2025
- Wilhelm 2023 : QJRMS, doi:10.1002/qj.4505
- Lightning VLF-only 2025 : doi:10.1016/j.atmosres.2025.107740 (Atmos. Res.)

---

## Analyse opérationnelle : gain de temps vs règle des 30 minutes (2026-03-11)

### Ce que mesure l'AUC — clarification importante

**AUC = 0.914 ne signifie PAS "précision de 91%".**

L'AUC mesure : P(score(dernier éclair) > score(éclair quelconque)) = 91.4%.
C'est une métrique de **ranking**, pas de classification binaire.

En pratique avec seuil = 0.5 :
- Précision = 24% (1 bonne décision sur 4 triggers)
- Rappel = 68% (on rate 32% des vrais derniers éclairs)

Raison : le dataset est très déséquilibré (3.7% positifs = derniers éclairs).

### Simulation opérationnelle : K éclairs consécutifs >= seuil

Pour déclarer la cessation, on exige K éclairs consécutifs avec score >= θ.
Résultats sur 527 alertes éval (XGBoost default 400 arbres) :

| Stratégie | FAR | Alertes couvertes | Gain moyen |
|-----------|-----|-------------------|-----------|
| θ=0.5, K=1 (baseline) | 64% | 170/527 (32%) | +30 min |
| θ=0.7, K=2 consécutifs | 21% | 48/527 (9%) | +30 min |
| θ=0.7, K=3 consécutifs | 8% | 22/527 (4%) | +30 min |
| θ=0.85, K=3 consécutifs | 2% | 0/527 (0%) | — |
| **Conserv. θ=0.5, K=2** | **5%** | **13/527 (2.5%)** | **+30 min** |
| Conserv. θ=0.5, K=3 | 1% | 3/527 (0.6%) | +30 min |

"Conservateur" = modèle entraîné avec scale_pos_weight=2.0 au lieu de 19.3
→ AUC légèrement meilleur (0.908 vs 0.903) mais probabilités mieux calibrées.

**Interprétation** : quand le modèle se déclenche correctement, il tire EXACTEMENT
sur le dernier éclair → gain toujours ≈ 30 min. Mais la couverture est faible.

### Contexte vs littérature

Les meilleurs systèmes publiés (Stano 2010, Shafer 2019) gagnent **10-15 min**
avec FAR ~5-10% en utilisant radar dual-pol + capteurs. Nous gagnons potentiellement
30 min mais sur seulement 4-9% des alertes, avec données lightning uniquement.

### Météo live (Open-Meteo) — analyse coût/bénéfice

**Techniquement faisable** : `scripts/fetch_weather.py` peut retrouver les données
ERA5 historiques pour n'importe quelle date/coordonnée via Open-Meteo API.
Pour un jeu de test avec des dates connues, on pourrait enrichir les features.

**Mais l'ablation montre que la météo ERA5 N'AIDE PAS** :
- Lightning seul : AUC=0.902
- Lightning + Weather : AUC=0.902 (+0.000)
- Nantes spécifiquement : -0.009 avec météo (bruit dominant le signal)

→ **Ne pas ajouter la météo live.** Gain nul, complexité inutile.

### CNN orographie — pourquoi ça ne marche pas

Les grilles DEM 91×91 existent pour 5 aéroports. L'idée d'un CNN pour extraire
des patterns spatiaux de terrain est bonne en principe, mais structurellement limitée :

**5 aéroports = 5 images uniques**. Un CNN entraîné sur 5 images mémoriserait
les 5 profils (= apprend à identifier l'aéroport, pas l'orographie).
Ce que les 26 features terrain agrégées font déjà.

Fonctionnerait avec CNN si : position de l'éclair → image DEM centrée sur l'éclair
(56k images différentes). Mais les éclairs bougent dans le rayon de 20km,
et les grilles DEM sont centrées sur l'aéroport à 50km de rayon → pas aligné.

### Ce qui reste pour améliorer

Par ordre de ROI décroissant :
1. **Optuna bien fini sur 102 features** (60 trials) → estimé +0.003-0.005 AUC
2. **Ensemble XGB + GRU** → +0.003-0.008 si erreurs décorrélées
3. **Features lightning fines** : rolling_ili_median_5, n_flashes_last_2min
4. **Modèle Nantes séparé** (seul sous-performant : AUC=0.84 vs 0.94 Bastia)
5. CNN, météo live : non prioritaires

