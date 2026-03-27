# Script de présentation — EasyOrage
## Data Battle IA PAU 2026 — 10 minutes

---

## Timing

| Slide | Sujet | Durée |
|---|---|---|
| 1 | Titre | 30s |
| 2 | Problématique | 1min 30s |
| 3 | Analyse des données | 1min 30s |
| 4 | Solution ML | 2min |
| 5 | Explicabilité | 1min 30s |
| 6 | Dashboard | 1min |
| 7 | Impact & Perspectives | 1min 30s |
| 8 | Conclusion | 30s |

---

## Slide 1 — Titre *(~30s)*

Bonjour à tous. Nous sommes l'équipe EasyOrage — Selim Mohamed, Lilian Pichard, Louis Besle.

Notre sujet : prédire en temps réel la fin d'une alerte orage autour d'un aéroport. Sans radar, sans satellite — uniquement à partir des données foudre Météorage.

---

## Slide 2 — Problématique *(~1min 30s)*

Quand un orage menace un aéroport, toutes les opérations au sol s'arrêtent. Les avions restent cloués au tarmac, les équipes se mettent à l'abri. C'est une règle de sécurité absolument justifiée.

Mais la fin de l'alerte, elle, suit une règle purement conservative : on attend **30 minutes après le dernier éclair détecté**. Peu importe si l'orage est passé depuis 25 minutes.

Ce que vous voyez ici, c'est la timeline d'une alerte type. Le dernier éclair tombe — et le chrono de 30 minutes démarre. Pendant ce temps, la piste est immobilisée pour rien.

Notre objectif : placer cette ligne en pointillé **le plus tôt possible après le dernier éclair**, tout en garantissant un Risk inférieur à 2 % — c'est-à-dire ne jamais lever l'alerte alors qu'un éclair dangereux peut encore survenir.

---

## Slide 3 — Analyse des données *(~1min 30s)*

Nous avons travaillé sur 507 000 éclairs nuage-sol, autour de 5 aéroports, sur 9 ans de données.

Première observation importante : les alertes sont courtes. 43 % ont cinq éclairs ou moins. Ça veut dire que le modèle doit être capable de décider très tôt, avec très peu d'historique.

Deuxième observation : le target est fortement déséquilibré. Seulement 3,7 % des flashs sont le dernier de leur alerte. On a compensé ça avec un pos_weight de 19 dans XGBoost.

Pour capturer la physique de l'orage, on a construit **106 features depuis zéro** : des features sur la foudre elle-même, sur le terrain SRTM, et sur la météo ERA5.

Et là, résultat contre-intuitif : notre ablation study montre que les features foudre seules atteignent un AUC de 0.907. Ajouter la météo n'apporte strictement rien — voire nuit légèrement sur Nantes. On a donc concentré tous nos efforts sur les features foudre.

---

## Slide 4 — Solution ML *(~2min)*

Notre pipeline est le suivant. À chaque nouveau flash, on calcule les 106 features en temps réel. XGBoost produit un score entre 0 et 1 — la probabilité que ce flash soit le dernier de l'alerte. Et on applique une règle de déclenchement : on ne prédit la fin d'alerte que si K flashs **consécutifs** passent tous au-dessus d'un seuil θ. Ça évite les faux positifs isolés en début d'alerte.

Ce qui est important ici, c'est qu'on n'a pas optimisé l'AUC. On a optimisé **directement le Gain G** — la métrique du challenge — via Optuna, en cherchant le meilleur couple (K, θ) sous la contrainte Risk < 2 %. 200 trials, environ 30 minutes d'entraînement sur CPU.

Les résultats sur le dataset test officiel, que nous n'avons pas vu pendant l'entraînement : **212 heures de Gain**, avec un Risk de 0,8 % — très en dessous de la contrainte.

Pour donner un ordre de grandeur : la baseline à 30 minutes immobilise les pistes en moyenne 30 minutes après le dernier éclair. Nous, on les libère en moyenne bien avant — avec une sécurité maintenue.

---

## Slide 5 — Explicabilité *(~1min 30s)*

Regardons maintenant pourquoi le modèle fonctionne. La feature numéro 1 de loin, c'est `fr_log_slope` — la pente du logarithme du flash rate sur les dernières minutes.

Ce n'est pas un artefact statistique. C'est une observation issue de la littérature : Schultz (2009) a montré que la fréquence des éclairs chute de manière **exponentielle** dans les dernières minutes d'un orage. Notre feature capture exactement ça.

Ensuite, `rolling_ili_3` et `rolling_ili_5` : l'inter-éclair interval glissant. Quand les éclairs s'espacent de plus en plus, l'orage s'essouffle.

Et `centroid_dist_5` : la distance entre l'aéroport et le centroïde des 5 derniers éclairs. Quand l'orage se déplace et s'éloigne, la distance augmente.

Chacune de ces features peut être **expliquée à un contrôleur aérien** en une phrase. C'est un atout majeur pour la mise en production : la décision du modèle n'est pas une boîte noire.

---

## Slide 6 — Dashboard *(~1min)*

Pour rendre la solution concrète et démontrer son usage opérationnel, on a développé un dashboard temps réel.

Il permet de rejouer n'importe quelle alerte historique éclair par éclair, à vitesse accélérée. À chaque flash, le score du modèle est mis à jour et affiché. On voit le score monter progressivement vers 1 à mesure que l'orage s'affaiblit.

L'architecture : FastAPI côté backend avec une connexion WebSocket, React et Recharts côté frontend. La latence est inférieure à 50 millisecondes.

L'idée derrière cet outil, c'est de donner au contrôleur **une seule métrique à surveiller** — la probabilité de fin d'alerte — sans le noyer dans des données brutes.

---

## Slide 7 — Impact & Perspectives *(~1min 30s)*

Sur l'impact environnemental. Notre choix de XGBoost n'est pas un choix par défaut — c'est un choix délibéré. Inférence en moins d'une milliseconde, CPU seul, aucune dépendance cloud. On a évalué et rejeté des alternatives plus lourdes : un CNN sur données orographiques, un LLM — 100 fois plus gourmands en énergie pour zéro gain mesurable.

Le gain indirect est réel : un A320 immobilisé au sol consomme environ 2,5 tonnes de kérosène par heure. Chaque heure de piste récupérée, c'est autant d'émissions évitées.

Sur les perspectives. Ce modèle est déployable tel quel : une API légère, des données foudre en entrée, une prédiction en sortie. L'intégration dans le système Météorage est techniquement directe.

Et parce que le modèle est **unifié** — entraîné sur tous les aéroports à la fois — il généralise naturellement à un nouvel aéroport avec peu de données historiques.

---

## Slide 8 — Conclusion *(~30s)*

Pour conclure. Nous avons résolu le problème posé : prédire la fin d'une alerte orage en temps réel, avec des données foudre uniquement. Le modèle est interprétable, léger, et prêt pour la production.

212 heures de piste récupérées sur le dataset test. Un Risk de 0,8 %. Sans radar. Sans satellite. Juste les éclairs.

Merci de votre attention — on est disponibles pour vos questions.
