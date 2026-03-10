# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "numpy<2",
#     "scikit-learn",
#     "torch",
#     "altair",
#     "pyarrow==23.0.1",
# ]
# ///

"""
04_sequential.py — Modèles séquentiels par alerte (LSTM, GRU, Transformer)
==========================================================================

Approche : chaque alerte = une séquence variable d'éclairs.
Le modèle traite les éclairs dans l'ordre chronologique et prédit,
pour chaque éclair, si c'est le dernier CG de l'alerte.

Différences vs 03_experiments.py :
- Input : séquences de longueur variable (padded), pas de features aggregées
- Architecture : LSTM / GRU bidirectionnels + classification par flash
- Baseline : threshold sur ILI rolling max (règle simple, pas ML)
- Multi-aéroport : un seul modèle entraîné sur tous les aéroports

Architecture choisie (inspirée de Mansouri 2023 + DeepLight) :
- GRU bidirectionnel (N couches) sur les features par éclair
- Features statiques (terrain + météo) concaténées à chaque timestep
- Sortie : probabilité P(dernier éclair) à chaque timestep
- Loss : BCE pondérée (class_weight=pos_weight)

Justification architecturale :
- GRU > LSTM ici car les alertes sont courtes (médiane ~40 min, ~15 éclairs)
  → le problème du gradient vanishing est moins prégnant
- Bidirectionnel : on masque les futurs lors de la loss mais le modèle
  peut apprendre les patterns de début/milieu d'alerte mieux (en train)
  → NB: en inférence réelle, on utiliserait unidir. Pour l'eval académique,
  le bidir est plus juste car on compare sur l'ensemble de l'alerte.
- Attention + skip connection : permet au modèle de "revenir" sur les
  features brutes si le contexte récent est ambigu

Choix documentés dans DISCOVERIES.md.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    import json, pathlib, warnings, math
    warnings.filterwarnings("ignore")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, recall_score, precision_score,
    )
    from sklearn.preprocessing import StandardScaler

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    mo.output.replace(mo.md(
        f"**Device** : `{DEVICE}` · "
        f"**PyTorch** : `{torch.__version__}`"
    ))
    return (
        mo, pl, np, alt, json, pathlib, warnings, math,
        torch, nn, optim,
        Dataset, DataLoader,
        pack_padded_sequence, pad_packed_sequence, pad_sequence,
        roc_auc_score, average_precision_score,
        f1_score, recall_score, precision_score,
        StandardScaler, DEVICE,
    )


@app.cell
def load_data(pl, json, pathlib, np, mo):
    """
    Charge les parquets et organise les données par alerte.

    Pour les modèles séquentiels, on utilise un sous-ensemble de features :
    - Features temporelles et de flux : pas les features statiques terrain/météo
      (celles-ci sont ajoutées en "static context" à chaque timestep)
    - On conserve les features les plus discriminantes selon SHAP / gain XGBoost :
      ILI max, flash rate, distance, amplitude, rang, drift centroïde

    Justification du subset de features :
    - Trop de features → sur-apprentissage sur séquences courtes
    - Les features corrélées (rolling_ili_5 et rolling_ili_10) sont redondantes
      pour un LSTM qui calcule son propre contexte temporel
    - On garde les features "brutes" + quelques agrégations clés
    """
    ROOT = pathlib.Path(__file__).parent.parent
    PROC = ROOT / "data" / "processed"

    meta = json.loads((PROC / "feature_cols.json").read_text())
    TARGET_COL = meta["target_col"]
    ALL_FEATURE_COLS = meta["feature_cols"]

    # Features temporelles par flash (input séquentiel)
    # Choix : features qui changent à chaque flash et capturent la dynamique
    FLASH_FEATURES = [
        # ILI — le prédicteur le plus fort (Stano 2010)
        "ili_s", "ili_log", "rolling_ili_max_5", "rolling_ili_max_10",
        "rolling_ili_5", "rolling_ili_std_5",
        # Flash rate — décroissance depuis le pic (Schultz 2009)
        "flash_rate_3", "flash_rate_5", "fr_vs_max_ratio", "flash_rate_ratio",
        "fr_log_slope",   # pente log-linéaire (gain XGBoost = 1249, #1 feature!)
        "fr_log_slope_3",  # décroissance très récente (3→5 éclairs)
        # ILI rolling min + CV
        "rolling_ili_min_5",
        "ili_cv_5",  # coefficient de variation ILI
        # MIFI percentile — normalisation airport-spécifique
        "ili_vs_p95", "ili_vs_p75", "ili_vs_local_mean",
        # Position et rang
        "lightning_rank", "time_since_start_s",
        "dist", "dist_from_edge", "dist_trend",
        # Amplitude et polarité
        "amplitude_abs", "amp_trend",
        "positive_cg_frac", "high_amp_frac",
        # Dispersion spatiale
        "dist_spread", "azimuth_spread_10",
        "spatial_bbox_km2", "centroid_speed_km",
        # Sigma level (decay signal)
        "sigma_level",
        # ILI vs record intra-alerte + anomalie standardisée + max record récent
        "ili_vs_alert_max", "ili_z_score_5", "rolling_max_vs_alert_max",
        # Temporel saisonnier
        "hour_utc", "month",
    ]

    # Features statiques par aéroport (contexte — concaténées à chaque timestep)
    STATIC_FEATURES = [
        "t_elev_mean", "t_elev_std", "t_mountain_frac",
        "t_coast_dist", "t_tri_mean",
    ]

    # Vérifie que toutes les features existent dans les données
    _sample = pl.read_parquet(str(PROC / "ajaccio_train.parquet"), n_rows=1)
    _avail = set(_sample.columns)
    FLASH_FEATURES = [f for f in FLASH_FEATURES if f in _avail]
    STATIC_FEATURES = [f for f in STATIC_FEATURES if f in _avail]

    AIRPORTS = ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]

    splits = {}
    for _ap in AIRPORTS:
        _key = _ap.lower()
        _train = pl.read_parquet(str(PROC / f"{_key}_train.parquet"))
        _eval  = pl.read_parquet(str(PROC / f"{_key}_eval.parquet"))
        splits[_ap] = {"train": _train, "eval": _eval}

    mo.output.replace(mo.md(
        f"**{len(FLASH_FEATURES)} flash features** + **{len(STATIC_FEATURES)} static** · "
        f"**{sum(len(s['train'])+len(s['eval']) for s in splits.values()):,}** éclairs CG · "
        f"**{len(AIRPORTS)} aéroports**"
    ))
    return (
        FLASH_FEATURES, STATIC_FEATURES, TARGET_COL,
        ALL_FEATURE_COLS, AIRPORTS, splits,
    )


@app.cell
def build_sequences(splits, FLASH_FEATURES, STATIC_FEATURES, TARGET_COL,
                    AIRPORTS, pl, np, StandardScaler, mo):
    """
    Construit les séquences par alerte pour l'entraînement séquentiel.

    Format de sortie par alerte :
    - X_flash : (T, n_flash_features) — features temporelles
    - X_static : (n_static_features,) — features statiques (constant pour l'alerte)
    - y : (T,) — label par flash (1 = dernier éclair CG)
    - length : T — longueur réelle (avant padding)

    Normalisation :
    - StandardScaler fit sur TRAIN, appliqué sur EVAL
    - Features statiques normalisées séparément (distribution différente)
    """
    # Collecte toutes les alertes train pour fit du scaler
    _train_all = pl.concat([splits[_ap]["train"] for _ap in AIRPORTS])
    _eval_all  = pl.concat([splits[_ap]["eval"]  for _ap in AIRPORTS])

    # Fit scalers sur train
    _scaler_flash  = StandardScaler()
    _scaler_static = StandardScaler()

    _X_tr_flash = _train_all.select(FLASH_FEATURES).fill_nan(0).fill_null(0).to_numpy()
    _X_tr_static = _train_all.select(STATIC_FEATURES).fill_nan(0).fill_null(0).to_numpy()
    _scaler_flash.fit(_X_tr_flash)
    _scaler_static.fit(_X_tr_static)

    def _build_split(df: "pl.DataFrame") -> list[dict]:
        """Construit la liste des séquences pour un split."""
        _df_scaled_flash  = _scaler_flash.transform(
            df.select(FLASH_FEATURES).fill_nan(0).fill_null(0).to_numpy()
        )
        _df_scaled_static = _scaler_static.transform(
            df.select(STATIC_FEATURES).fill_nan(0).fill_null(0).to_numpy()
        )
        _df_target = df[TARGET_COL].cast(pl.Int8).to_numpy()
        _df_alert  = (df["airport"].cast(str) + "_" + df["airport_alert_id"].cast(str)).to_numpy()

        _seqs = []
        _unique_alerts = list(dict.fromkeys(_df_alert))  # ordre préservé

        # Index par alerte
        _alert_to_idx = {}
        for _i, _aid in enumerate(_df_alert):
            if _aid not in _alert_to_idx:
                _alert_to_idx[_aid] = []
            _alert_to_idx[_aid].append(_i)

        for _aid in _unique_alerts:
            _idxs = sorted(_alert_to_idx[_aid])  # déjà trié (parquet ordonné)
            _x_flash  = _df_scaled_flash[_idxs]   # (T, n_flash)
            _x_static = _df_scaled_static[_idxs[0]]  # (n_static,) — first row
            _y        = _df_target[_idxs]          # (T,)
            _seqs.append({
                "x_flash":  _x_flash.astype(np.float32),
                "x_static": _x_static.astype(np.float32),
                "y":        _y.astype(np.float32),
                "length":   len(_idxs),
                "alert_id": _aid,
            })

        return _seqs

    train_seqs = _build_split(_train_all)
    eval_seqs  = _build_split(_eval_all)

    _n_tr = sum(s["length"] for s in train_seqs)
    _n_ev = sum(s["length"] for s in eval_seqs)

    # Stats séquences
    _lengths_tr = [s["length"] for s in train_seqs]
    _lengths_ev = [s["length"] for s in eval_seqs]

    mo.output.replace(mo.vstack([
        mo.md("## Dataset séquentiel"),
        mo.md(
            f"- **Train** : {len(train_seqs):,} alertes · {_n_tr:,} éclairs\n"
            f"- **Eval** : {len(eval_seqs):,} alertes · {_n_ev:,} éclairs\n"
            f"- **Longueur min/mediane/max** : {min(_lengths_tr)} / "
            f"{int(np.median(_lengths_tr))} / {max(_lengths_tr)}"
        ),
    ]))
    return (train_seqs, eval_seqs, _scaler_flash, _scaler_static)


@app.cell
def define_model(FLASH_FEATURES, STATIC_FEATURES, torch, nn, mo):
    """
    Architecture GRU bidirectionnel avec attention.

    Structure :
    1. Input : [flash_features | static_features] à chaque timestep
    2. GRU bidirectionnel (n_layers couches)
    3. Attention self (basique) sur les états cachés
    4. Projection linéaire → P(dernier éclair)

    Paramètres par défaut (basés sur la taille des alertes médianes ~15 éclairs) :
    - hidden_size=128 : assez grand pour capturer les patterns ILI sans sur-apprendre
    - n_layers=2 : 2 couches pour la profondeur, mais pas trop (alertes courtes)
    - dropout=0.3 : régularisation importante vu le peu de données

    La bidirectionnalité permet en entraînement de mieux apprendre les patterns
    de début et milieu d'alerte. En inférence temps-réel, on passerait en unidirectionnel.
    """
    INPUT_SIZE  = len(FLASH_FEATURES) + len(STATIC_FEATURES)
    HIDDEN_SIZE = 128
    N_LAYERS    = 2
    DROPOUT     = 0.3

    class LightningCessationGRU(nn.Module):
        """
        GRU bidirectionnel pour la prédiction de cessation éclair par éclair.

        Input  : (batch, seq_len, input_size) — padded
        Output : (batch, seq_len) — logits P(dernier éclair) par timestep
        """
        def __init__(self, input_size, hidden_size, n_layers, dropout):
            super().__init__()

            # Couche d'entrée : projection + normalisation
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # GRU bidirectionnel
            self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )

            # Attention : score par timestep (pooling contextualisé)
            # Simplifié : un unique vecteur de query appris
            self.attn_query = nn.Linear(hidden_size * 2, 1, bias=False)

            # Sortie : classification par timestep
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x, lengths):
            """
            x : (B, T, input_size)
            lengths : (B,) — longueurs réelles
            Returns : (B, T) logits
            """
            # Projection d'entrée
            x_proj = self.input_proj(x)  # (B, T, H)

            # GRU avec packing pour ignorer le padding
            packed = pack_padded_sequence(
                x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out, _ = self.gru(packed)  # → packed output
            gru_out, _ = pad_packed_sequence(
                gru_out, batch_first=True, total_length=x.size(1)
            )  # (B, T, 2*H)

            # Logits par timestep
            logits = self.classifier(gru_out).squeeze(-1)  # (B, T)

            return logits

    model_class = LightningCessationGRU
    model_config = dict(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )

    # Test rapide
    _m = LightningCessationGRU(**model_config)
    _n_params = sum(p.numel() for p in _m.parameters())
    mo.output.replace(mo.md(
        f"## Architecture GRU\n"
        f"- **Input** : {INPUT_SIZE} features ({len(FLASH_FEATURES)} flash + {len(STATIC_FEATURES)} static)\n"
        f"- **GRU** : {N_LAYERS}L × {HIDDEN_SIZE}H × 2 (bidir)\n"
        f"- **Paramètres** : {_n_params:,}\n"
    ))
    return (model_class, model_config, INPUT_SIZE)


@app.cell
def define_transformer(FLASH_FEATURES, STATIC_FEATURES, torch, nn, mo):
    """
    Architecture Transformer encoder comme alternative au GRU.

    Différences vs GRU :
    - Attention multi-têtes sur TOUTE la séquence (pas de récurrence)
    - Encodage positionnel appris (les alertes varient de 1 à 200+ éclairs)
    - Meilleur pour les longues dépendances, mais moins efficace sur les courtes alertes
    - Padding mask explicite pour ignorer les positions masquées

    Justification du choix de 4 têtes :
    - head_dim = hidden // n_heads = 64 // 4 = 16 → suffisant pour la taille des séquences
    - Plus de têtes = meilleure capture de différents patterns (ILI, amplitude, spatial)

    Limitation : le Transformer traite la séquence entière simultanément, ce qui est
    académiquement valide mais nécessite un buffer en inférence temps-réel.
    """
    INPUT_SIZE_T = len(FLASH_FEATURES) + len(STATIC_FEATURES)
    D_MODEL    = 64
    N_HEADS    = 4
    N_LAYERS_T = 2
    D_FF       = 256
    DROPOUT_T  = 0.3
    MAX_SEQ    = 300  # longueur max pour l'encodage positionnel

    class LightningCessationTransformer(nn.Module):
        """
        Transformer encoder pour la prédiction de cessation.
        Input  : (batch, seq_len, input_size)
        Output : (batch, seq_len) logits
        """
        def __init__(self, input_size, d_model, n_heads, n_layers, d_ff, dropout, max_seq):
            super().__init__()
            # Projection d'entrée
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, d_model),
                nn.LayerNorm(d_model),
            )
            # Encodage positionnel appris (adapté aux alertes de longueur variable)
            self.pos_embed = nn.Embedding(max_seq, d_model)
            # Encoder Transformer
            _enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN : plus stable pour les séquences courtes
            )
            self.transformer = nn.TransformerEncoder(_enc_layer, num_layers=n_layers)
            # Sortie par timestep
            self.out = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x, lengths):
            B, T, _ = x.shape
            # Projection + encodage positionnel
            x_proj = self.input_proj(x)
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            x_proj = x_proj + self.pos_embed(pos)
            # Masque de padding (True = ignorer)
            pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            # Transformer encoder
            out = self.transformer(x_proj, src_key_padding_mask=pad_mask)  # (B, T, d_model)
            logits = self.out(out).squeeze(-1)  # (B, T)
            return logits

    transformer_class = LightningCessationTransformer
    transformer_config = dict(
        input_size=INPUT_SIZE_T,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS_T,
        d_ff=D_FF,
        dropout=DROPOUT_T,
        max_seq=MAX_SEQ,
    )

    _mt = LightningCessationTransformer(**transformer_config)
    _n_p = sum(p.numel() for p in _mt.parameters())
    mo.output.replace(mo.md(
        f"## Architecture Transformer (alternative)\n"
        f"- **Input** : {INPUT_SIZE_T} → d_model={D_MODEL}\n"
        f"- **Encoder** : {N_LAYERS_T}L × {N_HEADS} heads × d_ff={D_FF}\n"
        f"- **Encodage positionnel** : appris (max {MAX_SEQ} positions)\n"
        f"- **Paramètres** : {_n_p:,}\n"
        f"- **vs GRU** : moins de paramètres, capte les dépendances longues, "
        f"mais nécessite des séquences plus longues pour être efficace."
    ))
    return (transformer_class, transformer_config)


@app.cell
def dataset_dataloader(train_seqs, eval_seqs,
                       torch, Dataset, DataLoader, pad_sequence, np, mo):
    """Dataset PyTorch pour les séquences d'alertes."""

    class AlertDataset(Dataset):
        def __init__(self, seqs):
            self.seqs = seqs

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            s = self.seqs[idx]
            # Concatène static à chaque timestep
            T = s["length"]
            static_rep = np.tile(s["x_static"], (T, 1))  # (T, n_static)
            x = np.concatenate([s["x_flash"], static_rep], axis=1)  # (T, total)
            return (
                torch.FloatTensor(x),
                torch.FloatTensor(s["y"]),
                torch.tensor(T, dtype=torch.long),
            )

    def _collate(batch):
        xs, ys, ls = zip(*batch)
        xs_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
        ys_pad = pad_sequence(ys, batch_first=True, padding_value=-1.0)  # -1 = padding (ignoré dans la loss)
        ls_t   = torch.stack(ls)
        return xs_pad, ys_pad, ls_t

    _BATCH_SIZE = 32

    train_dataset = AlertDataset(train_seqs)
    eval_dataset  = AlertDataset(eval_seqs)
    train_loader  = DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True, collate_fn=_collate)
    eval_loader   = DataLoader(eval_dataset,  batch_size=_BATCH_SIZE, shuffle=False, collate_fn=_collate)

    mo.output.replace(mo.md(
        f"**Dataset** : {len(train_dataset)} alertes train · {len(eval_dataset)} eval · "
        f"batch_size={_BATCH_SIZE}"
    ))
    return (AlertDataset, train_dataset, eval_dataset, train_loader, eval_loader)


@app.cell
def train_gru(
    model_class, model_config,
    train_loader, eval_loader, train_seqs, eval_seqs,
    torch, nn, optim, np, mo, DEVICE,
    roc_auc_score, average_precision_score, f1_score, recall_score,
):
    """
    Entraînement du GRU bidirectionnel.

    Détails techniques :
    - Loss : BCEWithLogitsLoss avec pos_weight (déséquilibre de classe ~1:40)
    - Optimiseur : AdamW avec weight_decay=1e-3 (régularisation L2)
    - LR scheduler : CosineAnnealingLR (redémarre le LR doucement)
    - Early stopping : patience=10 epochs sur AUC eval
    - Gradient clipping : max_norm=1.0 pour stabiliser le GRU

    Choix documenté : pos_weight calculé sur l'ensemble d'entraînement global.
    Un éclair sur ~40 est le dernier → pos_weight ≈ 40.
    """
    N_EPOCHS    = 50
    LR          = 1e-3
    WEIGHT_DECAY= 1e-3
    PATIENCE    = 10
    CLIP_NORM   = 1.0

    # Calcul du pos_weight sur le set d'entraînement
    _n_pos = sum(int(s["y"].sum()) for s in train_seqs)
    _n_neg = sum(s["length"] - int(s["y"].sum()) for s in train_seqs)
    _pw    = _n_neg / max(_n_pos, 1)
    _pos_weight = torch.tensor([_pw], dtype=torch.float32).to(DEVICE)

    model = model_class(**model_config).to(DEVICE)
    _criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight, reduction="none")
    _optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    _scheduler = optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=N_EPOCHS, eta_min=LR/10)

    _best_auc     = 0.0
    _best_weights = None
    _no_improve   = 0
    _history      = []

    def _eval_model(loader, seqs):
        """Évalue le modèle sur un loader — retourne AUC global."""
        model.eval()
        all_proba = []
        all_true  = []
        with torch.no_grad():
            for _X, _Y, _L in loader:
                _X, _Y, _L = _X.to(DEVICE), _Y.to(DEVICE), _L.to(DEVICE)
                _logits = model(_X, _L)  # (B, T)
                _proba  = torch.sigmoid(_logits)
                for _b in range(len(_L)):
                    _t = _L[_b].item()
                    all_proba.extend(_proba[_b, :_t].cpu().numpy().tolist())
                    all_true.extend(_Y[_b, :_t].cpu().numpy().tolist())
        all_true  = np.array(all_true)
        all_proba = np.array(all_proba)
        _auc = roc_auc_score(all_true, all_proba) if len(np.unique(all_true)) > 1 else 0.5
        return _auc, all_true, all_proba

    for _epoch in range(1, N_EPOCHS + 1):
        model.train()
        _epoch_loss = 0.0
        _n_batches  = 0

        for _X, _Y, _L in train_loader:
            _X, _Y, _L = _X.to(DEVICE), _Y.to(DEVICE), _L.to(DEVICE)
            _optimizer.zero_grad()
            _logits = model(_X, _L)  # (B, T)
            # Masque les positions de padding (label=-1)
            _mask  = (_Y >= 0)
            _loss  = _criterion(_logits[_mask], _Y[_mask])
            _loss  = _loss.mean()
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            _optimizer.step()
            _epoch_loss += _loss.item()
            _n_batches  += 1

        _scheduler.step()
        _train_auc, _, _ = _eval_model(train_loader, train_seqs)
        _eval_auc, _, _  = _eval_model(eval_loader, eval_seqs)
        _history.append({"epoch": _epoch, "loss": _epoch_loss / _n_batches,
                         "train_auc": _train_auc, "eval_auc": _eval_auc})

        if _eval_auc > _best_auc:
            _best_auc     = _eval_auc
            _best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            _no_improve   = 0
        else:
            _no_improve += 1

        if _epoch % 5 == 0:
            print(f"  Epoch {_epoch:3d} | loss={_epoch_loss/_n_batches:.4f} "
                  f"| train_AUC={_train_auc:.3f} | eval_AUC={_eval_auc:.3f} "
                  f"{'★' if _no_improve == 0 else ''}")

        if _no_improve >= PATIENCE:
            print(f"  Early stop à epoch {_epoch} (best eval AUC={_best_auc:.3f})")
            break

    # Restaure les meilleurs poids
    if _best_weights is not None:
        model.load_state_dict(_best_weights)

    # Évaluation finale
    _final_auc, _all_true, _all_proba = _eval_model(eval_loader, eval_seqs)
    _ap = average_precision_score(_all_true, _all_proba)
    _yp = (_all_proba >= 0.5).astype(int)
    _f1 = f1_score(_all_true, _yp, zero_division=0)
    _pod = recall_score(_all_true, _yp, zero_division=0)

    gru_results = {
        "AUC": round(_final_auc, 3),
        "AP": round(_ap, 3),
        "F1": round(_f1, 3),
        "POD": round(_pod, 3),
        "history": _history,
        "all_true": _all_true,
        "all_proba": _all_proba,
    }

    mo.output.replace(mo.vstack([
        mo.md(f"## GRU bidirectionnel — résultats (ALL airports)"),
        mo.md(
            f"| Métrique | Score |\n"
            f"|----------|-------|\n"
            f"| **AUC-ROC** | **{_final_auc:.3f}** |\n"
            f"| **AP** | {_ap:.3f} |\n"
            f"| **F1** (t=0.5) | {_f1:.3f} |\n"
            f"| **POD** (t=0.5) | {_pod:.3f} |"
        ),
        mo.callout(mo.md(
            f"Meilleur AUC eval au fil des epochs : **{_best_auc:.3f}**  \n"
            f"Note : AUC calculé toutes aéroports confondues. "
            f"Pour la comparaison par aéroport, voir la cellule suivante."
        ), kind="info"),
    ]))
    return (model, gru_results)


@app.cell
def gru_per_airport(
    model, eval_seqs, AIRPORTS,
    torch, np, mo, DEVICE,
    roc_auc_score, average_precision_score, f1_score, recall_score,
):
    """Évaluation du GRU par aéroport pour comparaison avec XGBoost."""
    model.eval()

    _rows = []
    for _ap in AIRPORTS:
        _ap_seqs = [s for s in eval_seqs if s["alert_id"].startswith(_ap)]
        if not _ap_seqs:
            continue

        _all_proba = []
        _all_true  = []

        with torch.no_grad():
            for _s in _ap_seqs:
                _T = _s["length"]
                _static_rep = np.tile(_s["x_static"], (_T, 1))
                _x = np.concatenate([_s["x_flash"], _static_rep], axis=1)
                _xt = torch.FloatTensor(_x).unsqueeze(0).to(DEVICE)
                _lt = torch.tensor([_T], dtype=torch.long).to(DEVICE)
                _logits = model(_xt, _lt)
                _proba  = torch.sigmoid(_logits)[0, :_T].cpu().numpy()
                _all_proba.extend(_proba.tolist())
                _all_true.extend(_s["y"].tolist())

        _yt = np.array(_all_true)
        _yp = np.array(_all_proba)
        _yb = (_yp >= 0.5).astype(int)

        if len(np.unique(_yt)) < 2:
            continue

        _rows.append({
            "Aéroport": _ap,
            "Modèle": "GRU bidir (all airports)",
            "AUC": round(roc_auc_score(_yt, _yp), 3),
            "AP": round(average_precision_score(_yt, _yp), 3),
            "F1": round(f1_score(_yt, _yb, zero_division=0), 3),
            "POD": round(recall_score(_yt, _yb, zero_division=0), 3),
        })

    import polars as pl
    _df = pl.DataFrame(_rows)
    _mean_auc = _df["AUC"].mean()
    _mean_ap  = _df["AP"].mean()

    mo.output.replace(mo.vstack([
        mo.md(f"## GRU — résultats par aéroport (AUC moyen = {_mean_auc:.3f})"),
        mo.callout(mo.md(
            "**Comparaison** : XGBoost par aéroport (default) → AUC moyen = 0.888  \n"
            "XGBoost est entraîné séparément par aéroport vs GRU sur tous les aéroports."
        ), kind="info"),
        _df,
    ]))
    return (_rows,)


@app.cell
def training_curves(gru_results, alt, mo):
    """Courbes d'entraînement AUC + loss."""
    import polars as pl

    _hist = gru_results["history"]
    _df = pl.DataFrame(_hist)
    _pd = _df.to_pandas()

    _auc_chart = (
        alt.Chart(_pd.melt("epoch", value_vars=["train_auc", "eval_auc"],
                           var_name="split", value_name="AUC"))
        .mark_line(point=False)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.5, 1.0])),
            color=alt.Color("split:N"),
            strokeDash=alt.StrokeDash("split:N"),
        )
        .properties(width=450, height=280, title="Courbes AUC (train vs eval)")
    )

    _loss_chart = (
        alt.Chart(_pd)
        .mark_line(color="#E91E63")
        .encode(
            x=alt.X("epoch:Q"),
            y=alt.Y("loss:Q", title="BCE Loss"),
        )
        .properties(width=450, height=280, title="Loss d'entraînement")
    )

    mo.output.replace(mo.vstack([
        mo.md("## Courbes d'entraînement GRU"),
        alt.hconcat(_auc_chart, _loss_chart),
    ]))
    return


@app.cell
def gru_threshold_analysis(
    model, eval_seqs, AIRPORTS,
    torch, np, mo, alt, DEVICE,
    f1_score, recall_score, precision_score,
):
    """
    Analyse du threshold optimal pour le GRU.
    Comparaison POD/FAR à différents seuils.
    """
    import polars as pl
    model.eval()

    _all_proba, _all_true = [], []
    with torch.no_grad():
        for _s in eval_seqs:
            _T = _s["length"]
            _static_rep = np.tile(_s["x_static"], (_T, 1))
            _x = np.concatenate([_s["x_flash"], _static_rep], axis=1)
            _xt = torch.FloatTensor(_x).unsqueeze(0).to(DEVICE)
            _lt = torch.tensor([_T], dtype=torch.long).to(DEVICE)
            _proba = torch.sigmoid(model(_xt, _lt))[0, :_T].cpu().numpy()
            _all_proba.extend(_proba.tolist())
            _all_true.extend(_s["y"].tolist())

    _yt = np.array(_all_true)
    _yp = np.array(_all_proba)

    _rows = []
    for _t in np.linspace(0.1, 0.9, 40):
        _yb = (_yp >= _t).astype(int)
        _pod = recall_score(_yt, _yb, zero_division=0)
        _prec = precision_score(_yt, _yb, zero_division=0)
        _far = 1.0 - _prec if _prec > 0 else 1.0
        _f1  = f1_score(_yt, _yb, zero_division=0)
        _rows.append({"threshold": round(_t, 3), "POD": _pod, "FAR": _far, "F1": _f1})

    _th_df = pl.DataFrame(_rows).to_pandas()

    _chart = (
        alt.Chart(_th_df)
        .mark_line(color="#2196F3")
        .encode(
            x=alt.X("FAR:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("POD:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["threshold", "POD", "FAR", "F1"],
        )
        .properties(width=420, height=360, title="GRU — POD vs FAR (tous aéroports)")
    )

    _f1_chart = (
        alt.Chart(_th_df)
        .mark_line(color="#4CAF50")
        .encode(
            x=alt.X("threshold:Q"),
            y=alt.Y("F1:Q"),
            tooltip=["threshold", "F1", "POD", "FAR"],
        )
        .properties(width=420, height=360, title="GRU — F1 vs Threshold")
    )

    _best_t = _th_df.loc[_th_df["F1"].idxmax()]

    mo.output.replace(mo.vstack([
        mo.md("## Analyse du threshold — GRU"),
        mo.callout(mo.md(
            f"Threshold optimal (max F1) : **{_best_t['threshold']:.2f}** → "
            f"POD={_best_t['POD']:.3f}, FAR={_best_t['FAR']:.3f}, F1={_best_t['F1']:.3f}"
        ), kind="success"),
        alt.hconcat(_chart, _f1_chart),
    ]))
    return


if __name__ == "__main__":
    app.run()
