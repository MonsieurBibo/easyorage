# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "numpy<2",
#     "scikit-learn",
#     "torch",
#     "pyarrow==23.0.1",
#     "joblib",
# ]
# ///
"""
GRU bidirectionnel — script standalone avec checkpoint.

Résumable : si models/gru_checkpoint.pt existe, reprend depuis le dernier epoch.
Résultat  : models/gru_best.pt (meilleurs poids, AUC-based)
            models/gru_checkpoint.pt (dernier epoch, pour reprendre)
            models/gru_scalers.joblib (scalers train, pour inférence)
"""

import json
import pathlib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

ROOT  = pathlib.Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
MODEL = ROOT / "models"
MODEL.mkdir(exist_ok=True)

DEVICE = "cpu"  # AMD Cezanne integrated GPU — ROCm non supporté

# ── Hyperparamètres ───────────────────────────────────────────────────────────
N_EPOCHS    = 30
LR          = 1e-3
WEIGHT_DECAY= 1e-3
PATIENCE    = 10
CLIP_NORM   = 1.0
BATCH_SIZE  = 32
HIDDEN_SIZE = 128
N_LAYERS    = 2
DROPOUT     = 0.3

AIRPORTS = ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]

# ── Features ──────────────────────────────────────────────────────────────────
FLASH_FEATURES = [
    "ili_s", "ili_log", "rolling_ili_max_5", "rolling_ili_max_10",
    "rolling_ili_5", "rolling_ili_std_5",
    "flash_rate_3", "flash_rate_5", "fr_vs_max_ratio", "flash_rate_ratio",
    "fr_log_slope", "fr_log_slope_3",
    "rolling_ili_min_5", "ili_cv_5",
    "ili_vs_p95", "ili_vs_p75", "ili_vs_local_mean",
    "lightning_rank", "time_since_start_s",
    "dist", "dist_from_edge", "dist_trend",
    "amplitude_abs", "amp_trend",
    "positive_cg_frac", "high_amp_frac",
    "dist_spread", "azimuth_spread_10",
    "spatial_bbox_km2", "centroid_speed_km",
    "sigma_level",
    "ili_vs_alert_max", "ili_z_score_5", "rolling_max_vs_alert_max",
    "hour_utc", "month",
]
STATIC_FEATURES = [
    "t_elev_mean", "t_elev_std", "t_mountain_frac",
    "t_coast_dist", "t_tri_mean",
]
TARGET_COL = "is_last_lightning_cloud_ground"

# ── Chargement données ────────────────────────────────────────────────────────
splits = {}
for ap in AIRPORTS:
    key = ap.lower()
    splits[ap] = {
        "train": pl.read_parquet(str(PROC / f"{key}_train.parquet")),
        "eval":  pl.read_parquet(str(PROC / f"{key}_eval.parquet")),
    }

# Filtre les features disponibles
_sample = splits["Ajaccio"]["train"]
_avail  = set(_sample.columns)
FLASH_FEATURES  = [f for f in FLASH_FEATURES  if f in _avail]
STATIC_FEATURES = [f for f in STATIC_FEATURES if f in _avail]
INPUT_SIZE = len(FLASH_FEATURES) + len(STATIC_FEATURES)
print(f"Features: {len(FLASH_FEATURES)} flash + {len(STATIC_FEATURES)} static = {INPUT_SIZE} total")

# ── Normalisation ─────────────────────────────────────────────────────────────
train_all = pl.concat([splits[ap]["train"] for ap in AIRPORTS])
eval_all  = pl.concat([splits[ap]["eval"]  for ap in AIRPORTS])

scaler_flash  = StandardScaler()
scaler_static = StandardScaler()
scaler_flash.fit(train_all.select(FLASH_FEATURES).fill_nan(0).fill_null(0).to_numpy())
scaler_static.fit(train_all.select(STATIC_FEATURES).fill_nan(0).fill_null(0).to_numpy())

joblib.dump({"scaler_flash": scaler_flash, "scaler_static": scaler_static,
             "flash_features": FLASH_FEATURES, "static_features": STATIC_FEATURES},
            MODEL / "gru_scalers.joblib")

# ── Construction séquences ────────────────────────────────────────────────────
def build_sequences(df: pl.DataFrame) -> list[dict]:
    X_flash  = scaler_flash.transform(df.select(FLASH_FEATURES).fill_nan(0).fill_null(0).to_numpy())
    X_static = scaler_static.transform(df.select(STATIC_FEATURES).fill_nan(0).fill_null(0).to_numpy())
    y_arr    = df[TARGET_COL].cast(pl.Int8).to_numpy()
    alert_id = (df["airport"].cast(str) + "_" + df["airport_alert_id"].cast(str)).to_numpy()

    alert_to_idx: dict[str, list[int]] = {}
    for i, aid in enumerate(alert_id):
        alert_to_idx.setdefault(aid, []).append(i)

    seqs = []
    for aid in dict.fromkeys(alert_id):
        idxs = sorted(alert_to_idx[aid])
        seqs.append({
            "x_flash":  X_flash[idxs].astype(np.float32),
            "x_static": X_static[idxs[0]].astype(np.float32),
            "y":        y_arr[idxs].astype(np.float32),
            "length":   len(idxs),
        })
    return seqs

print("Construction des séquences...")
train_seqs = build_sequences(train_all)
eval_seqs  = build_sequences(eval_all)
print(f"Train: {len(train_seqs)} alertes · Eval: {len(eval_seqs)} alertes")

# ── Dataset / DataLoader ──────────────────────────────────────────────────────
class AlertDataset(Dataset):
    def __init__(self, seqs): self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]
        x = np.concatenate([s["x_flash"], np.tile(s["x_static"], (s["length"], 1))], axis=1)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(s["y"], dtype=torch.float32),
            torch.tensor(s["length"], dtype=torch.long),
        )

def collate(batch):
    xs, ys, ls = zip(*batch)
    return (
        pad_sequence(xs, batch_first=True, padding_value=0.0),
        pad_sequence(ys, batch_first=True, padding_value=-1.0),
        torch.stack(ls),
    )

train_loader = DataLoader(AlertDataset(train_seqs), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
eval_loader  = DataLoader(AlertDataset(eval_seqs),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

# ── Architecture GRU ──────────────────────────────────────────────────────────
class LightningCessationGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn_query = nn.Linear(hidden_size * 2, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x, lengths):
        B, T, _ = x.shape
        x = self.proj(x.view(B * T, -1)).view(B, T, -1)
        packed  = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _  = self.gru(packed)
        out, _  = pad_packed_sequence(out, batch_first=True, total_length=T)
        logits  = self.classifier(out).squeeze(-1)  # (B, T)
        return logits

# ── Init modèle ───────────────────────────────────────────────────────────────
model = LightningCessationGRU(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modèle GRU : {n_params:,} paramètres · device={DEVICE}")

n_pos = sum(int(s["y"].sum()) for s in train_seqs)
n_neg = sum(s["length"] - int(s["y"].sum()) for s in train_seqs)
pw = n_neg / max(n_pos, 1)
print(f"pos_weight={pw:.1f}")

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pw], dtype=torch.float32).to(DEVICE),
    reduction="none",
)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR / 10)

# ── Chargement checkpoint si existant ────────────────────────────────────────
CKPT_PATH = MODEL / "gru_checkpoint.pt"
start_epoch = 1
best_auc    = 0.0
no_improve  = 0
history     = []

if CKPT_PATH.exists():
    print(f"\nCheckpoint trouvé → reprise depuis {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_auc    = ckpt["best_auc"]
    no_improve  = ckpt["no_improve"]
    history     = ckpt.get("history", [])
    print(f"  Epoch {ckpt['epoch']} · best_auc={best_auc:.4f} · no_improve={no_improve}")
else:
    print("\nAucun checkpoint — entraînement depuis zéro.")

# ── Evaluation ────────────────────────────────────────────────────────────────
def eval_model(loader):
    model.eval()
    all_true, all_proba = [], []
    with torch.no_grad():
        for X, Y, L in loader:
            X, Y, L = X.to(DEVICE), Y.to(DEVICE), L.to(DEVICE)
            logits = model(X, L)
            proba  = torch.sigmoid(logits)
            for b in range(len(L)):
                t = L[b].item()
                all_proba.extend(proba[b, :t].cpu().numpy().tolist())
                all_true.extend(Y[b, :t].cpu().numpy().tolist())
    all_true  = np.array(all_true)
    all_proba = np.array(all_proba)
    auc = roc_auc_score(all_true, all_proba) if len(np.unique(all_true)) > 1 else 0.5
    return auc, all_true, all_proba

# ── Boucle d'entraînement ─────────────────────────────────────────────────────
print(f"\n{'Epoch':>6} {'Loss':>8} {'Train AUC':>10} {'Eval AUC':>10} {'Best':>6}")
print("─" * 46)

for epoch in range(start_epoch, N_EPOCHS + 1):
    model.train()
    epoch_loss, n_batches = 0.0, 0

    for X, Y, L in train_loader:
        X, Y, L = X.to(DEVICE), Y.to(DEVICE), L.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X, L)
        mask   = Y >= 0
        loss   = criterion(logits[mask], Y[mask]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    train_auc, _, _ = eval_model(train_loader)
    eval_auc,  _, _ = eval_model(eval_loader)
    history.append({"epoch": epoch, "loss": epoch_loss / n_batches,
                    "train_auc": train_auc, "eval_auc": eval_auc})

    is_best = eval_auc > best_auc
    if is_best:
        best_auc   = eval_auc
        no_improve = 0
        torch.save(model.state_dict(), MODEL / "gru_best.pt")
    else:
        no_improve += 1

    flag = "★" if is_best else ""
    print(f"{epoch:>6} {epoch_loss/n_batches:>8.4f} {train_auc:>10.4f} {eval_auc:>10.4f} {flag}")

    # Checkpoint chaque epoch
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_auc":        best_auc,
        "no_improve":      no_improve,
        "history":         history,
    }, CKPT_PATH)

    if no_improve >= PATIENCE:
        print(f"\nEarly stop à epoch {epoch} (best eval AUC={best_auc:.4f})")
        break

# ── Résultats finaux ──────────────────────────────────────────────────────────
print(f"\n=== Résultats finaux ===")
print(f"Best eval AUC = {best_auc:.4f}")
print(f"Modèle sauvegardé → models/gru_best.pt")
print(f"Checkpoint        → models/gru_checkpoint.pt (résumable)")
print(f"Scalers           → models/gru_scalers.joblib")
