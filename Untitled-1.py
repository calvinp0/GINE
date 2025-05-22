# %%
from data import MultiMolGraphDataset, EquiDataset
from torch_geometric.loader import DataLoader
import torch
import torchmetrics
from torchmetrics import MeanAbsoluteError
import random
import pandas as pd
import numpy as np
from siamesepairwise import SiameseDimeNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from schedulers import CosineRestartsDecay
import os
import argparse
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm  # for progress bars
from loss_utils import cosine_angle_loss, AngularErrorMetric


device = 'cpu'

# %%
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%
equi_data = EquiDataset(
    root='data/equi',
    geoms_csv='/home/calvin/code/GINE/rxn_geometries.csv',
    target_csv='/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_data/target_data_sin_cos.csv',
    target_columns=['psi_1_dihedral_sin', 'psi_1_dihedral_cos'],
    transform=None,
    pre_transform=None,
    pre_filter=None,
    force_reload=True
)

# %%
from dimenet import DimeNetPPEncoder

encoder = DimeNetPPEncoder(hidden_channels=128,
                           dropout=0.2)
encoder = encoder.to(device)

# %%


# %%
from torch_geometric.loader import DataLoader
# Random split using torch
train_size = int(0.8 * len(equi_data))
valid_size = int(0.1 * len(equi_data))
test_size = len(equi_data) - train_size - valid_size
# Shuffle the dataset
train_data, valid_data, test_data = torch.utils.data.random_split(equi_data, [train_size, valid_size, test_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# %%
def train_epoch(model, loader, optimizer, loss_fn, metric_fn):
    model.train()
    total_loss = 0.0
    total_err  = 0.0
    n_samples  = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # forward + loss
        out   = model(batch)              # [B,2]

        loss  = loss_fn(out, batch.y)     # scalar
        loss.backward()
        optimizer.step()

        # metric: mean abs angle error (degrees)
        err = metric_fn(out, batch.y)     # scalar tensor

        bsize = batch.y.size(0)
        total_loss += loss.item() * bsize
        total_err  += err.item()  * bsize
        n_samples  += bsize

    return total_loss / n_samples, total_err / n_samples

def eval_epoch(model, loader, loss_fn, metric_fn):
    model.eval()
    total_loss = 0.0
    total_err  = 0.0
    n_samples  = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = batch.to(device)
            out   = model(batch)
            loss  = loss_fn(out, batch.y)
            err   = metric_fn(out, batch.y)

            bsize = batch.y.size(0)
            total_loss += loss.item() * bsize
            total_err  += err.item()  * bsize
            n_samples  += bsize

    return total_loss / n_samples, total_err / n_samples


# %%
from siamesepairwise import DimeNet

model = DimeNet(
    encoder=encoder,
    dropout=0.2,
    head_hidden_dims=[128, 128],
)
model = model.to(device)
num_epochs  = 200
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5, verbose=True
# )
#scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
scheduler = CosineRestartsDecay(
    optimizer,
    T_0     = 20,
    T_mult  = 2,
    eta_min = 1e-4,
    decay   = 0.3) 


import torch.nn.functional as F

best_val_loss = float('inf')

loss_fn = cosine_angle_loss
metric_fn   = AngularErrorMetric(in_degrees=True)

# %%
for epoch in range(1, num_epochs + 1):
    tr_loss, tr_err = train_epoch(model, train_loader, optimizer, loss_fn, metric_fn)
    va_loss, va_err = eval_epoch(model, valid_loader, loss_fn, metric_fn)

    # Step the scheduler once per epoch
    # If your scheduler wants the epoch number, do: scheduler.step(epoch)
    # Otherwise just:
    scheduler.step()

    lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch:02d} | "
          f"Train L={tr_loss:.4f}, Err={tr_err:.1f}° | "
          f"Val   L={va_loss:.4f}, Err={va_err:.1f}° | "
          f"LR={lr:.2e}")

    # (Optional) save best‐model checkpoint
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        torch.save(model.state_dict(), 'best_dimenet_model.pt')
        print(" ↳ New best model saved!")

# %%
for batch in train_loader:
    batch = batch.to(device)
    print(batch.y)
    break

# %%
torch.cuda.empty_cache()
print("CUDA memory cleared")
# Print CUDA memory summary



