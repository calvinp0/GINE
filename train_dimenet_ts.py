#!/usr/bin/env python
"""
DimeNet++ training script with
  • **offset‑tag** injection already handled inside `EquiDataset`
  • **warm‑up stage** (raw output + MSE loss, no unit‑circle normalisation)
  • **gradient‑clipping**
Run:  python train_dimenet_warmup.py --help
"""
import argparse, math, csv, random, os
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# local imports
from siamesepairwise import DimeNet       # <‑ head & encoder wrapper
from dimenet import DimeNetPPEncoder      # <‑ DimeNet++ graph encoder
from data import EquiDataset              # <‑ single‑graph dataset w/ offset tagging
from loss_utils import cosine_angle_loss, AngularErrorMetric
from schedulers import CosineRestartsDecay

# ---------------------------------------------------------------------------
# helpers & utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader(dataset, batch, shuffle):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True)

# ---------------------------------------------------------------------------
# run‑epoch helpers (supporting warm‑up logic)
# ---------------------------------------------------------------------------

def _forward_unit(model, batch):
    """Standard forward path – returns **normalised** sin/cos."""
    return model(batch)

def _forward_raw(model, batch):
    """Bypass final F.normalize for warm‑up (raw head output)."""
    h = model.encode(batch)
    raw = model.head(h)             # shape [B,2]
    return raw

def run_epoch(model, loader, loss_fn, metric_fn, *, optimizer=None,
              scaler=None, use_amp=False, forward_fn=_forward_unit,
              grad_clip: float | None = None, device="cpu") -> Tuple[float,float]:
    """Shared loop for train / eval."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    tot_loss = tot_err = n = 0
    ctx_outer = nullcontext() if train_mode else torch.no_grad()
    with ctx_outer:
        for batch in loader:
            batch = batch.to(device)
            with (torch.cuda.amp.autocast() if use_amp else nullcontext()):
                out   = forward_fn(model, batch)
                loss  = loss_fn(out, batch.y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            err      = metric_fn(out, batch.y)
            bs       = batch.num_graphs
            tot_loss += loss.item()*bs
            tot_err  += err.item()*bs
            n        += bs

    return tot_loss/n, tot_err/n

# ---------------------------------------------------------------------------
# main – argument parsing + full training loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DimeNet++ dihedral training (with warm‑up)")
    ap.add_argument("--root", default="data/equi")
    ap.add_argument("--geom_csv", required=True)
    ap.add_argument("--sdf_dir", required=True)
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup_epochs", type=int, default=10,
                    help="epochs to train with raw output + MSE loss before switching to cosine loss")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="enable CUDA AMP")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    outdir = Path(args.outdir)
    (outdir/"checkpoints").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Dataset (offset tags handled inside EquiDataset!)
    # ---------------------------------------------------------------------
    ds = EquiDataset(
        root=args.root,
        geoms_csv=args.geom_csv,
        sdf_folder=args.sdf_dir,
        target_csv=args.target_csv,
        target_columns=["psi_1_dihedral_sin", "psi_1_dihedral_cos"],
        force_reload=False,
    )
    n_train = int(0.8*len(ds))
    n_val   = int(0.1*len(ds))
    n_test  = len(ds) - n_train - n_val
    train_ds, val_ds, _ = torch.utils.data.random_split(ds, [n_train, n_val, n_test])

    train_ld = make_loader(train_ds, args.batch, True)
    val_ld   = make_loader(val_ds,   args.batch, False)

    # ---------------------------------------------------------------------
    # Model (depth‑12, width‑256 default)
    # ---------------------------------------------------------------------
    encoder = DimeNetPPEncoder(hidden_channels=args.hidden,
                               out_channels=args.hidden,
                               num_blocks=args.blocks,
                               dropout=0.1).to(device)
    model   = DimeNet(encoder=encoder, dropout=0.2, head_hidden_dims=[128,128]).to(device)

    # ---------------------------------------------------------------------
    # Optimiser / scheduler / scaler
    # ---------------------------------------------------------------------
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineRestartsDecay(opt, T_0=20, T_mult=2, eta_min=1e-4, decay=0.3)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.startswith("cuda")) else None

    cosine_loss = cosine_angle_loss
    mse_loss    = torch.nn.MSELoss()
    metric_fn   = AngularErrorMetric(in_degrees=True)

    best_val = math.inf
    for epoch in range(1, args.epochs+1):
        warm = epoch <= args.warmup_epochs

        fwd = _forward_raw if warm else _forward_unit
        loss_fn = mse_loss if warm else cosine_loss

        tr_loss, tr_err = run_epoch(model, train_ld, loss_fn, metric_fn,
                                    optimizer=opt, scaler=scaler, use_amp=args.amp,
                                    forward_fn=fwd, grad_clip=args.grad_clip, device=device)
        vl_loss, vl_err = run_epoch(model, val_ld, loss_fn, metric_fn,
                                    optimizer=None, scaler=None, use_amp=False,
                                    forward_fn=fwd, device=device)
        sched.step()
        lr_now = sched.get_last_lr()[0]

        phase = "(warm‑up)" if warm else "(main)  "
        print(f"Epoch {epoch:03d}{phase} | "
              f"Train L={tr_loss:.4f}, Err={tr_err:.1f}° | "
              f"Val L={vl_loss:.4f}, Err={vl_err:.1f}° | "
              f"LR={lr_now:.2e}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), outdir/"checkpoints/best.pt")
            print("  ↳ saved new best model")

if __name__ == "__main__":
    main()
