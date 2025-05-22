#!/usr/bin/env python
"""
Single-GPU (or CPU) DimeNet training script.
Run with:  python train_dimenet.py --help
"""
import argparse, os, json, random, time, csv, math
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from siamesepairwise import DimeNet
from dimenet import DimeNetPPEncoder
from data import EquiDataset
from loss_utils import cosine_angle_loss, AngularErrorMetric
from schedulers import CosineRestartsDecay
from pathlib import Path
from tqdm import tqdm

# ---------- helpers ----------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader(dataset, batch, shuffle):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True)

def amp_context(enable_cuda_amp):
    return (torch.cuda.amp.autocast() if enable_cuda_amp else torch.autocast("cpu", enabled=False))

# ---------- training + evaluation ----------
def run_epoch(model, loader, loss_fn, metric_fn, optimizer=None, scaler=None, amp=False, device="cpu"):
    train = optimizer is not None
    total_loss = total_err = n = 0
    model.train() if train else model.eval()

    ctx = torch.no_grad() if not train else nullcontext()
    with ctx:
        for batch in tqdm(loader, leave=False, desc="Train" if train else "Valid"):
            batch = batch.to(device)
            use_amp = (amp and device.startswith("cuda"))
            with (torch.cuda.amp.autocast() if use_amp else nullcontext()):
                out = model(batch)
                loss = loss_fn(out, batch.y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            err = metric_fn(out, batch.y)
            bsz = batch.num_graphs
            total_loss += loss.item() * bsz
            total_err  += err.item()   * bsz
            n += bsz
    return total_loss / n, total_err / n

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",   default="data/equi")
    p.add_argument("--geom_csv", required=True)
    p.add_argument("--sdf_dir",  required=True)
    p.add_argument("--target_csv", required=True)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--amp",    action="store_true")         # enable CUDA AMP
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--blocks", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir(exist_ok=True)

    # dataset
    ds = EquiDataset(
        root=args.root,
        geoms_csv=args.geom_csv,
        sdf_folder=args.sdf_dir,
        target_csv=args.target_csv,
        target_columns=["psi_1_dihedral_sin", "psi_1_dihedral_cos"],
        force_reload=False,
    )
    train_len = int(0.8 * len(ds))
    valid_len = int(0.1 * len(ds))
    test_len  = len(ds) - train_len - valid_len
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [train_len, valid_len, test_len])

    train_ld = make_loader(train_ds, args.batch, True)
    valid_ld = make_loader(valid_ds, args.batch, False)

    encoder = DimeNetPPEncoder(
        hidden_channels=args.hidden,
        out_channels=args.hidden,
        num_blocks=args.blocks,
        dropout=0.1
    ).to(device)
    model = DimeNet(encoder=encoder, dropout=0.2, head_hidden_dims=[128, 128]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineRestartsDecay(opt, T_0=20, T_mult=2, eta_min=1e-4, decay=0.3)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.startswith("cuda") else None

    loss_fn   = cosine_angle_loss
    metric_fn = AngularErrorMetric(in_degrees=True)

    # CSV logger
    csv_path = outdir / "metrics.csv"
    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["epoch", "train_loss", "train_err_deg", "val_loss", "val_err_deg", "lr"])

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_err = run_epoch(model, train_ld, loss_fn, metric_fn,
                                    optimizer=opt, scaler=scaler, amp=args.amp, device=device)
        vl_loss, vl_err = run_epoch(model, valid_ld, loss_fn, metric_fn,
                                    optimizer=None, scaler=None, amp=False, device=device)
        sched.step()
        lr_now = sched.get_last_lr()[0]

        print(f"Epoch {epoch:03d} | "
              f"Train L={tr_loss:.4f}, Err={tr_err:.1f}° | "
              f"Val L={vl_loss:.4f}, Err={vl_err:.1f}° | "
              f"LR={lr_now:.2e}", flush=True)

        with open(csv_path, "a", newline="") as fp:
            csv.writer(fp).writerow([epoch, tr_loss, tr_err, vl_loss, vl_err, lr_now])

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), outdir / "checkpoints/best.pt")
            print("  ↳ saved new best model", flush=True)

    print("Training complete. Metrics in", csv_path, flush=True)

if __name__ == "__main__":
    from contextlib import nullcontext   # py3.9+
    main()
