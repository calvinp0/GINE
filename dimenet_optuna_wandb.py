# dimenet_optuna_wandb_fixed.py  – fully self‑contained script
# Key fixes:
#   1. Robust handling when `input_type` or other optional keys are missing in
#      the saved config (avoids KeyError during final evaluation).
#   2. Robust locating of the split file even if it was not copied into the
#      trial directory.
#   3. Ensures the split file is copied into each trial directory so later
#      stages always find it.
#   4. Saves YAML with `sort_keys=False` so list keys such as `input_type` are
#      preserved exactly.
#   5. Minor hygiene: add `flush()` after YAML save, use `with` context for
#      file ops, and protect against empty glob matches.

import argparse, os, json, shutil, time, glob
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import optuna
import wandb
from torch.amp import autocast, GradScaler

from loss_utils import (
    von_mises_nll_fixed_kappa,
    angular_error,
)
from siamesepairwise import SiameseDimeNet, DimeNetPPEncoder
from schedulers import CosineRestartsDecay
from data import EquiMultiMolDataset
from split_utils import get_or_make_splits
from pytorch_lightning import seed_everything

###########################################################################
# -------------------------- helper utilities --------------------------- #
###########################################################################

def save_experiment(config, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = os.path.join(out_dir, f"run_{ts}.json")
    with open(fname, "w") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)
    print(f"👉  Saved experiment to {fname}")


def safe_get(d: dict, key: str, default):
    """Return d[key] if present else default (and warn once)."""
    if key not in d:
        print(f"⚠️  '{key}' missing in config – using default {default}")
    return d.get(key, default)

###########################################################################
# --------------------------- objective maker --------------------------- #
###########################################################################

def make_objective(config_path: str, disable_amp: bool, splits: dict, split_path: str):
    """Return an Optuna objective function bound to a particular YAML config."""

    def objective(trial):
        # ----------------------------------------------------------------------------
        # 1. Load and patch config ---------------------------------------------------
        # ----------------------------------------------------------------------------
        cfg = yaml.safe_load(open(config_path))

        # - wandb offline ensures no accidental online logging on cluster nodes
        os.environ.setdefault("WANDB_MODE", "offline")
        run = wandb.init(
            config=cfg,
            name=f"optuna_trial_{trial.number}",
            reinit=True,
            group="optuna_dimenet",
        )

        mcfg = cfg["model"]
        tcfg = cfg["training"]
        dcfg = cfg["data"]

        # ---------------- optuna suggestions ---------------
        mcfg.update(
            {
                "hidden_channels": trial.suggest_int("hidden_channels", 64, 1024),
                "out_channels": trial.suggest_int("out_channels", 64, 256),
                "num_blocks": trial.suggest_int("num_blocks", 2, 6),
                "num_spherical": trial.suggest_int("num_spherical", 3, 10),
                "num_radial": trial.suggest_int("num_radial", 3, 10),
                "cutoff": trial.suggest_float("cutoff", 3.0, 8.0),
                "fusion": trial.suggest_categorical("fusion", ["cat", "symm"]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            }
        )

        tcfg.update(
            {
                "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            }
        )
        tcfg["epochs"] = 200
        scfg = tcfg.setdefault("scheduler", {})
        scfg.update(
            {
                "T_0": trial.suggest_int("T_0", 5, 20),
                "T_mult": trial.suggest_int("T_mult", 1, 4),
                "eta_min": trial.suggest_float("eta_min", 1e-5, 1e-3),
                "decay": trial.suggest_float("decay", 0.1, 0.9),
            }
        )

        # ----------------------------------------------------------------------------
        # 2. Device & AMP -------------------------------------------------------------
        # ----------------------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda" and not disable_amp
        scaler = GradScaler(enabled=use_amp)
        print(f"Using device: {device} –  AMP {'ON' if use_amp else 'OFF'}")

        # ----------------------------------------------------------------------------
        # 3. Output directory ---------------------------------------------------------
        # ----------------------------------------------------------------------------
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(cfg["logging"]["output_root"], f"dihed_{ts}_trial{trial.number}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "used_config.yml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
            f.flush()

        # also copy the split file so downstream evaluation does not break
        shutil.copy(split_path, os.path.join(out_dir, "split_indices.json"))

        # ----------------------------------------------------------------------------
        # 4. Determinism -------------------------------------------------------------
        # ----------------------------------------------------------------------------
        seed_everything(tcfg["seed"])

        # ----------------------------------------------------------------------------
        # 5. Model -------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        encoder = DimeNetPPEncoder(
            hidden_channels=mcfg["hidden_channels"],
            out_channels=mcfg["out_channels"],
            num_blocks=mcfg["num_blocks"],
            num_spherical=mcfg["num_spherical"],
            num_radial=mcfg["num_radial"],
            cutoff=mcfg["cutoff"],
        )
        model = SiameseDimeNet(encoder, fusion=mcfg["fusion"], dropout=mcfg["dropout"]).to(device)

        # xavier on all eligible params
        def _init(m):
            if hasattr(m, "weight") and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
        model.apply(_init)

        # ----------------------------------------------------------------------------
        # 6. Optimiser / scheduler ----------------------------------------------------
        # ----------------------------------------------------------------------------
        opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
        scheduler = CosineRestartsDecay(opt, **scfg)

        # ----------------------------------------------------------------------------
        # 7. Data ---------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        dataset = EquiMultiMolDataset(**dcfg)
        train_ds = Subset(dataset, splits["train"])
        val_ds   = Subset(dataset, splits["val"])

        loaders = {
            "train": DataLoader(
                train_ds,
                batch_size=tcfg["batch_size"],
                shuffle=True,
                follow_batch=["z_s", "z_t"],
                generator=torch.Generator().manual_seed(tcfg["seed"]),
            ),
            "val": DataLoader(
                val_ds,
                batch_size=tcfg["batch_size"],
                shuffle=False,
                follow_batch=["z_s", "z_t"],
                generator=torch.Generator().manual_seed(tcfg["seed"]),
            ),
        }

        # ----------------------------------------------------------------------------
        # 8. Training loop ------------------------------------------------------------
        # ----------------------------------------------------------------------------
        best_val_err, bad_epochs = 1e9, 0
        loss_fn, metric_fn = von_mises_nll_fixed_kappa, angular_error

        for epoch in range(1, tcfg["epochs"] + 1):
            # ---- TRAIN ---------------------------------------------------------
            model.train()
            tot_loss = tot_err = tot_n = 0
            for batch in tqdm(loaders["train"], desc="Train", leave=False):
                batch = batch.to(device)
                target_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
                opt.zero_grad()
                with autocast(device_type=device.type, enabled=use_amp):
                    mu, kappa = model(batch)
                    loss = loss_fn(mu, target_angle, kappa=kappa)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                err = metric_fn(mu, target_angle) * 180 / np.pi
                bsz = batch.y.size(0)
                tot_loss += loss.item() * bsz
                tot_err += err.item() * bsz
                tot_n += bsz
            tr_l, tr_e = tot_loss / tot_n, tot_err / tot_n

            # ---- VALID ---------------------------------------------------------
            model.eval()
            tot_loss = tot_err = tot_n = 0
            with torch.no_grad():
                for batch in tqdm(loaders["val"], desc="Val", leave=False):
                    batch = batch.to(device)
                    target_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
                    mu, kappa = model(batch)
                    loss = loss_fn(mu, target_angle, kappa=kappa)
                    err = metric_fn(mu, target_angle) * 180 / np.pi
                    bsz = batch.y.size(0)
                    tot_loss += loss.item() * bsz
                    tot_err += err.item() * bsz
                    tot_n += bsz
            va_l, va_e = tot_loss / tot_n, tot_err / tot_n

            # ---- Optuna pruning ----------------------------------------------
            trial.report(va_e, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if va_e < best_val_err:
                best_val_err = va_e
                bad_epochs = 0
                torch.save(model.state_dict(), os.path.join(out_dir, "best_dimenet_model_manual.pt"))
            else:
                bad_epochs += 1
            if bad_epochs >= 20:
                raise optuna.TrialPruned()

            scheduler.step()
            print(
                f"Epoch {epoch:03d} | Train L={tr_l:.4f}, Err={tr_e:.1f}° | "
                f"Val L={va_l:.4f}, Err={va_e:.1f}° | LR={scheduler.get_last_lr()[0]:.2e}"
            )
            wandb.log({"epoch": epoch, "train_err": tr_e, "val_err": va_e, "lr": scheduler.get_last_lr()[0]})

        trial.set_user_attr("val_err", best_val_err)
        wandb.finish()

        # clear GPU fragmentation
        torch.cuda.empty_cache()
        return best_val_err

    return objective

###########################################################################
# ------------------------------ main ----------------------------------- #
###########################################################################

if __name__ == "__main__":
    start = time.time()

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    # --- load config & build splits once ----------------------------------
    base_cfg = yaml.safe_load(open(args.config))
    dataset_once = EquiMultiMolDataset(**base_cfg["data"])
    split_path = os.path.join(base_cfg["logging"]["output_root"], "split_indices.json")
    splits = get_or_make_splits(dataset_once, base_cfg["training"]["seed"], split_path)
    del dataset_once  # free RAM

    # --- set up Optuna study ---------------------------------------------
    objective = make_objective(args.config, args.no_amp, splits, split_path)
    study = optuna.create_study(
        study_name="dimenet_hpo",
        direction="minimize",
        storage="sqlite:///dimenet_hpo.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15),
    )
    study.optimize(objective, n_trials=args.trials)

    # ---------------------------------------------------------------------
    #  Final evaluation on best trial -------------------------------------
    # ---------------------------------------------------------------------
    elapsed = time.time() - start
    print(f"Total wall time: {elapsed/60:.1f} min")

    # locate best dir (retry‑safe)
    pattern = os.path.join(base_cfg["logging"]["output_root"], f"dihed_*_trial{study.best_trial.number}")
    trial_dirs = sorted(glob.glob(pattern))
    if not trial_dirs:
        raise RuntimeError("Could not locate best trial directory with pattern: " + pattern)
    best_dir = trial_dirs[-1]

    cfg_trial = yaml.safe_load(open(os.path.join(best_dir, "used_config.yml")))
    mcfg = cfg_trial["model"]
    dcfg = cfg_trial["data"]
    tcfg = cfg_trial["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rebuild model
    encoder = DimeNetPPEncoder(
        hidden_channels=mcfg["hidden_channels"],
        out_channels=mcfg["out_channels"],
        num_blocks=mcfg["num_blocks"],
        num_spherical=mcfg["num_spherical"],
        num_radial=mcfg["num_radial"],
        cutoff=mcfg["cutoff"],
    )
    model = SiameseDimeNet(encoder, fusion=mcfg["fusion"], dropout=mcfg["dropout"]).to(device)
    model.load_state_dict(torch.load(os.path.join(best_dir, "best_dimenet_model_manual.pt"), map_location=device))
    model.eval()

    # ensure we can always resolve split file
    split_file = os.path.join(best_dir, "split_indices.json")
    if not os.path.exists(split_file):
        split_file = split_path
    with open(split_file) as f:
        splits_final = json.load(f)

    dataset = EquiMultiMolDataset(
        root=dcfg["root"],
        sdf_folder=dcfg["sdf_folder"],
        target_csv=dcfg["target_csv"],
        input_type=safe_get(dcfg, "input_type", ["r1h", "r2h"]),
        target_columns=dcfg["target_columns"],
        keep_hs=safe_get(dcfg, "keep_hs", True),
        sanitize=safe_get(dcfg, "sanitize", False),
        force_reload=safe_get(dcfg, "force_reload", False),
    )
    test_loader = DataLoader(
        Subset(dataset, splits_final["test"]),
        batch_size=tcfg["batch_size"],
        shuffle=False,
        follow_batch=["z_s", "z_t"],
    )

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch = batch.to(device)
            mu, kappa = model(batch)
            true_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
            for m, k, t in zip(mu.cpu().numpy(), kappa.cpu().numpy(), true_angle.cpu().numpy()):
                results.append({"pred_angle_rad": float(m), "pred_kappa": float(k), "true_angle_rad": float(t)})

    res_df = pd.DataFrame(results)
    res_df.to_csv("test_predictions_with_uncertainty.csv", index=False)
    print("✅  Saved test predictions → test_predictions_with_uncertainty.csv")

    # quick summary prints (no wandb in final stage to keep cluster logs clean)
    kappa_vals = res_df["pred_kappa"].values
    errs = np.abs(np.arctan2(np.sin(res_df["pred_angle_rad"] - res_df["true_angle_rad"]),
                             np.cos(res_df["pred_angle_rad"] - res_df["true_angle_rad"]))) * 180 / np.pi
    print(f"Mean kappa = {kappa_vals.mean():.2f}, mean error = {errs.mean():.2f}°")

    print("Best hyper‑params:", study.best_trial.params)
