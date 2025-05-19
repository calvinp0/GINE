import argparse, os, json
from datetime import datetime
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from loss_utils import cosine_angle_loss, AngularErrorMetric, von_mises_nll_fixed_kappa, angular_error
from siamesepairwise import SiameseDimeNet, DimeNetPPEncoder
from schedulers import CosineRestartsDecay
from data import EquiMultiMolDataset
from pytorch_lightning import seed_everything  # or roll your own
import optuna
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import torch.nn as nn
import time


def save_experiment(config, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = os.path.join(out_dir, f"run_{ts}.json")
    with open(fname, "w") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)
    print(f"👉 Saved experiment to {fname}")


def make_objective(config_path, disable_amp):
    """
    Create an Optuna objective function that loads config from the given path for each trial.
    """
    def objective(trial):
        # WandB offline mode and init per trial
        os.environ.setdefault("WANDB_MODE", "offline")
        # Load config per trial
        cfg = yaml.safe_load(open(config_path))
        run = wandb.init(
            config=cfg,
            name=f"optuna_trial_{trial.number}",
            reinit=True,
            group="optuna_dimenet"
        )
        # Sync Optuna trial hyperparams into WandB config
        wandb.config.update({k: v for k, v in cfg.items()})

        # Suggest hyperparameters from config sections
        mcfg = cfg["model"]
        mcfg["hidden_channels"] = trial.suggest_int("hidden_channels", 64, 1024)
        mcfg["out_channels"] = trial.suggest_int("out_channels", 64, 256)
        mcfg["num_blocks"] = trial.suggest_int("num_blocks", 2, 6)
        mcfg["num_spherical"] = trial.suggest_int("num_spherical", 3, 10)
        mcfg["num_radial"] = trial.suggest_int("num_radial", 3, 10)
        mcfg["cutoff"] = trial.suggest_float("cutoff", 3.0, 8.0)
        mcfg["fusion"] = trial.suggest_categorical("fusion", ["cat", "symm"])
        mcfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

        tcfg = cfg["training"]
        tcfg["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        tcfg["weight_decay"] = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
        tcfg["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
        tcfg["epochs"] = 200

        scfg = tcfg.get("scheduler", {})
        scfg["T_0"] = trial.suggest_int("T_0", 5, 20)
        scfg["T_mult"] = trial.suggest_int("T_mult", 1, 4)
        scfg["eta_min"] = trial.suggest_float("eta_min", 1e-5, 1e-3)
        scfg["decay"] = trial.suggest_float("decay", 0.1, 0.9)
        tcfg["scheduler"] = scfg

        # ── device setup ─────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # ── AMP setup ──────────────────────────────────────
        # allow disabling AMP via CLI flag
        use_amp = (device.type == "cuda") and not disable_amp
        scaler = GradScaler(enabled=use_amp)
        print(f"{'🟢' if use_amp else '🔴'} AMP {'enabled' if use_amp else 'disabled'}")

        # ── make output dir & snapshot config ─────────────
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"dihed_{ts}_trial{trial.number}"
        out_root = cfg["logging"]["output_root"]
        out_dir = os.path.join(out_root, exp_name)
        os.makedirs(out_dir, exist_ok=True)
        yaml.safe_dump(cfg, open(os.path.join(out_dir, "used_config.yml"), "w"))

        # ── set seed ──────────────────────────────────────
        seed_everything(cfg["training"]["seed"])

        # ── build model ───────────────────────────────────
        encoder = DimeNetPPEncoder(
            hidden_channels=mcfg["hidden_channels"],
            out_channels=mcfg["out_channels"],
            num_blocks=mcfg["num_blocks"],
            num_spherical=mcfg["num_spherical"],
            num_radial=mcfg["num_radial"],
            cutoff=mcfg["cutoff"],
        )
        model = SiameseDimeNet(
            encoder=encoder,
            fusion=mcfg["fusion"],
            dropout=mcfg["dropout"],
        ).to(device)
        # Apply Xavier initialization to all parameters
        def init_weights(m):
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        model.apply(init_weights)

        # ── optimizer & scheduler ────────────────────────
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(tcfg["lr"]),
            weight_decay=float(tcfg["weight_decay"]),
        )
        scheduler = CosineRestartsDecay(
            opt,
            T_0=int(scfg["T_0"]),
            T_mult=int(scfg["T_mult"]),
            eta_min=float(scfg["eta_min"]),
            decay=float(scfg["decay"]),
        )

        # ── data loaders ──────────────────────────────────
        dcfg = cfg["data"]
        os.makedirs(dcfg["root"], exist_ok=True)

        dataset = EquiMultiMolDataset(
            root=dcfg["root"],
            sdf_folder=dcfg["sdf_folder"],
            target_csv=dcfg["target_csv"],
            input_type=dcfg["input_types"],
            target_columns=dcfg["target_columns"],
            keep_hs=dcfg["keep_hs"],
            sanitize=dcfg["sanitize"],
            force_reload=dcfg["force_reload"],
        )

        n = len(dataset)
        g = torch.Generator().manual_seed(cfg["training"]["seed"])
        t, v, te = int(0.8 * n), int(0.1 * n), n - int(0.8 * n) - int(0.1 * n)
        train_ds, val_ds, test_ds = random_split(dataset, [t, v, te], generator=g)

        loaders = {
            "train": DataLoader(
                train_ds,
                batch_size=tcfg["batch_size"],
                shuffle=True,
                follow_batch=["z_s", "z_t"],
                generator=g,
            ),
            "val": DataLoader(
                val_ds,
                batch_size=tcfg["batch_size"],
                shuffle=False,
                follow_batch=["z_s", "z_t"],
                generator=g,
            ),
            "test": DataLoader(
                test_ds,
                batch_size=tcfg["batch_size"],
                shuffle=False,
                follow_batch=["z_s", "z_t"],
                generator=g,
            ),
        }

        print(
            f"📊 Dataset split: total={len(dataset)} | "
            f"train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}"
        )

        # ── loss + metric ─────────────────────────────────
        loss_fn = von_mises_nll_fixed_kappa
        metric_fn = angular_error

        # ── train/val loop ────────────────────────────────
        best_val_err = 1e9
        for epoch in range(1, tcfg["epochs"] + 1):
            # train
            model.train()
            tot_loss, tot_err, count = 0, 0, 0
            for batch in tqdm(loaders["train"], desc="Training", leave=False):
                batch = batch.to(device)
                opt.zero_grad()

                if use_amp:
                    with autocast(enabled=False, device_type=device.type):
                        h_s, h_t = model.encode(batch)

                    with autocast(device_type=device.type):
                        h_fused = model.fuse(h_s, h_t)
                        mu, kapp = model.head_mu_kappa(h_fused)
                        target_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
                        loss = loss_fn(mu=mu, target=target_angle, kappa=kapp)

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    target_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
                    mu, kappa = model(batch)
                    loss = loss_fn(mu=mu, target=target_angle, kappa=kappa)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                err = metric_fn(mu, target_angle) * 180.0 / torch.pi
                b = batch.y.size(0)
                tot_loss += loss.item() * b
                tot_err += err.item() * b
                count += b
                
                # Optional: print angles for debugging
                angle1 = torch.atan2(batch.y[:, 0], batch.y[:, 1])  # your code now
                angle2 = torch.atan2(batch.y[:, 1], batch.y[:, 0])  # swapped

                print('Sanity check: angle1[0]', angle1[0], 'angle2[0]', angle2[0])

            tr_l, tr_e = tot_loss / count, tot_err / count
            # Log training error as user attr
            trial.set_user_attr("train_err", tr_e)

            # val
            model.eval()
            tot_loss, tot_err, count = 0, 0, 0
            with torch.no_grad():
                for batch in tqdm(loaders["val"], desc="Validation", leave=False):
                    batch = batch.to(device)
                    target_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
                    mu, kappa = model(batch)
                    l = loss_fn(mu=mu, target=target_angle, kappa=kappa)
                    err = metric_fn(mu, target_angle) * 180.0 / torch.pi
                    b = batch.y.size(0)
                    tot_loss += l.item() * b
                    tot_err += err.item() * b
                    count += b
            va_l, va_e = tot_loss / count, tot_err / count
            # Log validation loss as user attr
            trial.set_user_attr("val_loss", va_l)

            # scheduler step
            scheduler.step()
            print(
                f"Epoch {epoch:03d} | "
                f"Train L={tr_l:.4f}, Err={tr_e:.1f}° | "
                f"Val   L={va_l:.4f}, Err={va_e:.1f}° | "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )

            # Log metrics to WandB
            wandb.log({
                "epoch": epoch,
                "train_loss": tr_l,
                "train_err": tr_e,
                "val_loss": va_l,
                "val_err": va_e,
                "lr": scheduler.get_last_lr()[0]
            })

            if va_e < best_val_err:
                best_val_err = va_e
                torch.save(
                    model.state_dict(),
                    os.path.join(out_dir, "best_dimenet_model_manual.pt"),
                )
                print(" ↳ New best model saved!")
                # Save trial parameters for this trial
                with open(os.path.join(out_dir, "trial_params.json"), "w") as f:
                    json.dump(trial.params, f, indent=2)
                # Save model artifact to WandB
                wandb.save(os.path.join(out_dir, "best_dimenet_model_manual.pt"))

        trial.set_user_attr("val_err", va_e)
        # Finish WandB run for this trial
        wandb.finish()
        return va_e
    return objective


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even if CUDA is available")
    args = parser.parse_args()

    # Use a closure to pass config path and amp flag into the objective
    objective_fn = make_objective(args.config, args.no_amp)

    # Create a shared study for parallel executions
    study = optuna.create_study(
        study_name="dimenet_hpo",
        direction="minimize",
        storage="sqlite:///dimenet_hpo.db",
        load_if_exists=True,
    )

    # Optimize using the closure-based objective
    study.optimize(objective_fn, n_trials=args.trials)
    
    # Timing here
    end_time = time.time()
    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec / 60
    elapsed_hr  = elapsed_sec / 3600
    n_trials_run = len(study.trials)
    print(f"\n🕒 Total elapsed time: {elapsed_sec:.1f} s  ({elapsed_min:.2f} min, {elapsed_hr:.2f} hr)")
    print(f"Average time per trial: {elapsed_sec/n_trials_run:.1f} s  ({elapsed_min/n_trials_run:.2f} min)")
    print("Best trial:", study.best_trial.params)

    # Save a DataFrame of all trial results
    study.trials_dataframe().to_csv("optuna_results.csv")

    print("Best trial:", study.best_trial.params)
