import argparse, os, json
from datetime import datetime
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
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
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt


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
        # Save split indices for consistent test-time loading
        splits = {
            "train": train_ds.indices,
            "val": val_ds.indices,
            "test": test_ds.indices
        }
        with open(os.path.join(out_dir, "split_indices.json"), "w") as f:
            json.dump(splits, f)

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
        initial_penalty = 0.2
        final_penalty = 0.0
        num_anneal_epochs = 15
        for epoch in range(1, tcfg["epochs"] + 1):
            # train
            train_kappas = []  # collect per-sample train kappa values
            train_errors = []  # collect per-sample train angular errors
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
                        # Apply penalty to kappa
                        penalty_weight = max(
                            initial_penalty * (1 - (epoch / num_anneal_epochs)),
                            final_penalty,
                        )
                        kappa_penalty = penalty_weight * torch.mean(torch.relu(kappa-2))
                        loss = loss_fn(mu=mu, target=target_angle, kappa=kapp) + kappa_penalty

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    target_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
                    mu, kappa = model(batch)
                    # Apply penalty to kappa
                    penalty_weight = max(
                        initial_penalty * (1 - (epoch / num_anneal_epochs)),
                        final_penalty,
                    )
                    kappa_penalty = penalty_weight * torch.mean(torch.relu(kappa-2))
                    loss = loss_fn(mu=mu, target=target_angle, kappa=kappa) + kappa_penalty
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                err = metric_fn(mu, target_angle) * 180.0 / torch.pi
                b = batch.y.size(0)
                tot_loss += loss.item() * b
                tot_err += err.item() * b
                count += b
                
                # record per-sample uncertainties and errors
                err_samples = torch.abs(torch.atan2(torch.sin(mu - target_angle), torch.cos(mu - target_angle))) * 180.0 / torch.pi
                train_kappas.extend(kappa.cpu().detach().numpy().tolist())
                train_errors.extend(err_samples.cpu().detach().numpy().tolist())
                
            tr_l, tr_e = tot_loss / count, tot_err / count
            # Log training error as user attr
            trial.set_user_attr("train_err", tr_e)

            # val
            val_kappas = []  # collect per-sample validation kappa values
            val_errors = []  # collect per-sample validation angular errors
            model.eval()
            tot_loss, tot_err, count = 0, 0, 0
            with torch.no_grad():
                for batch in tqdm(loaders["val"], desc="Validation", leave=False):
                    batch = batch.to(device)
                    target_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
                    mu, kappa = model(batch)
                    val_kappas.extend(kappa.cpu().numpy().tolist())
                    # record per-sample uncertainties and errors
                    err_samples_val = torch.abs(torch.atan2(torch.sin(mu - target_angle), torch.cos(mu - target_angle))) * 180.0 / torch.pi
                    val_errors.extend(err_samples_val.cpu().numpy().tolist())
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

            # Compute kappa statistics
            train_kappa_arr = np.array(train_kappas)
            val_kappa_arr = np.array(val_kappas)
            train_kappa_mean = train_kappa_arr.mean()
            train_kappa_median = np.median(train_kappa_arr)
            train_kappa_iqr = np.percentile(train_kappa_arr, [25, 75])
            train_kappa_min, train_kappa_max = train_kappa_arr.min(), train_kappa_arr.max()
            val_kappa_mean = val_kappa_arr.mean()
            val_kappa_median = np.median(val_kappa_arr)
            val_kappa_iqr = np.percentile(val_kappa_arr, [25, 75])
            val_kappa_min, val_kappa_max = val_kappa_arr.min(), val_kappa_arr.max()

            # Log metrics to WandB
            wandb.log({
                "epoch": epoch,
                "train_loss": tr_l,
                "train_err": tr_e,
                "val_loss": va_l,
                "val_err": va_e,
                "lr": scheduler.get_last_lr()[0],
                # Histograms
                "train_kappa_hist": wandb.Histogram(train_kappas),
                "val_kappa_hist": wandb.Histogram(val_kappas),
                # Summary stats
                "train/kappa_mean": train_kappa_mean,
                "train/kappa_median": train_kappa_median,
                "train/kappa_25%": train_kappa_iqr[0],
                "train/kappa_75%": train_kappa_iqr[1],
                "train/kappa_min": train_kappa_min,
                "train/kappa_max": train_kappa_max,
                "val/kappa_mean": val_kappa_mean,
                "val/kappa_median": val_kappa_median,
                "val/kappa_25%": val_kappa_iqr[0],
                "val/kappa_75%": val_kappa_iqr[1],
                "val/kappa_min": val_kappa_min,
                "val/kappa_max": val_kappa_max,
            })

            # Scatter plot of kappa vs error
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(train_kappas, train_errors, alpha=0.3)
            ax.set_xscale('log')
            ax.set_xlabel('Train Kappa')
            ax.set_ylabel('Train Error (deg)')
            ax.set_title(f'Epoch {epoch}: Kappa vs Error')
            wandb.log({"train/kappa_error_scatter": wandb.Image(fig)})
            plt.close(fig)

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

    # After Optuna study is done
    print("Loading best model for uncertainty reporting...")
    # load main config
    cfg = yaml.safe_load(open(args.config))
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # locate best trial run directory
    trial_dirs = glob.glob(os.path.join(cfg["logging"]["output_root"], f"dihed_*_trial{study.best_trial.number}"))
    best_dir = sorted(trial_dirs)[-1]
    # load trial-specific config
    cfg_trial = yaml.safe_load(open(os.path.join(best_dir, "used_config.yml")))
    # build model from best config
    mcfg = cfg_trial["model"]
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
    # load best model weights
    best_model_path = os.path.join(best_dir, "best_dimenet_model_manual.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    # prepare test loader
    dcfg = cfg_trial["data"]
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
    g = torch.Generator().manual_seed(cfg_trial["training"]["seed"])
    # Load saved split indices for consistent test set
    with open(os.path.join(best_dir, "split_indices.json"), "r") as f:
        splits = json.load(f)
    test_ds = Subset(dataset, splits["test"])
    test_loader = DataLoader(test_ds,
                            batch_size=cfg_trial["training"]["batch_size"],
                            shuffle=False,
                            follow_batch=["z_s", "z_t"],
                            generator=g)
    loaders = {"test": test_loader}

    results = []
    with torch.no_grad():
        for batch in tqdm(loaders["test"], desc="Test evaluation"):
            batch = batch.to(device)
            mu, kappa = model(batch)
            # mu is predicted angle (in radians), kappa is uncertainty
            true_angle = torch.atan2(batch.y[:, 0], batch.y[:, 1])
            for m, k, t in zip(mu.cpu().numpy(), kappa.cpu().numpy(), true_angle.cpu().numpy()):
                results.append({"pred_angle_rad": m, "pred_kappa": k, "true_angle_rad": t})

    results_df = pd.DataFrame(results)
    results_df.to_csv("test_predictions_with_uncertainty.csv", index=False)
    print("Saved uncertainty predictions to test_predictions_with_uncertainty.csv")

    # Compute and log kappa uncertainty summary to WandB
    kappa_vals = results_df["pred_kappa"].values
    angle_errs = np.abs(np.arctan2(np.sin(results_df["pred_angle_rad"] - results_df["true_angle_rad"]),
                                np.cos(results_df["pred_angle_rad"] - results_df["true_angle_rad"])))
    mean_kappa = np.mean(kappa_vals)
    median_kappa = np.median(kappa_vals)
    iqr_kappa = np.percentile(kappa_vals, [25, 75])
    mean_error_deg = np.mean(angle_errs) * 180 / np.pi

    print(f"Mean kappa: {mean_kappa:.2f} (lower = more uncertainty)")
    print(f"Median kappa: {median_kappa:.2f}, IQR: {iqr_kappa[0]:.2f} - {iqr_kappa[1]:.2f}")
    print(f"Mean angular error: {mean_error_deg:.2f}°")

    # Optional: Scatter plot of Kappa vs Error
    plt.figure(figsize=(10, 6))
    plt.scatter(kappa_vals, angle_errs * 180 / np.pi, alpha=0.5)
    plt.xscale('log')
    plt.xlabel("Predicted Kappa (Uncertainty)")
    plt.ylabel("Angular Error (degrees)")
    plt.title("Kappa vs Angular Error")
    plt.grid(True)
    plt.savefig("kappa_vs_error_scatter.png")
    plt.close()

    wandb.log({
        "test/kappa_mean": mean_kappa,
        "test/kappa_median": median_kappa,
        "test/kappa_25%": iqr_kappa[0],
        "test/kappa_75%": iqr_kappa[1],
        "test/mean_angle_error_deg": mean_error_deg,
        "test/kappa_hist": wandb.Histogram(kappa_vals),
        "test/angle_error_hist": wandb.Histogram(angle_errs * 180 / np.pi),
    })
