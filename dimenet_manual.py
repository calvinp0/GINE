import argparse, os, json
from datetime import datetime
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from loss_utils import cosine_angle_loss, AngularErrorMetric
from siamesepairwise import SiameseDimeNet, DimeNetPPEncoder
from schedulers import CosineRestartsDecay
from data import EquiMultiMolDataset
from pytorch_lightning import seed_everything  # or roll your own
import wandb
from torch.amp import autocast, GradScaler
from tqdm import tqdm



def save_experiment(config, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = os.path.join(out_dir, f"run_{ts}.json")
    with open(fname, "w") as f:
        json.dump({"config":config, "metrics":metrics}, f, indent=2)
    print(f"👉 Saved experiment to {fname}")

def main():
    # ── parse args ───────────────────────────────────
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True, help="path to YAML config")
    p.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="turn off all wandb logging"
    )
    p.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        help="turn on anomaly detection for debugging"
    )
    p.add_argument(
        "--no-amp",
        action="store_false",
        dest="use_amp",
        help="do not use automatic mixed precision"
    )
    p.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device to use for training (default: cuda)'
    )
    # defaults
    p.set_defaults(use_wandb=True, debug=False, use_amp=True)
    args = p.parse_args()


    # ── device setup ─────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please use CPU")
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            device = torch.device("cuda:0")
        else:
            print("Using single GPU")
    else:
        # default auto-select
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── anomaly detection? ─────────────────────────────
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("🐞 Anomaly detection is ON")

    # ── AMP setup ──────────────────────────────────────
    use_amp = args.use_amp and (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)
    print(f"{'🟢' if use_amp else '🔴'} AMP {'enabled' if use_amp else 'disabled'}")
    # ── load config ───────────────────────────────────
    cfg = yaml.safe_load(open(args.config))

    # ── init wandb ───────────────────────────────────

    init_kwargs = dict(
      project="dihedral-prediction",
      config=cfg,
      save_code=True,
    )
    if not args.use_wandb:
        init_kwargs["mode"] = "disabled"
    wandb.init(**init_kwargs)
    config = wandb.config  # now a wandb.Config

    # ── make output dir & snapshot config ─────────────
    ts       = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"dihed_{ts}"
    out_root = cfg["logging"]["output_root"]
    out_dir  = os.path.join(out_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "used_config.yml"), "w"))

    # ── set seed ──────────────────────────────────────
    seed_everything(cfg["training"]["seed"])

    # ── build model ───────────────────────────────────
    mcfg = cfg["model"]
    encoder = DimeNetPPEncoder(
        hidden_channels=mcfg["hidden_channels"],
        out_channels=  mcfg["out_channels"],
        num_blocks=    mcfg["num_blocks"],
        num_spherical= mcfg["num_spherical"],
        num_radial=    mcfg["num_radial"],
        cutoff=        mcfg["cutoff"],
    )
    model = SiameseDimeNet(
        encoder=encoder,
        fusion=  mcfg["fusion"],
        dropout= mcfg["dropout"],
    ).to(device)
    # ── optimizer & scheduler ────────────────────────
    tcfg = cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    scfg = tcfg["scheduler"]
    scheduler = CosineRestartsDecay(
        opt,
        T_0=     int(scfg["T_0"]),
        T_mult=  int(scfg["T_mult"]),
        eta_min= float(scfg["eta_min"]),
        decay=   float(scfg["decay"]),
    )

    # ── data loaders ──────────────────────────────────
    # make data root
    dcfg = cfg["data"]
    os.makedirs(dcfg["root"], exist_ok=True)

    # instantiate your PyG dataset
    dataset = EquiMultiMolDataset(
        root            = dcfg["root"],
        sdf_folder      = dcfg["sdf_folder"],
        target_csv      = dcfg["target_csv"],
        input_type      = dcfg["input_types"],
        target_columns  = dcfg["target_columns"],
        keep_hs         = dcfg["keep_hs"],
        sanitize        = dcfg["sanitize"],
        force_reload    = dcfg["force_reload"],
    )

    
    n = len(dataset)
    g = torch.Generator().manual_seed(cfg["training"]["seed"])
    t,v,te = int(0.8*n), int(0.1*n), n - int(0.8*n) - int(0.1*n)
    train_ds, val_ds, test_ds = random_split(dataset, [t,v,te], generator=g)

    loaders = {
      "train": DataLoader(train_ds, batch_size=tcfg["batch_size"],
                          shuffle=True, follow_batch=['z_s','z_t'], generator=g),
      "val":   DataLoader(val_ds,   batch_size=tcfg["batch_size"],
                          shuffle=False,follow_batch=['z_s','z_t'], generator=g),
        "test":  DataLoader(test_ds,  batch_size=tcfg["batch_size"],
                            shuffle=False,follow_batch=['z_s','z_t'], generator=g),
    }

    print(f"📊 Dataset split: total={len(dataset)} | "
      f"train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    if args.use_wandb:
        wandb.summary["dataset/total"] = len(dataset)
        wandb.summary["dataset/train"] = len(train_ds)
        wandb.summary["dataset/val"]   = len(val_ds)
        wandb.summary["dataset/test"]  = len(test_ds)
        # W&B

    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=20)


    # ── loss + metric ─────────────────────────────────
    loss_fn   = cosine_angle_loss
    metric_fn = AngularErrorMetric(in_degrees=True)

    # ── train/val loop ────────────────────────────────
    best_val_err = 1e9
    for epoch in range(1, tcfg["epochs"]+1):
        # train
        model.train()
        tot_loss, tot_err, count = 0,0,0
        for batch in tqdm(loaders["train"], desc="Training", leave=False):
            batch = batch.to(device)
            opt.zero_grad()

            if use_amp:

                # 1) Run your encoder in full precision (FP32)
                with autocast(enabled=False):
                    h_s, h_t = model.encode(batch)        # you need encode() to return both embeddings

                # 2) Run fusion, head and loss in mixed precision
                with autocast(device_type=device.type):
                    h_fused = model.fuse(h_s, h_t)        # your _fuse() method
                    out     = model.head_and_norm(h_fused)  # head + normalize
                    loss    = loss_fn(out, batch.y)

                # 3) Backward + step
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                # same as before
                out  = model(batch)
                loss = loss_fn(out, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()


            err = metric_fn(out, batch.y)
            b  = batch.y.size(0)
            tot_loss += loss.item()*b
            tot_err  += err.item()*b
            count    += b
        tr_l, tr_e = tot_loss/count, tot_err/count

        # val
        model.eval()
        tot_loss, tot_err, count = 0,0,0
        with torch.no_grad():
            for batch in tqdm(loaders["val"], desc="Validation", leave=False):
                batch = batch.to(device)
                out  = model(batch)
                l    = loss_fn(out, batch.y)
                err  = metric_fn(out, batch.y)
                b    = batch.y.size(0)
                tot_loss += l.item()*b
                tot_err  += err.item()*b
                count    += b
        va_l, va_e = tot_loss/count, tot_err/count

        # scheduler step
        scheduler.step()
        # Print to console
        print(f"Epoch {epoch:03d} | "
              f"Train L={tr_l:.4f}, Err={tr_e:.1f}° | "
              f"Val   L={va_l:.4f}, Err={va_e:.1f}° | "
              f"LR={scheduler.get_last_lr()[0]:.2e}")
        # Log to W&B
        wandb.log({
            "epoch":     epoch,
            "train/loss": tr_l,
            "train/err":  tr_e,
            "val/loss":   va_l,
            "val/err":    va_e,
            "lr":         scheduler.get_last_lr()[0],
        })
        # keep best
        if va_e < best_val_err:
            best_val_err = va_e
            torch.save(model.state_dict(), os.path.join(out_dir, "best_dimenet_model_manual.pt"))
            print(" ↳ New best model saved!")

    # ── finally, record config + metrics ─────────────
    final_metrics = {
      "best_val_loss":   va_l,
      "best_val_error":  best_val_err,
      "final_val_loss":  va_l,
      "final_val_error": va_e,
    }
    save_experiment(cfg, final_metrics, out_dir)

    # Record to W&B
    model.eval()
    tot_loss, tot_err, count = 0,0,0
    with torch.no_grad():
        for batch in tqdm(loaders["test"], desc="Testing", leave=False):
            batch = batch.to(device)
            out  = model(batch)
            l    = loss_fn(out, batch.y)
            err  = metric_fn(out, batch.y)
            b    = batch.y.size(0)
            tot_loss += l.item()*b
            tot_err  += err.item()*b
            count    += b
    test_l, test_e = tot_loss/count, tot_err/count
    print(f"Test L={test_l:.4f}, Err={test_e:.1f}°")
    if args.use_wandb:
        wandb.log({
            "test/loss": test_l,
            "test/err":  test_e,
        })
        wandb.finish()

if __name__=="__main__":
    main()
