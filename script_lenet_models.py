# sweep_lenet.py
# Sequentially runs run_experiment.py multiple times with different LeNet configs.
# - Windows-friendly (sequential, no multiprocessing here)
# - writes a log file per run
# - relies on run_experiment.py to save runs/artifacts (runs/... + config.json + model.pth etc.)

from __future__ import annotations

import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

RUN_EXPERIMENT = Path(__file__).parent / "run_experiment.py"
LOG_DIR = Path(__file__).parent / "sweep_logs_lenet"

VARIED_KEYS = ["dropout", "lr", "augment", "epochs"]  

def format_varied(cfg: dict) -> str:
    return ", ".join(f"{k}={cfg[k]}" for k in VARIED_KEYS if k in cfg)


def dict_to_cli_args(cfg: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in cfg.items():
        args.append(f"--{k.replace('_', '-')}")
        args.append(str(v))
    return args


def run_one(cfg: Dict[str, Any], idx: int, total: int) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"lenet_run_{idx:03d}_{stamp}.log"

    cmd = [sys.executable, str(RUN_EXPERIMENT)] + dict_to_cli_args(cfg)

    # ONLY show the parameters that change (no full command, no full config)
    summary = format_varied(cfg)
    header = f"=== RUN {idx}/{total} | {summary} ===\n"
    print(header, end="", flush=True)

    with log_path.open("w", encoding="utf-8") as f:
        # If you also want the log header to be minimal, keep it like this:
        f.write(header)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")   # show run_experiment output live
            f.write(line)

        rc = proc.wait()

    print(f"[DONE] rc={rc} | log={log_path}", flush=True)
    return rc


def make_configs() -> List[Dict[str, Any]]:
    """
    Define the sweep grid here.
    Keep it small first, then expand.
    """
    base = dict(
        model="lenet5",       # if your run_experiment expects this argument; otherwise remove
        mode="train",
        runs_dir="runs",
        save_run=1,
        device="auto",
        num_workers=0,        # Windows: safest
        seed=42,
        normalize=1,
        debug_fraction=1,   # set e.g. 0.05 to test quickly
        img_size=32,          # keep fixed if you decided to stick to 32x32
        dataset="gtsrb",      # or "cifar10"
        # dataset="cifar10",     
        data_root="data/GTSRB",  # adjust if needed
    )

    # Grid: change only what you actually want to compare
    activations = ["tanh"]
    dropouts = [0.0, 0.2, 0.5]
    # lrs = [1e-3, 5e-4]
    lrs = [1e-3]
    # augments = [0, 1]
    augments = [1]
    # epochs = [5, 10, 15, 20, 25, 30] 
    # epochs = [35, 40] 
    epochs = [1] 
    batch_sizes = [128]

    # Grid: change only what you actually want to compare
    activations = ["tanh"]
    # dropouts = [0.0, 0.2, 0.5]
    dropouts = [0.0]
    lrs = [1e-3]
    augments = [0, 1]
    epochs = [5, 10, 15, 20, 25, 30] 
    epochs = [1]
    batch_sizes = [128]

    configs: List[Dict[str, Any]] = []
    for ep, act, do, lr, aug, bs in itertools.product(
        epochs, activations, dropouts, lrs, augments, batch_sizes
    ):
        cfg = dict(base)
        cfg.update(
            activation=act,
            dropout=do,
            lr=lr,
            augment=aug,
            epochs=ep,
            batch_size=bs,
            optimizer="adam",
            weight_decay=0.0,
            adapt_lenet=0,
        )
        configs.append(cfg)

    return configs

DONE = {
    # (0.0, 1e-3, 0, 5),   # dropout, lr, augment, epochs
    # (0.0, 1e-3, 0, 10),
    # (0.0, 1e-3, 0, 15),    
}

def main() -> None:
    if not RUN_EXPERIMENT.exists():
        raise FileNotFoundError(f"Cannot find: {RUN_EXPERIMENT}")
    
    configs = make_configs()

    # Re-order configs: smallest epochs first, then all combos for that epoch, etc.
    # (epochs is the primary key; rest define within-epoch order)
    configs.sort(key=lambda c: (int(c["epochs"]), float(c["dropout"]), float(c["lr"]), int(c["augment"])))

    total = len(configs)
    print(f"Planned LeNet runs: {total}")
    if total == 0:
        return

    failed = 0
    skipped = 0

    for i, cfg in enumerate(configs, start=1):
        key = (float(cfg["dropout"]), float(cfg["lr"]), int(cfg["augment"]), int(cfg["epochs"]))

        if key in DONE:
            skipped += 1
            print(
                f"[SKIP] {i}/{total} "
                f"dropout={cfg['dropout']} lr={cfg['lr']} augment={cfg['augment']} epochs={cfg['epochs']}",
                flush=True,
            )
            continue

        print(
            f"\n=== Running {i}/{total} | "
            f"dropout={cfg['dropout']} lr={cfg['lr']} augment={cfg['augment']} epochs={cfg['epochs']} ===",
            flush=True,
        )

        rc = run_one(cfg, i, total)
        if rc != 0:
            failed += 1

        print(f"=== Finished {i}/{total} ===", flush=True)

    print(f"\nSweep finished. Skipped: {skipped}/{total} | Failed: {failed}/{total}")
    print(f"Logs: {LOG_DIR}")
    print("Runs saved under: runs/ (by run_experiment.py)")

if __name__ == "__main__":
    main()
