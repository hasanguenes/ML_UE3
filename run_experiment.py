# run_experiment.py
# TODO: hier die komentare anpassen!
# PURPOSE
# -------
# Command-line entry point to run ONE LeNet experiment:
#   - train LeNet5 on a dataset (GTSRB or CIFAR10)
#   - optionally save run artifacts (weights + config + history + plots)
#   - run "detailed" evaluation (confusion matrix, per-class recall, balanced accuracy)
#
# This satisfies the assignment requirement:
# "configurable via command-line options or a configuration file"
#
# HOW TO RUN (examples)
# ---------------------
# 1) Show CLI options:
#    python run_experiment.py --help
#
# 2) Train on GTSRB (debug subset), save artifacts, detailed eval:
#    python run_experiment.py --mode train --dataset gtsrb --data-root ../data/GTSRB ^
#        --img-size 32 --epochs 3 --batch-size 128 --lr 1e-3 ^
#        --dropout 0.2 --activation tanh --normalize 1 --debug-fraction 0.05 ^
#        --save-run 1 --eval detailed
#
# 3) Train on full GTSRB, no debug, save:
#    python run_experiment.py --mode train --dataset gtsrb --data-root ../data/GTSRB ^
#        --img-size 32 --epochs 10 --batch-size 128 --lr 1e-3 ^
#        --dropout 0.0 --activation tanh --normalize 1 --debug-fraction 1.0 ^
#        --save-run 1 --eval detailed
#
# 4) Evaluate a saved run again (no training):
#    python run_experiment.py --mode eval --run-dir runs/GTSRB/<RUN_FOLDER> --eval detailed
#
# NOTE (Windows):
# Keep the if __name__ == "__main__": main() guard at the end,
# otherwise DataLoader multiprocessing can break.

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# ---- Adjust these imports to your filenames ----
from data.gtsrb_loader import get_gtsrb_dataloaders
from data.cifar_loader import get_cifar10_dataloaders
from DL_approach.train_utils import train_model
from DL_approach.LeNet import LeNet5  

# print("RUN_EXPERIMENT PATH =", __file__)

# =============================================================================
# 1) Small helper functions
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Makes randomness more reproducible (not perfect, but good enough for class projects).
    Affects weight init, shuffles, your debug subset selection, etc.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    """
    Converts CLI value into torch.device:
      - auto: cuda if available else cpu
      - cuda: force cuda
      - cpu : force cpu
    """
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def slugify(s: str, max_len: int = 140) -> str:
    """
    Creates a filesystem-safe folder name (for run directories).
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-_\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    """
    Balanced accuracy (multi-class):
      mean over classes of recall_k
    recall_k = cm[k,k] / sum(cm[k,:])

    Useful when classes are imbalanced.
    """
    row_sums = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.where(row_sums > 0, np.diag(cm) / row_sums, 0.0)
    return float(np.mean(recalls))


# =============================================================================
# 2) Detailed evaluation: collect predictions once, then compute metrics/plots
# =============================================================================

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the model over the full dataloader and returns all labels & predictions.

    Why do this?
    - confusion matrix needs all y_true and y_pred
    - classification_report needs all y_true and y_pred
    """
    model.eval()  # disables dropout during evaluation automatically

    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)

        logits = model(images)              # (batch, num_classes)
        preds = torch.argmax(logits, dim=1) # (batch,)

        y_pred.extend(preds.cpu().numpy().tolist())
        y_true.extend(labels.cpu().numpy().tolist())

    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
    dpi: int = 200,
) -> None:
    """
    Saves a confusion matrix plot. For many classes, labels become unreadable,
    so for >20 classes we hide tick labels by default.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_per_class_recall(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
    dpi: int = 200,
) -> None:
    """
    Plots per-class recall (= per-class accuracy in your previous terminology):
      recall_k = cm[k,k] / sum(cm[k,:])
    """
    row_sums = cm.sum(axis=1)
    recalls = np.where(row_sums > 0, np.diag(cm) / row_sums, 0.0)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)

    ax.bar(np.arange(len(recalls)), recalls)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Recall (per-class accuracy)")

    if len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
    else:
        ax.set_xticks(np.arange(len(recalls)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def evaluate_detailed_and_save(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    out_dir: Path,
    title: str,
) -> Dict[str, Any]:
    """
    Detailed evaluation with artifacts saved to disk:
      - report.txt: precision/recall/f1
      - metrics.json: overall + balanced accuracy
      - confusion_matrix.png
      - per_class_recall.png
    """
    y_true, y_pred = collect_predictions(model, dataloader, device=device)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    overall_acc = float((y_true == y_pred).mean())
    bal_acc = balanced_accuracy_from_cm(cm)

    report_txt = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=3,
        zero_division=0,
    )

    # Save metrics and report
    (out_dir / "report.txt").write_text(report_txt, encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps({"overall_accuracy": overall_acc, "balanced_accuracy": bal_acc}, indent=2),
        encoding="utf-8",
    )

    # Save plots
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title=f"Confusion Matrix – {title}",
        out_path=out_dir / "confusion_matrix.png",
    )
    plot_per_class_recall(
        cm=cm,
        class_names=class_names,
        title=f"Per-class Recall – {title}",
        out_path=out_dir / "per_class_recall.png",
    )

    return {"overall_accuracy": overall_acc, "balanced_accuracy": bal_acc, "confusion_matrix": cm}


# =============================================================================
# 3) What we store per run (config.json)
# =============================================================================

@dataclass
class Runconfig:
    """
    Everything needed to understand and re-load a run later.
    """
    dataset: str
    data_root: str
    img_size: int
    normalize: int
    debug_fraction: float
    seed: int

    # LeNet params
    in_channels: int
    num_classes: int
    activation: str
    dropout: float
    adapt_lenet: int

    # training params
    epochs: int
    batch_size: int
    optimizer: str
    lr: float
    weight_decay: float

    created_at: str


def make_run_dir(args: argparse.Namespace) -> Path:
    """
    Creates one unique run folder containing:
      model.pth, config.json, history.json, curves, confusion matrix, etc.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    tag = (
        f"lenet5__{args.dataset}{args.img_size}"
        f"__act{args.activation}"
        f"__do{args.dropout}"
        f"__ep{args.epochs}"
        f"__opt{args.optimizer}{args.lr}"
        f"__dbg{args.debug_fraction}"
        f"__s{args.seed}"
        f"__{stamp}"
    )
    run_name = slugify(tag)
    run_dir = Path(args.runs_dir) / args.dataset.upper() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_checkpoint(model: nn.Module, run_dir: Path) -> None:
    """
    Save model weights only (state_dict).
    Re-loading later:
      model = LeNet5(...)
      model.load_state_dict(torch.load(...))
    """
    torch.save(model.state_dict(), run_dir / "model.pth")


def save_config(config: Runconfig, run_dir: Path) -> None:
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_config(run_dir: Path) -> Runconfig:
    config_dict = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return Runconfig(**config_dict)


# =============================================================================
# 4) Build dataloaders based on dataset choice
# =============================================================================

def build_loaders_and_classes(
    dataset: str,
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    normalize: bool,
    debug_fraction: float,
    seed: int,
):
    """
    Returns:
      train_loader, test_loader, class_names, in_channels, num_classes
    """
    if dataset == "gtsrb":
        train_loader, test_loader = get_gtsrb_dataloaders(
            root=data_root,
            img_size=(img_size, img_size),
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            debug_fraction=debug_fraction,
            seed=seed,
        )
        class_names = [str(i) for i in range(43)]
        return train_loader, test_loader, class_names, 3, 43

    if dataset == "cifar10":
        train_loader, test_loader = get_cifar10_dataloaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            img_size=(img_size, img_size),
            train_transform=None,
            test_transform=None,
        )
        class_names = [
            "airplane","automobile","bird","cat","deer",
            "dog","frog","horse","ship","truck"
        ]
        return train_loader, test_loader, class_names, 3, 10

    raise ValueError(f"Unknown dataset: {dataset}")


def make_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    You can extend this later if you need more options (SGD momentum, etc.).
    """
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError("optimizer must be 'adam' or 'sgd'")


# =============================================================================
# 5) CLI parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one LeNet experiment (train/eval) via CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Execution mode. 'train' trains a new model; 'eval' loads a saved run from --run-dir and evaluates it."
    )
    # p.add_argument(
    #     "--eval",
    #     choices=["none", "basic", "detailed"],
    #     default="basic",
    #     help="Evaluation level. 'none' skips evaluation; 'basic' prints overall accuracy; 'detailed' also saves confusion matrix, per-class recall and a classification report."
    # )

    p.add_argument(
        "--dataset",
        choices=["gtsrb", "cifar10"],
        default="gtsrb",
        help="Which dataset to use."
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="../data/GTSRB",
        help="Root directory of the dataset on disk (folder that contains the dataset subfolders/files)."
    )

    p.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base output directory where run folders (weights, history, plots) will be stored."
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Path to an existing run folder (containing model.pth + config.json). Required when --mode eval."
    )

    p.add_argument(
        "--img-size",
        type=int,
        default=32,
        help="Input image size (images will be resized to IMG_SIZE x IMG_SIZE by the dataloader)."
    )
    p.add_argument(
        "--normalize",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to normalize inputs using the dataset mean/std (1=yes, 0=no)."
    )
    p.add_argument(
        "--augment",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable data augmentation for the training set (1=yes, 0=no). Applied only to train split."
    )

    p.add_argument(
        "--debug-fraction",
        type=float,
        default=1.0,
        help="Load only this fraction of the dataset for quick debugging (e.g., 0.05 loads 5%%)."
    )

    # LeNet-specific
    p.add_argument(
        "--activation",
        choices=["tanh", "relu"],
        default="tanh",
        help="Activation function used inside LeNet-5 (classic: tanh; modern variant: relu)."
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability applied in the fully connected layer(s). Set 0.0 to disable dropout."
    )
    p.add_argument(
        "--adapt-lenet",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, enables adaptive pooling to match classic LeNet geometry when using non-32x32 inputs."
    )

    # training
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (one epoch = one full pass over the training set)."
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size used by the DataLoader during training."
    )
    p.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer type used for training (adam or sgd with momentum)."
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer."
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay (regularization). Use 0.0 to disable."
    )

    # runtime
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device. 'auto' uses CUDA if available; otherwise CPU."
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes. On Windows, 0 is often the most reliable choice."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (initialization, shuffling, debug subset selection)."
    )

    p.add_argument(
        "--save-run",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, saves the run (weights + config/history/plots) under --runs-dir. If 0, trains/evaluates without writing files."
    )


    return p.parse_args()


# =============================================================================
# 6) Main orchestration
# =============================================================================

def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    set_seed(args.seed)

    print("Device:", device)

    # -------------------------------------------------------------------------
    # TRAIN MODE
    # -------------------------------------------------------------------------
    if args.mode == "train":
        train_loader, test_loader, class_names, in_channels, num_classes = build_loaders_and_classes(
            dataset=args.dataset,
            data_root=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize=bool(args.normalize),
            debug_fraction=args.debug_fraction,
            seed=args.seed,
        )

        print("Train samples:", len(train_loader.dataset))
        print("Test  samples:", len(test_loader.dataset))

        # Build LeNet model
        model = LeNet5(
            in_channels=in_channels,
            num_classes=num_classes,
            input_size=args.img_size,
            activation=args.activation,
            adapt_to_lenet_geometry=bool(args.adapt_lenet),
            dropout_p=float(args.dropout),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = make_optimizer(args.optimizer, model, lr=args.lr, weight_decay=args.weight_decay)

        # Decide run directory and save config early
        run_dir: Optional[Path] = None
        if args.save_run:
            run_dir = make_run_dir(args)
            config = Runconfig(
                dataset=args.dataset,
                data_root=args.data_root,
                img_size=args.img_size,
                normalize=args.normalize,
                debug_fraction=args.debug_fraction,
                seed=args.seed,
                in_channels=in_channels,
                num_classes=num_classes,
                activation=args.activation,
                dropout=float(args.dropout),
                adapt_lenet=args.adapt_lenet,
                epochs=args.epochs,
                batch_size=args.batch_size,
                optimizer=args.optimizer,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            save_config(config, run_dir)

        # Train (your existing function)
        run_tag = (
            f"{args.dataset}{args.img_size}"
            f"_do{args.dropout}_act{args.activation}"
            f"_ep{args.epochs}_{args.optimizer}{args.lr}"
            f"_dbg{args.debug_fraction}"
        )
        model, history = train_model(
            model=model,
            trainloader=train_loader,
            evalloader=test_loader,   # you currently use test as eval (consistent with your workflow)
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=args.epochs,
            save_curves=True,
            curves_dir=str(run_dir) if run_dir is not None else "training_curves",
            run_tag=run_tag,
        )

        # Save artifacts
        if run_dir is not None:
            save_checkpoint(model, run_dir)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            title = f"LeNet5 on {args.dataset.upper()} ({args.img_size}x{args.img_size})"
            metrics = evaluate_detailed_and_save(
                model=model,
                dataloader=test_loader,
                class_names=class_names,
                device=device,
                out_dir=run_dir,
                title=title,
            )
            print("\nDetailed eval:")
            print(f"  overall accuracy : {metrics['overall_accuracy']:.4f}")
            print(f"  balanced accuracy: {metrics['balanced_accuracy']:.4f}")

            print(f"\nSaved run to: {run_dir}")
        else:
            print("\nTraining finished (no run saved).")

    # -------------------------------------------------------------------------
    # EVAL MODE (load an existing run)
    # -------------------------------------------------------------------------
    else:
        if not args.run_dir:
            raise ValueError("--run-dir is required for --mode eval")

        run_dir = Path(args.run_dir)
        config = load_config(run_dir)

        # Rebuild dataloaders consistently with config
        train_loader, test_loader, class_names, in_channels, num_classes = build_loaders_and_classes(
            dataset=config.dataset,
            data_root=config.data_root,
            img_size=config.img_size,
            batch_size=config.batch_size,
            num_workers=args.num_workers,
            normalize=bool(config.normalize),
            debug_fraction=1.0,  # evaluate on full set by default
            seed=config.seed,
        )

        # Rebuild model consistently with config
        model = LeNet5(
            in_channels=in_channels,
            num_classes=num_classes,
            input_size=config.img_size,
            activation=config.activation,
            adapt_to_lenet_geometry=bool(config.adapt_lenet),
            dropout_p=float(config.dropout),
        ).to(device)

        # Load weights
        state = torch.load(run_dir / "model.pth", map_location="cpu")
        model.load_state_dict(state)

        # Evaluate
        title = f"LeNet5 on {config.dataset.upper()} ({config.img_size}x{config.img_size})"
        metrics = evaluate_detailed_and_save(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            device=device,
            out_dir=run_dir,
            title=title,
        )
        print("\nDetailed eval:")
        print(f"  overall accuracy : {metrics['overall_accuracy']:.4f}")
        print(f"  balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"\nArtifacts written to: {run_dir}")


if __name__ == "__main__":
    # On Windows, this guard is required for DataLoader multiprocessing (num_workers > 0).
    main()
