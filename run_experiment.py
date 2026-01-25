# run_experiment.py
# TODO: hier die komentare anpassen!
# PURPOSE
# -------
# Command-line entry point to run ONE deep-learning experiment from the terminal.
# Supported:
#   - Models: LeNet5, ResNet18
#   - Datasets: GTSRB, CIFAR10
#
# What this script does:
#   - trains the selected model on the selected dataset
#   - optionally applies training-time augmentation (train split only)
#   - evaluates on the test set (confusion matrix, per-class recall, balanced accuracy, report)
#   - optionally saves a complete run folder (weights + config + history + plots)
#
# This satisfies the assignment requirement:
# "configurable via command-line options or a configuration file"
# (no code edits required to change hyperparameters or evaluation behavior).
#
# HOW TO RUN (examples)
# ---------------------
# 1) Show CLI options (including defaults):
#    python run_experiment.py --help
#
# 2) Train LeNet5 on a small GTSRB subset (debug), with augmentation and saved artifacts:
#    python run_experiment.py --mode train --model lenet5 --dataset gtsrb --data-root data/GTSRB ^
#        --epochs 3 --batch-size 128 --lr 1e-3 --dropout 0.2 --activation tanh ^
#        --normalize 1 --augment 1 --debug-fraction 0.05 --save-run 1
#
# 3) Train ResNet18 on full GTSRB (optionally pretrained, optionally frozen backbone), save artifacts:
#    python run_experiment.py --mode train --model resnet18 --dataset gtsrb --data-root data/GTSRB ^
#        --epochs 10 --batch-size 128 --lr 1e-3 --dropout 0.3 --pretrained 1 --freeze-backbone 1 ^
#        --normalize 1 --augment 1 --debug-fraction 1.0 --save-run 1
#
# 4) Evaluate a saved run again (no training). Uses the stored config.json to rebuild model + loaders:
#    python run_experiment.py --mode eval --run-dir runs/RESNET18/GTSRB/<RUN_FOLDER>
#
# NOTE (Windows)
# --------------
# Keep the "if __name__ == '__main__': main()" guard at the end,
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
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# ---- Adjust these imports to your filenames ----
from data.gtsrb_loader import get_gtsrb_dataloaders
from data.cifar_loader import get_cifar10_dataloaders
from DL_approach.train_utils import train_model
from DL_approach.LeNet import LeNet5  
from DL_approach.resnet import ResNet18

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
    model: str

    dataset: str
    data_root: str
    img_size: int
    normalize: int
    debug_fraction: float
    augment: int 
    seed: int

    # LeNet params
    in_channels: int
    num_classes: int
    activation: str
    dropout: float
    adapt_lenet: int

    # ResNet params
    pretrained: int
    freeze_backbone: int

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

    base = (
        f"{args.model}__{args.dataset}{args.img_size}"
        f"__aug{args.augment}"
        f"__do{args.dropout}"
        f"__ep{args.epochs}"
        f"__opt{args.optimizer}{args.lr}"
        f"__dbg{args.debug_fraction}"
        f"__s{args.seed}"
    )

    if args.model == "lenet5":
        base += f"__act{args.activation}__adapt{args.adapt_lenet}"
    elif args.model == "resnet18":
        base += f"__pt{args.pretrained}__frz{args.freeze_backbone}"

    tag = base + f"__{stamp}"

    # tag = (
    #     f"lenet5__{args.dataset}{args.img_size}"
    #     f"__aug{args.augment}" 
    #     f"__act{args.activation}"
    #     f"__do{args.dropout}"
    #     f"__ep{args.epochs}"
    #     f"__opt{args.optimizer}{args.lr}"
    #     f"__dbg{args.debug_fraction}"
    #     f"__s{args.seed}"
    #     f"__{stamp}"
    # )
    run_name = slugify(tag)
    run_dir = Path(args.runs_dir) / args.model.upper() / args.dataset.upper() / run_name
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
    augment: int
):
    """
    Returns:
      train_loader, test_loader, class_names, in_channels, num_classes
    """
    if dataset == "gtsrb":
        use_aug = bool(augment)  # oder augment, je nachdem wie du es speicherst

        # Important: Normalizing happens in dataset -> NO transforms.Normalize here
        train_tfm_list = [transforms.Resize((img_size, img_size))]
        test_tfm_list  = [transforms.Resize((img_size, img_size))]

        if use_aug:
            train_tfm_list += [
                # Mild lighting changes; keep hue small because sign color matters
                transforms.ColorJitter(
                    brightness=0.15,   # Mild illumination changes (shadows/sun) without destroying signal information
                    contrast=0.15,     # Handles varying camera exposure / contrast conditions
                    saturation=0.10,   # Slight color intensity variation; keep small to preserve class-discriminative colors
                    hue=0.02,          # Very small hue shift because sign color (e.g., red/blue) is often class-relevant
                ),
                # Realistic slight camera tilt
                transforms.RandomRotation(degrees=8), # Small camera tilt / imperfect alignment; avoid large angles that change sign appearance too much
                # Slight shift/scale; keep it small to avoid cutting off the sign at 32x32
                transforms.RandomAffine(
                    degrees=0,                               # Rotation already handled above; keep affine focused on shift/scale only
                    translate=(0.03, 0.03),                  # Small position jitter to tolerate imperfect cropping/localization
                    scale=(0.95, 1.05),                      # Small scale jitter to simulate distance changes; kept tight to avoid losing details at 32x32
                ),
            ]

        train_transform = transforms.Compose(train_tfm_list + [transforms.ToTensor()])
        test_transform  = transforms.Compose(test_tfm_list  + [transforms.ToTensor()])

        train_loader, test_loader = get_gtsrb_dataloaders(
            root=data_root,
            img_size=(img_size, img_size),
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            debug_fraction=debug_fraction,
            seed=seed,
            train_transform=train_transform,
            test_transform=test_transform,
        )
        class_names = [str(i) for i in range(43)]
        return train_loader, test_loader, class_names, 3, 43

    if dataset == "cifar10":
        
        # DL pipeline: CIFAR10 is always 32x32
        if img_size != 32:
            raise ValueError("CIFAR10 DL pipeline expects img_size=32.")
        
        use_aug = bool(augment)

        # TODO: source for augmentation
        if use_aug:
            train_transform = v2.Compose([
                # Adds small random translations -> improves shift/position robustness
                v2.RandomCrop(size=(32, 32), padding=4),
                # Enforces left-right invariance; doubles effective views for many classes
                v2.RandomHorizontalFlip(p=0.5),
                # Simulates lighting/color variations -> reduces overfitting to color/illumination
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            train_transform = None

        test_transform = None

        train_loader, test_loader = get_cifar10_dataloaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            img_size=(img_size, img_size),
            train_transform=train_transform,
            test_transform = test_transform
        )
        class_names = [
            "airplane","automobile","bird","cat","deer",
            "dog","frog","horse","ship","truck"
        ]
        return train_loader, test_loader, class_names, 3, 10

    raise ValueError(f"Unknown dataset: {dataset}")


def make_optimizer(
    optimizer_name: str,
    params,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    You can extend this later if you need more options (SGD momentum, etc.).
    """
    if optimizer_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
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
        "--model",
        choices=["lenet5", "resnet18"],
        default="lenet5",
        help="Which model to run."
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
        help="Input size. For CIFAR10 it is forced to 32; for GTSRB you can use 32 or 64."
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

    # ResNet-specific
    p.add_argument(
        "--pretrained",
        type=int,
        choices=[0, 1],
        default=0,
        help="ResNet18 only: 1 loads ImageNet pretrained weights, 0 trains from scratch."
    )
    p.add_argument(
        "--freeze-backbone",
        type=int,
        choices=[0, 1],
        default=0,
        help="ResNet18 only: 1 freezes backbone and trains only the final layer (fc)."
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

    # CIFAR10 is defined at 32x32. For the DL pipeline we force it.
    if args.dataset == "cifar10" and args.img_size != 32:
        print(f"[INFO] Forcing img_size from {args.img_size} to 32 for CIFAR10 (DL pipeline).")
        args.img_size = 32

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
            augment=args.augment
        )

        print("Train samples:", len(train_loader.dataset))
        print("Test  samples:", len(test_loader.dataset))

        # # Build LeNet model
        # model = LeNet5(
        #     in_channels=in_channels,
        #     num_classes=num_classes,
        #     input_size=args.img_size,
        #     activation=args.activation,
        #     adapt_to_lenet_geometry=bool(args.adapt_lenet),
        #     dropout_p=float(args.dropout),
        # ).to(device)

        if args.model == "lenet5":
            model = LeNet5(
                in_channels=in_channels,
                num_classes=num_classes,
                input_size=args.img_size,
                activation=args.activation,
                adapt_to_lenet_geometry=bool(args.adapt_lenet),
                dropout_p=float(args.dropout),
            ).to(device)

        elif args.model == "resnet18":
            model = ResNet18(
                in_channels=in_channels,
                num_classes=num_classes,
                pretrained=bool(args.pretrained),
                freeze_backbone=bool(args.freeze_backbone),
                dropout=float(args.dropout),
            ).to(device)

        else:
            raise ValueError(f"Unknown model: {args.model}")


        criterion = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = make_optimizer(args.optimizer, params, lr=args.lr, weight_decay=args.weight_decay)

        # Decide run directory and save config early
        run_dir: Optional[Path] = None
        if args.save_run:
            run_dir = make_run_dir(args)
            config = Runconfig(
                model=args.model,
                
                dataset=args.dataset,
                data_root=args.data_root,
                img_size=args.img_size,
                normalize=args.normalize,
                debug_fraction=args.debug_fraction,
                augment=args.augment,
                seed=args.seed,

                in_channels=in_channels,
                num_classes=num_classes,

                # LeNet params (only meaningful for LeNet, but must exist)
                activation=args.activation if args.model == "lenet5" else "tanh",
                dropout=float(args.dropout),
                adapt_lenet=args.adapt_lenet if args.model == "lenet5" else 0,

                # ResNet params (only meaningful for ResNet, but must exist)
                pretrained=int(args.pretrained) if args.model == "resnet18" else 0,
                freeze_backbone=int(args.freeze_backbone) if args.model == "resnet18" else 0,

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
            f"{args.model}_{args.dataset}{args.img_size}"
            f"_aug{args.augment}"
            f"_do{args.dropout}"
            f"_ep{args.epochs}_{args.optimizer}{args.lr}"
            f"_dbg{args.debug_fraction}"
        )

        if args.model == "lenet5":
            run_tag += f"_act{args.activation}_adapt{args.adapt_lenet}"
        else:
            run_tag += f"_pt{args.pretrained}_frz{args.freeze_backbone}"

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

            title = f"{args.model.upper()} on {args.dataset.upper()} ({args.img_size}x{args.img_size}, aug={args.augment})"

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
            augment=config.augment
        )

        # # Rebuild model consistently with config
        # model = LeNet5(
        #     in_channels=in_channels,
        #     num_classes=num_classes,
        #     input_size=config.img_size,
        #     activation=config.activation,
        #     adapt_to_lenet_geometry=bool(config.adapt_lenet),
        #     dropout_p=float(config.dropout),
        # ).to(device)

        if config.model == "lenet5":
            model = LeNet5(
                in_channels=in_channels,
                num_classes=num_classes,
                input_size=config.img_size,
                activation=config.activation,
                adapt_to_lenet_geometry=bool(config.adapt_lenet),
                dropout_p=float(config.dropout),
            ).to(device)

        elif config.model == "resnet18":
            model = ResNet18(
                in_channels=in_channels,
                num_classes=num_classes,
                pretrained=bool(config.pretrained),
                freeze_backbone=bool(config.freeze_backbone),
                dropout=float(config.dropout),
            ).to(device)

        else:
            raise ValueError(f"Unknown model in config: {config.model}")


        # Load weights
        state = torch.load(run_dir / "model.pth", map_location="cpu")
        model.load_state_dict(state)

        # Evaluate
        title = f"{config.model.upper()} on {config.dataset.upper()} ({config.img_size}x{config.img_size}, aug={config.augment})"

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
