# train_utils.py
"""
Training utilities for PyTorch image classification models.
"""

from __future__ import annotations

import time
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def _move_to_device(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Moves a batch to the target device.

    :param images: Input batch tensor, typically (B, C, H, W).
    :param labels: Ground-truth labels, typically (B,).
    :param device: torch.device("cpu") or torch.device("cuda").
    :return: (images_on_device, labels_on_device)
    """
    # non_blocking=True can improve throughput when using pinned memory in DataLoader (pin_memory=True)
    images = images.to(device=device, non_blocking=True)
    labels = labels.to(device=device, non_blocking=True)
    return images, labels


def _count_correct_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> int:
    """
    Computes the number of correct predictions for a batch.

    :param logits: Raw model outputs (before softmax), shape (B, num_classes).
    :param labels: Ground-truth class indices, shape (B,).
    :return: Number of correct predictions in this batch (int).
    """
    # Predicted class = argmax over class dimension
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return int(correct)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Trains the model for exactly one epoch (one full pass over the dataloader).

    What happens in each batch:
      1) forward pass:  logits = model(images)
      2) loss:          loss = criterion(logits, labels)
      3) backward:      loss.backward()
      4) update step:   optimizer.step()

    :param model: The neural network (nn.Module).
    :param dataloader: Iterates over the training set batches.
    :param optimizer: Updates model parameters (e.g., Adam, SGD).
    :param criterion: Loss function (e.g., nn.CrossEntropyLoss()).
    :param device: CPU or GPU device.
    :return: (epoch_loss_mean, epoch_accuracy)
    """
    # IMPORTANT: model.train() enables training-specific behaviors, e.g. Dropout active, BatchNorm updates running stats.
    model.train()

    # Accumulators over the *entire epoch* (not just per batch)
    sum_loss_over_samples: float = 0.0
    num_correct: int = 0
    num_samples: int = 0

    # leave=False avoids cluttering the console with many progress bars across epochs
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = _move_to_device(images=images, labels=labels, device=device)

        # Zero gradients from the previous step.
        # set_to_none=True is a performance/memory optimization recommended by PyTorch in many cases.
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: compute logits for each class
        logits = model(images)

        # Loss: for CrossEntropyLoss, logits should NOT have softmax applied
        loss = criterion(logits, labels)

        # Backward pass: compute gradients of loss w.r.t. parameters
        loss.backward()

        # Parameter update step (e.g., one Adam/SGD step)
        optimizer.step()

        # Update statistics (use per-sample totals, not per-batch averages)
        batch_size = labels.size(0)
        sum_loss_over_samples += loss.item() * batch_size
        num_correct += _count_correct_predictions(logits=logits, labels=labels)
        num_samples += batch_size

    # Mean loss over all samples in the epoch
    epoch_loss = sum_loss_over_samples / num_samples

    # Accuracy over all samples in the epoch
    epoch_acc = num_correct / num_samples

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates a model on a dataloader (validation or test).

    Differences to training:
      - model.eval() disables training-specific behaviors (Dropout off, BatchNorm uses stored stats).
      - torch.no_grad() disables gradient tracking, making it faster and using less memory.

    :param model: The neural network (nn.Module).
    :param dataloader: Iterates over validation/test batches.
    :param criterion: Loss function (e.g., nn.CrossEntropyLoss()).
    :param device: CPU or GPU device.
    :return: (epoch_loss_mean, epoch_accuracy)
    """
    model.eval()

    sum_loss_over_samples: float = 0.0
    num_correct: int = 0
    num_samples: int = 0

    for images, labels in tqdm(dataloader, desc="Evaluation", leave=False):
        images, labels = _move_to_device(images=images, labels=labels, device=device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        sum_loss_over_samples += loss.item() * batch_size
        num_correct += _count_correct_predictions(logits=logits, labels=labels)
        num_samples += batch_size

    epoch_loss = sum_loss_over_samples / num_samples
    epoch_acc = num_correct / num_samples

    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    evalloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 10,
    save_curves: bool = True,
    curves_dir: str = "training_plots",
    curves_dpi: int = 300,
    run_tag: str = "",   # e.g. "GTSRB32_Adam_lr1e-3_aug"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Full training loop running for a fixed number of epochs.

    This function:
      - moves the model to the chosen device (CPU/GPU)
      - repeats:
          train_one_epoch(...)
          evaluate(...)
      - records metrics for plotting/analysis (loss, accuracy, times)

    :param model: The neural network (nn.Module).
    :param trainloader: DataLoader over training data (shuffle=True recommended).
    :param evalloader: DataLoader over evaluation data (shuffle=False recommended).
    :param optimizer: Optimizer (Adam/SGD/...).
    :param criterion: Loss function (CrossEntropyLoss for multi-class classification).
    :param device: CPU or GPU.
    :param epochs: Number of full passes over the training data.
    :return: (trained_model, history_dict)
    """
    model.to(device=device)

    history: Dict[str, List[float]] = {
        # Metrics per epoch (same length == epochs)
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": [],
        # Runtime per epoch
        "train_time_s": [],
        "eval_time_s": [],
    }

    for epoch_index in range(1, epochs + 1):
        print(f"\nEpoch {epoch_index}/{epochs}")

        # ---- Training phase (timed) ----
        if device.type == "cuda":
            # synchronize before/after timing so you measure actual GPU execution time
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        train_time_s = t1 - t0

        # ---- Evaluation phase (timed) ----
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        eval_loss, eval_acc = evaluate(
            model=model,
            dataloader=evalloader,
            criterion=criterion,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        eval_time_s = t1 - t0

        # Record everything for later plots / reporting
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)
        history["train_time_s"].append(train_time_s)
        history["eval_time_s"].append(eval_time_s)

        # Console output (minimal but informative)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Time: {train_time_s:.2f}s")
        print(f"Eval  Loss: {eval_loss:.4f} | Eval  Acc: {eval_acc:.4f} | Time: {eval_time_s:.2f}s")

    # ---- End-of-training summary + plots ----
    total_train_time_s = float(sum(history["train_time_s"]))
    total_eval_time_s = float(sum(history["eval_time_s"]))
    print(f"\nRuntime summary:")
    print(f"  total train time: {total_train_time_s:.2f}s")
    print(f"  total eval  time: {total_eval_time_s:.2f}s")

    # The idea of saving plots was given by a colleague. @credits to him
    if save_curves:
        out_dir = Path(curves_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        safe_tag = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in run_tag).strip("_")
        tag_part = f"_{safe_tag}" if safe_tag else ""

        file_name = out_dir / f"{model.__class__.__name__}{tag_part}_{stamp}_curves.png"

        epochs_axis = list(range(1, epochs + 1))
        title_suffix = f" ({run_tag})" if run_tag else ""

        try:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            # Loss subplot
            axes[0].plot(epochs_axis, history["train_loss"], label="train loss")
            axes[0].plot(epochs_axis, history["eval_loss"], label="eval loss")
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("loss")
            axes[0].set_title("Loss" + title_suffix)
            axes[0].grid(True)
            axes[0].legend()

            # Accuracy subplot
            axes[1].plot(epochs_axis, history["train_acc"], label="train accuracy")
            axes[1].plot(epochs_axis, history["eval_acc"], label="eval accuracy")
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("accuracy")
            axes[1].set_title("Accuracy" + title_suffix)
            axes[1].grid(True)
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(file_name, dpi=curves_dpi)
            plt.close(fig)

            print(f"Saved training curves to: {file_name}")
        except Exception as e:
            print(f"Could not save training curves figure: {e}")

    return model, history

