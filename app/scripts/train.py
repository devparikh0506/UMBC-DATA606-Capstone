from __future__ import annotations

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

from dataclasses import dataclass


@dataclass
class TrainConfig:
    csv: str  # path to EEG CSV file
    save_prefix: str = "outputs/eegnet"
    epochs: int = 250
    batch: int = 64
    lr: float = 0.001
    seed: int = 42


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    total = 0
    preds_all, labels_all = [], []
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)  # (B, 1, C, T)
        y = y.to(device).long()
        optimizer.zero_grad()
        logits = model(x)  # raw logits
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(y.cpu())
        total += x.size(0)

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()
    return running_loss / total, accuracy_score(labels_all, preds_all)


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    total = 0
    preds_all, labels_all = [], []
    for x, y in tqdm(dataloader, desc="Validating", leave=False):
        x = x.to(device)
        y = y.to(device).long()
        logits = model(x)
        loss = loss_fn(logits, y)
        running_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(y.cpu())
        total += x.size(0)

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()
    return running_loss / total, accuracy_score(labels_all, preds_all)




def run_training(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    loss_fn,
    device,
    epochs,
    batch_size=64,
    save_prefix="outputs/model",
    scheduler=None,
    early_stopping=None,
    label_map=None,
):
    """Reusable training loop with DataLoader creation inside."""

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}

    # Paths
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(save_prefix), "logs"), exist_ok=True)
    best_path = f"{save_prefix}_best.pth"
    last_path = f"{save_prefix}_last.pth"
    log_path = f"{save_prefix}_log.csv"


    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", leave=True)

    for epoch in epoch_bar:
        # ---- Train one epoch ----
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # ---- Validate ----
        va_loss, va_acc = validate(model, val_loader, loss_fn, device)

        # LR schedule
        if scheduler is not None:
            scheduler.step(va_loss)

        # Save history
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # Early stopping check
        improved = False
        if early_stopping:
            improved = early_stopping.step({"val_loss": va_loss, "val_acc": va_acc}, model, epoch)
            if improved:
                torch.save(model.state_dict(), best_path)

        # Progress bar logging
        postfix_dict = {
            "Train Loss": f"{tr_loss:.3f}",
            "Train Acc": f"{tr_acc:.3f}",
            "Val Loss": f"{va_loss:.3f}",
            "Val Acc": f"{va_acc:.3f}",
        }
        if early_stopping:
            metrics_str = ', '.join([f"{k}={v:.6f}" for k, v in early_stopping.best_metrics.items()])
            postfix_dict["Best Monitored"] = (
                f"{early_stopping.monitor}={early_stopping.best_score:.6f} @ epoch {early_stopping.best_epoch} "
                f"(Best Epoch Info: {metrics_str})"
            )
            postfix_dict["Status"] = (
                f"[Epoch {epoch}] Improved {early_stopping.monitor} → {early_stopping.best_score:.6f}"
                if improved else f"[Epoch {epoch}] No improvement. Patience {early_stopping.counter}/{early_stopping.patience}"
            )
        epoch_bar.set_postfix_str(str(postfix_dict))

        if early_stopping and early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final and restore best
    torch.save(model.state_dict(), last_path)
    if early_stopping and early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
        torch.save(model.state_dict(), best_path)
        print("Restored best model weights.")

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(log_path, index=False)


    # Summary
    if early_stopping and early_stopping.best_metrics:
        best_loss = early_stopping.best_metrics["val_loss"]
        best_acc = early_stopping.best_metrics["val_acc"]
        print(early_stopping.best_metrics)
        print(f"\nBest {early_stopping.monitor} = {early_stopping.best_score:.6f} @ epoch {early_stopping.best_epoch}")
        print(f"   Val Loss = {best_loss:.6f}")
        print(f"   Val Acc  = {best_acc:.4f}")
    else:
        print(f"\nBest Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"Training finished. Best model: {best_path} | Last model: {last_path}")

    return model, hist_df, best_path
from dataclasses import dataclass
