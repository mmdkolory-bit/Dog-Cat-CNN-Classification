from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch import nn

from .model import count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_mse = 0.0
    total_correct = 0
    total_samples = 0
    all_labels, all_predictions, all_probabilities = [], [], []

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            use_amp = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
            predictions = (probabilities >= 0.5).long()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_mse += torch.mean((probabilities - labels.float()) ** 2).item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    result = {
        "loss": total_loss / max(total_samples, 1),
        "mse": total_mse / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "labels": np.asarray(all_labels, dtype=np.int64),
        "predictions": np.asarray(all_predictions, dtype=np.int64),
        "probabilities": np.asarray(all_probabilities, dtype=np.float32),
    }
    return result


def train_model(model, train_loader, val_loader, device, output_dir: str | Path,
                epochs: int, learning_rate: float, patience: int, image_size: int = 128):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device,
                                optimizer=optimizer, scaler=scaler)
        val_stats = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_stats["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_mse": train_stats["mse"],
            "val_mse": val_stats["mse"],
            "train_accuracy": train_stats["accuracy"],
            "val_accuracy": val_stats["accuracy"],
            "learning_rate": current_lr,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss: {row['train_loss']:.4f}/{row['val_loss']:.4f} | "
            f"MSE: {row['train_mse']:.4f}/{row['val_mse']:.4f} | "
            f"acc: {row['train_accuracy']:.4f}/{row['val_accuracy']:.4f}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state": best_state,
                "model_config": model.config,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "image_size": image_size,
            }, output_dir / "best_model.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"توقف زودهنگام در epoch {epoch}; بهترین epoch={best_epoch}")
            break

    elapsed = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_seconds": elapsed,
        "parameter_count": count_parameters(model),
        "model": model,
    }


def calculate_metrics(labels, predictions, probabilities):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "mse": float(np.mean((probabilities - labels.astype(np.float32)) ** 2)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["Cat", "Dog"],
            zero_division=0, output_dict=True,
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics
