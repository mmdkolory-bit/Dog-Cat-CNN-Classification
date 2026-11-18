from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from .data import ID_TO_CLASS


def plot_history(history, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_mse"] for row in history], marker="o", label="Train MSE")
    ax.plot(epochs, [row["val_mse"] for row in history], marker="o", label="Validation MSE")
    ax.set_title("Mean Squared Error per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (Brier score)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mse_epochs.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(matrix, output_path: str | Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix), display_labels=["Cat", "Dog"]
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sample_predictions(model, dataset, device, output_path: str | Path, count=12):
    count = min(count, len(dataset))
    if count == 0:
        return
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    cols = 4
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    model.eval()
    with torch.inference_mode():
        for ax, index in zip(axes, indices):
            image, label = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu()
            pred = int(torch.argmax(probs).item())
            display_image = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(display_image)
            confidence = float(probs[pred]) * 100
            ax.set_title(
                f"True: {ID_TO_CLASS[label]} | Pred: {ID_TO_CLASS[pred]}\n"
                f"Confidence: {confidence:.1f}%",
                color="green" if pred == label else "red", fontsize=9,
            )
            ax.axis("off")
    for ax in axes[count:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_ablation(summary, output_path: str | Path):
    names = [item["experiment"] for item in summary]
    accuracy = [item["test_accuracy"] for item in summary]
    f1 = [item["test_f1"] for item in summary]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, accuracy, width, label="Test Accuracy")
    ax.bar(x + width / 2, f1, width, label="Test F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("CNN Architecture Ablation Study")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
