from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    CatsDogsDataset, ID_TO_CLASS, discover_images, make_loaders,
    make_transforms, stratified_split,
)
from src.engine import calculate_metrics, run_epoch, train_model
from src.model import CatsDogsCNN, count_parameters
from src.plots import (
    plot_ablation, plot_confusion_matrix, plot_history, plot_sample_predictions,
)
from src.report import generate_report


VARIANTS = [
    {"name": "CNN_2Blocks", "filters": [32, 64], "use_batchnorm": True, "use_dropout": True},
    {"name": "CNN_4Blocks", "filters": [32, 64, 128, 256], "use_batchnorm": True, "use_dropout": True},
    {"name": "CNN_NoBatchNorm", "filters": [32, 64, 128], "use_batchnorm": False, "use_dropout": True},
    {"name": "CNN_NoDropout", "filters": [32, 64, 128], "use_batchnorm": True, "use_dropout": False},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN for Cat-vs-Dog image classification."
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to extracted Kaggle dataset (folder containing Cat/Dog).")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-ablation", action="store_true",
                        help="Run architecture/regularization comparisons after the main model.")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Run comparisons only; reuse the previously trained main model metrics.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_splits(train_records, val_records, test_records, output_path: Path):
    rows = []
    for split_name, records in [
        ("train", train_records), ("validation", val_records), ("test", test_records)
    ]:
        rows.extend({"path": path, "label": label,
                     "class_name": ID_TO_CLASS[label], "split": split_name}
                    for path, label in records)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def evaluate_and_save(model, test_loader, device, out_dir: Path, test_dataset=None):
    criterion = torch.nn.CrossEntropyLoss()
    result = run_epoch(model, test_loader, criterion, device)
    metrics = calculate_metrics(result["labels"], result["predictions"], result["probabilities"])
    save_json(metrics, out_dir / "test_metrics.json")
    pd.DataFrame(metrics["confusion_matrix"], index=["Actual_Cat", "Actual_Dog"],
                 columns=["Predicted_Cat", "Predicted_Dog"]).to_csv(
                     out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    report = pd.DataFrame(metrics["classification_report"]).T
    report.to_csv(out_dir / "classification_report.csv", encoding="utf-8-sig")
    predictions = pd.DataFrame({
        "true_label": result["labels"],
        "predicted_label": result["predictions"],
        "dog_probability": result["probabilities"],
    })
    predictions["true_class"] = predictions["true_label"].map(ID_TO_CLASS)
    predictions["predicted_class"] = predictions["predicted_label"].map(ID_TO_CLASS)
    predictions.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    plot_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    if test_dataset is not None:
        plot_sample_predictions(model, test_dataset, device,
                                out_dir / "sample_predictions.png")
    print("\nنتایج مجموعه آزمون:")
    for key in ["accuracy", "precision", "recall", "f1_score", "mse", "roc_auc"]:
        print(f"  {key}: {metrics.get(key)}")
    print("  Confusion matrix [Cat, Dog]:", metrics["confusion_matrix"])
    return metrics


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise ValueError("epochs و batch-size باید مثبت و image-size حداقل 32 باشد.")

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_images(args.data_dir)
    train_records, val_records, test_records = stratified_split(records, args.seed)
    save_splits(train_records, val_records, test_records, output_dir / "data_splits.csv")

    counts = {
        "cat": sum(label == 0 for _, label in records),
        "dog": sum(label == 1 for _, label in records),
        "total": len(records),
    }
    split_counts = {
        "train": len(train_records),
        "validation": len(val_records),
        "test": len(test_records),
    }
    save_json({"dataset": counts, "splits": split_counts,
               "seed": args.seed, "image_size": args.image_size},
              output_dir / "dataset_summary.json")
    print(f"تقسیم داده: train={len(train_records):,}, val={len(val_records):,}, test={len(test_records):,}")

    train_loader, val_loader, test_loader, eval_transform = make_loaders(
        train_records, val_records, test_records, args.image_size,
        args.batch_size, args.workers, device, args.seed,
    )
    test_dataset = CatsDogsDataset(test_records, eval_transform)

    main_config = {
        "filters": [32, 64, 128],
        "use_batchnorm": True,
        "use_dropout": True,
        "num_classes": 2,
    }
    main_dir = output_dir / "main"
    main_metrics = None
    trained = None
    if args.ablation_only:
        required = [main_dir / "best_model.pth", main_dir / "test_metrics.json", main_dir / "run_summary.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "برای --ablation-only ابتدا مدل اصلی را اجرا کنید. فایل‌های پیدا نشده: "
                + ", ".join(missing)
            )
        main_metrics = json.loads((main_dir / "test_metrics.json").read_text(encoding="utf-8"))
        main_summary = json.loads((main_dir / "run_summary.json").read_text(encoding="utf-8"))
        print("مدل اصلی قبلاً آموزش داده شده است؛ معیارهای ذخیره‌شده برای مقایسه استفاده می‌شوند.")
    else:
        model = CatsDogsCNN(
            filters=tuple(main_config["filters"]),
            use_batchnorm=True, use_dropout=True,
        ).to(device)
        set_seed(args.seed)
        train_loader.generator.manual_seed(args.seed)
        print(f"\nمدل اصلی: سه بلوک کانولوشنی | پارامترهای قابل آموزش: {count_parameters(model):,}")
        trained = train_model(
            model, train_loader, val_loader, device, main_dir,
            epochs=args.epochs, learning_rate=args.learning_rate,
            patience=args.patience, image_size=args.image_size,
        )
        pd.DataFrame(trained["history"]).to_csv(
            main_dir / "history.csv", index=False, encoding="utf-8-sig")
        plot_history(trained["history"], main_dir)
        main_metrics = evaluate_and_save(
            trained["model"], test_loader, device, main_dir, test_dataset
        )
        best_main_epoch = next(row for row in trained["history"] if row["epoch"] == trained["best_epoch"])
        save_json({
            "model_config": main_config,
            "best_epoch": trained["best_epoch"],
            "best_val_accuracy": best_main_epoch["val_accuracy"],
            "best_validation_loss": trained["best_val_loss"],
            "training_seconds": trained["training_seconds"],
            "parameter_count": trained["parameter_count"],
            "test_metrics": {key: main_metrics[key] for key in
                              ["accuracy", "precision", "recall", "f1_score", "mse", "roc_auc"]},
        }, main_dir / "run_summary.json")

    experiment_rows = [{
        "experiment": "CNN_3Blocks_Baseline",
        "blocks": 3,
        "batchnorm": True,
        "dropout": True,
        "filters": "32-64-128",
        "best_epoch": (trained["best_epoch"] if trained else main_summary["best_epoch"]),
        "best_val_accuracy": (best_main_epoch["val_accuracy"] if trained else main_summary["best_val_accuracy"]),
        "best_val_loss": (trained["best_val_loss"] if trained else main_summary["best_validation_loss"]),
        "training_seconds": (trained["training_seconds"] if trained else main_summary["training_seconds"]),
        "parameter_count": (trained["parameter_count"] if trained else main_summary["parameter_count"]),
        "test_accuracy": main_metrics["accuracy"],
        "test_precision": main_metrics["precision"],
        "test_recall": main_metrics["recall"],
        "test_f1": main_metrics["f1_score"],
        "test_mse": main_metrics["mse"],
    }]

    if args.run_ablation or args.ablation_only:
        print("\n========== شروع آزمایش اثر معماری و لایه‌ها ==========")
        ablation_root = output_dir / "ablations"
        for variant in VARIANTS:
            print(f"\n--- {variant['name']} ---")
            set_seed(args.seed)
            variant_dir = ablation_root / variant["name"]
            variant_model = CatsDogsCNN(
                filters=tuple(variant["filters"]),
                use_batchnorm=variant["use_batchnorm"],
                use_dropout=variant["use_dropout"],
            ).to(device)
            set_seed(args.seed)
            train_loader.generator.manual_seed(args.seed)
            variant_result = train_model(
                variant_model, train_loader, val_loader, device, variant_dir,
                epochs=args.epochs, learning_rate=args.learning_rate,
                patience=args.patience, image_size=args.image_size,
            )
            pd.DataFrame(variant_result["history"]).to_csv(
                variant_dir / "history.csv", index=False, encoding="utf-8-sig")
            plot_history(variant_result["history"], variant_dir)
            variant_metrics = evaluate_and_save(
                variant_result["model"], test_loader, device, variant_dir
            )
            best_variant_epoch = next(row for row in variant_result["history"]
                                      if row["epoch"] == variant_result["best_epoch"])
            experiment_rows.append({
                "experiment": variant["name"],
                "blocks": len(variant["filters"]),
                "batchnorm": variant["use_batchnorm"],
                "dropout": variant["use_dropout"],
                "filters": "-".join(map(str, variant["filters"])),
                "best_epoch": variant_result["best_epoch"],
                "best_val_accuracy": best_variant_epoch["val_accuracy"],
                "best_val_loss": variant_result["best_val_loss"],
                "training_seconds": variant_result["training_seconds"],
                "parameter_count": variant_result["parameter_count"],
                "test_accuracy": variant_metrics["accuracy"],
                "test_precision": variant_metrics["precision"],
                "test_recall": variant_metrics["recall"],
                "test_f1": variant_metrics["f1_score"],
                "test_mse": variant_metrics["mse"],
            })
        ablation_root.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(experiment_rows)
        summary_df.to_csv(ablation_root / "experiment_summary.csv",
                          index=False, encoding="utf-8-sig")
        plot_ablation(experiment_rows, ablation_root / "architecture_comparison.png")
        print("\nمقایسه معماری‌ها:")
        print(summary_df[["experiment", "test_accuracy", "test_f1", "test_mse"]].to_string(index=False))

    generate_report(output_dir, counts, split_counts, image_size=args.image_size,
                    epochs=args.epochs, learning_rate=args.learning_rate,
                    batch_size=args.batch_size, seed=args.seed)
    print(f"\nپایان اجرا. خروجی‌ها در این مسیر ذخیره شدند: {output_dir.resolve()}")
    print(f"گزارش خودکار: {(output_dir / 'REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
