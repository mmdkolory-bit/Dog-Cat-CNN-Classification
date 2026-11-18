from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_ID = {"cat": 0, "dog": 1}
ID_TO_CLASS = {0: "Cat", 1: "Dog"}


def _label_from_path(path: Path) -> int | None:
    """Infer a label from a class directory or a conventional image filename."""
    for parent in path.parents:
        token = parent.name.lower().replace("_", " ").strip()
        if token in {"cat", "cats", "cat images", "catimage", "catimages"}:
            return 0
        if token in {"dog", "dogs", "dog images", "dogimage", "dogimages"}:
            return 1

    stem = path.stem.lower()
    if re.match(r"^cats?(?:[._ -]|\d|$)", stem):
        return 0
    if re.match(r"^dogs?(?:[._ -]|\d|$)", stem):
        return 1
    return None


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width >= 16 and image.height >= 16
    except (OSError, ValueError, SyntaxError):
        return False


def discover_images(data_dir: str | Path) -> list[tuple[str, int]]:
    """Recursively discover readable Cat/Dog images below data_dir.

    Supports PetImages/Cat and PetImages/Dog, nested folders, and filenames
    such as cat.123.jpg / dog.123.jpg. Corrupt or unlabelled files are skipped.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"دیتاست پیدا نشد: {root}")
    if root.is_file():
        raise NotADirectoryError(f"مسیر دیتاست باید پوشه باشد: {root}")

    records: list[tuple[str, int]] = []
    corrupt_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = _label_from_path(path)
        if label is None:
            continue
        if _is_valid_image(path):
            records.append((str(path), label))
        else:
            corrupt_count += 1

    class_counts = {name: sum(label == idx for _, label in records)
                    for name, idx in CLASS_TO_ID.items()}
    if not records or any(count == 0 for count in class_counts.values()):
        raise RuntimeError(
            "تصویر کافی از هر دو کلاس پیدا نشد. ساختار مورد انتظار شبیه "
            "PetImages/Cat و PetImages/Dog است. "
            f"تعداد شناسایی‌شده: {class_counts}"
        )

    print(f"تصاویر معتبر: {len(records):,} | تصاویر خراب/ناخوانا: {corrupt_count:,}")
    print(f"توزیع کلاس‌ها: گربه={class_counts['cat']:,}، سگ={class_counts['dog']:,}")
    return records


def stratified_split(records: list[tuple[str, int]], seed: int = 42):
    """Create reproducible, stratified 70/15/15 train/validation/test splits."""
    paths = np.array([item[0] for item in records])
    labels = np.array([item[1] for item in records], dtype=np.int64)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=seed,
        stratify=temp_labels
    )
    return (
        list(zip(train_paths.tolist(), train_labels.tolist())),
        list(zip(val_paths.tolist(), val_labels.tolist())),
        list(zip(test_paths.tolist(), test_labels.tolist())),
    )


class CatsDogsDataset(Dataset):
    def __init__(self, records: Iterable[tuple[str, int]], transform=None):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, label = self.records[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
            return image, int(label)
        except Exception as exc:
            raise RuntimeError(f"خطا در خواندن تصویر {path}: {exc}") from exc


def make_transforms(image_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


def make_loaders(train_records, val_records, test_records, image_size: int,
                 batch_size: int, workers: int, device: torch.device, seed: int = 42):
    train_tf, eval_tf = make_transforms(image_size)
    pin_memory = device.type == "cuda"
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        CatsDogsDataset(train_records, train_tf), shuffle=True,
        generator=train_generator, **common
    )
    val_loader = DataLoader(
        CatsDogsDataset(val_records, eval_tf), shuffle=False, **common
    )
    test_loader = DataLoader(
        CatsDogsDataset(test_records, eval_tf), shuffle=False, **common
    )
    return train_loader, val_loader, test_loader, eval_tf
