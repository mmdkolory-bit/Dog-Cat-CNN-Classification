from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.data import ID_TO_CLASS
from src.model import build_model


def main():
    parser = argparse.ArgumentParser(description="Predict Cat/Dog class for one image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--checkpoint", default="outputs/main/best_model.pth")
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_size = args.image_size or checkpoint.get("image_size", 128)
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    with Image.open(image_path) as image:
        tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    prediction = int(torch.argmax(probabilities))
    print(f"نتیجه: {ID_TO_CLASS[prediction]}")
    print(f"احتمال گربه: {probabilities[0].item() * 100:.2f}%")
    print(f"احتمال سگ: {probabilities[1].item() * 100:.2f}%")


if __name__ == "__main__":
    main()
