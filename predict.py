# predict.py — run inference on a single image using your from-scratch CNN
import argparse, json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

from model import SimpleSignNet

def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    num_classes = ckpt.get("num_classes")
    img_size = ckpt.get("img_size", 224)

    model = SimpleSignNet(num_classes=num_classes).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return model, tf, img_size

def load_labels(labels_path: Path, num_classes: int):
    if labels_path.exists():
        try:
            with open(labels_path, "r") as f:
                labels = json.load(f)
            if isinstance(labels, list) and len(labels) == num_classes:
                return labels
        except Exception:
            pass
    # fallback: numeric labels
    return [str(i) for i in range(num_classes)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/simple_signnet.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, _ = load_model(args.model, device)

    # try to load label names saved during training
    labels_file = Path(args.model).parent / "labels.json"
    # num_classes is in the checkpoint; get from model head
    num_classes = model.classifier[-1].out_features
    labels = load_labels(labels_file, num_classes)

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    topk = min(args.topk, len(labels))
    vals, idxs = torch.topk(probs, k=topk)
    for v, i in zip(vals.tolist(), idxs.tolist()):
        print(f"{labels[i]:20s}  {v*100:.1f}%")

if __name__ == "__main__":
    main()
