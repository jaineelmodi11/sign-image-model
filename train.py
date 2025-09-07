# train.py — train your from-scratch CNN on image folders
import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from tqdm import tqdm

from model import SimpleSignNet


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.10),
        transforms.RandomGrayscale(p=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return train_tf, val_tf


def make_loaders(data_dir, img_size, batch_size, workers):
    train_tf, val_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(Path(data_dir) / "val",   transform=val_tf)

    # handle class imbalance with a weighted sampler
    counts = np.bincount(train_ds.targets)
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = weights[train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, logits, target):
        n = logits.size(-1)
        logp = self.log_softmax(logits)
        with torch.no_grad():
            true = torch.zeros_like(logp).fill_(self.smoothing/(n-1))
            true.scatter_(1, target.unsqueeze(1), 1.0-self.smoothing)
        return torch.mean(torch.sum(-true*logp, dim=-1))


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        all_pred.append(preds.cpu()); all_true.append(y.cpu())
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    acc = (y_true == y_pred).mean() * 100.0
    return report, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="model")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = make_loaders(args.data_dir, args.img_size, args.batch_size, args.workers)
    with open(Path(args.out_dir)/"labels.json", "w") as f:
        json.dump(classes, f, indent=2)

    model = SimpleSignNet(num_classes=len(classes)).to(device)
    criterion = LabelSmoothingCE(0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = -1.0
    best_path = Path(args.out_dir)/"simple_signnet.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        run_loss = 0.0; seen = 0; correct = 0
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            pbar.set_postfix(loss=f"{run_loss/seen:.4f}", acc=f"{100*correct/seen:.2f}%")

        report, val_acc = evaluate(model, val_loader, device, classes)
        print("\nValidation report:\n", report)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "num_classes": len(classes),
                        "img_size": args.img_size},
                        best_path)
            print(f"✓ Saved new best to {best_path} (val acc {best_acc:.2f}%)")

    print(f"Best val acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
