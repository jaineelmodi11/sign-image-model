"""
Split Kaggle 'ASL Alphabet' training images into data/train and data/val.

Usage:
  python scripts/prepare_asl_alphabet.py \
    --src /path/to/asl_alphabet_train/asl_alphabet_train \
    --dst data --val_ratio 0.10
"""
import argparse, random, shutil
from pathlib import Path

def copy_split(src_class_dir: Path, dst_train: Path, dst_val: Path, val_ratio: float):
    imgs = [p for p in src_class_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    val_set = set(imgs[:n_val])
    for p in imgs:
        out_dir = dst_val if p in val_set else dst_train
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out_dir / p.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with 29 label subfolders from Kaggle")
    ap.add_argument("--dst", default="data", help="Output root (creates data/train and data/val)")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.src)
    dst = Path(args.dst)
    (dst/"train").mkdir(parents=True, exist_ok=True)
    (dst/"val").mkdir(parents=True, exist_ok=True)

    classes = [d for d in src.iterdir() if d.is_dir()]
    if not classes:
        raise SystemExit(f"No class folders found in {src}")

    for cls_dir in classes:
        cls = cls_dir.name
        print(f"[+] Splitting {cls}")
        copy_split(cls_dir, dst/"train"/cls, dst/"val"/cls, args.val_ratio)

    print("Done. Train/val folders created under", dst)

if __name__ == "__main__":
    main()

