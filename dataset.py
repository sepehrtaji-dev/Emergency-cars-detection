import os
import shutil
import random
import json
from pathlib import Path

DATASET_ROOT = Path("emergency_dataset_with_label")
OUTPUT_ROOT  = Path("dataset")
DATA_DIRS    = ["data1", "data2"]

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10

SEED         = 42
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def read_classes(data_dir: Path) -> list[str]:
    classes_file = data_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found in {data_dir}")
    with open(classes_file) as f:
        return [line.strip() for line in f if line.strip()]

def collect_samples(data_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found in {data_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels/ not found in {data_dir}")

    samples = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            samples.append((img_path, label_path))
        else:
            print(f"  [WARN] No label for {img_path.name} — skipping")

    return samples

def merge_and_split():
    all_classes = None
    for dir_name in DATA_DIRS:
        data_dir = DATASET_ROOT / dir_name
        if not data_dir.exists():
            print(f"[WARN] {data_dir} not found — skipping")
            continue
        classes = read_classes(data_dir)
        if all_classes is None:
            all_classes = classes
        elif all_classes != classes:
            print(f"[WARN] classes.txt mismatch in {dir_name} — using first one")

    if all_classes is None:
        raise RuntimeError("No valid data directories found.")

    print(f"Classes ({len(all_classes)}): {all_classes}")

    all_samples = []
    for dir_name in DATA_DIRS:
        data_dir = DATASET_ROOT / dir_name
        if not data_dir.exists():
            continue
        samples = collect_samples(data_dir)
        print(f"  {dir_name}: {len(samples)} samples")
        all_samples.extend(samples)

    print(f"Total samples: {len(all_samples)}")

    random.seed(SEED)
    random.shuffle(all_samples)

    n       = len(all_samples)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": all_samples[:n_train],
        "val":   all_samples[n_train:n_train + n_val],
        "test":  all_samples[n_train + n_val:],
    }

    for split, samples in splits.items():
        print(f"  {split}: {len(samples)} samples")

    for split, samples in splits.items():
        img_out = OUTPUT_ROOT / split / "images"
        lbl_out = OUTPUT_ROOT / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for i, (img_path, lbl_path) in enumerate(samples):
            new_name = f"{i:06d}_{img_path.name}"
            shutil.copy2(img_path, img_out / new_name)
            shutil.copy2(lbl_path, lbl_out / f"{i:06d}_{lbl_path.name}")

    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    yaml_content = f"""# Emergency Car Detection Dataset
path: {OUTPUT_ROOT.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(all_classes)}
names: {all_classes}
"""

    yaml_path.write_text(yaml_content)
    print(f"\nWrote {yaml_path.resolve()}")


if __name__ == "__main__":
    merge_and_split()