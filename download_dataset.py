import os, shutil, random
from pathlib import Path
import kagglehub


# Download latest version
def download_dataset():
    path = kagglehub.dataset_download("emmarex/plantdisease")
    print("Path to dataset files:", path)
    return path


# ==============================
# 🔀 4. Split into train/val/test
# ==============================
def split_dataset(
    raw_dir="./content/PlantVillage/PlantVillage",
    out_dir="data",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]

    for split in ["train", "val", "test"]:
        for c in classes:
            (out_dir / split / c).mkdir(parents=True, exist_ok=True)

    for c in classes:
        imgs = (
            list((raw_dir / c).glob("*.jpg"))
            + list((raw_dir / c).glob("*.JPG"))
            + list((raw_dir / c).glob("*.png"))
        )
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train, val, test = (
            imgs[:n_train],
            imgs[n_train : n_train + n_val],
            imgs[n_train + n_val :],
        )
        for split, subset in zip(["train", "val", "test"], [train, val, test]):
            for img in subset:
                shutil.copy(img, out_dir / split / c / img.name)


print("✅ PlantVillage prepared at ./data with train/val/test splits")

if __name__ == "__main__":
    downloaded_data_path = download_dataset()
    path_to_images = f"{downloaded_data_path}/PlantVillage/PlantVillage"
    split_dataset(raw_dir=path_to_images, out_dir="data")
