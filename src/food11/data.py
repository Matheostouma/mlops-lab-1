"""Prepare the Food-11 dataset for ResNet training.

Copies data/food11_raw (flat files named "<class_idx>_<n>.jpg") into
data/food11_processed and data/food11_processed_mini, resized to 128x128
and arranged as split/category/*.jpg, which is the layout torchvision's
ImageFolder (and therefore ResNet training pipelines) expect.
"""

from pathlib import Path

from PIL import Image

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "food11_raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "food11_processed"
MINI_DIR = Path(__file__).resolve().parents[2] / "data" / "food11_processed_mini"

IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100

CATEGORIES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}


def process_split(split_dir: Path) -> None:
    split = split_dir.name
    mini_counts = {category: 0 for category in CATEGORIES.values()}

    for image_path in sorted(split_dir.iterdir()):
        class_idx = int(image_path.stem.split("_")[0])
        category = CATEGORIES[class_idx]

        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize(IMAGE_SIZE)

            out_dir = PROCESSED_DIR / split / category
            out_dir.mkdir(parents=True, exist_ok=True)
            resized.save(out_dir / image_path.name)

            if mini_counts[category] < MINI_LIMIT:
                mini_out_dir = MINI_DIR / split / category
                mini_out_dir.mkdir(parents=True, exist_ok=True)
                resized.save(mini_out_dir / image_path.name)
                mini_counts[category] += 1

    print(f"{split}: processed {sum(1 for _ in split_dir.iterdir())} images")


def main() -> None:
    for split_dir in sorted(RAW_DIR.iterdir()):
        if split_dir.is_dir():
            process_split(split_dir)


if __name__ == "__main__":
    main()
