
"""Validate the processed Food-11 dataset."""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from src.food11.data import CLASSES


def validate_dataset(dataset_dir, min_per_class=50, tau=0.05):
    dataset_dir = Path(dataset_dir)
    splits = ["training", "validation", "evaluation"]

    results = {}
    counts = {}
    seen_hashes = {}

    # Check 1: Exactly 11 class folders in each split
    folders_ok = True

    for split in splits:
        split_dir = dataset_dir / split

        if not split_dir.is_dir():
            folders_ok = False
            counts[split] = Counter()
            continue

        folders = {
            p.name for p in split_dir.iterdir() if p.is_dir()
        }

        if folders != set(CLASSES):
            folders_ok = False

        counts[split] = Counter()

    results["Class folders"] = folders_ok

    # Checks 2, 3 and 4
    images_ok = True
    minimum_ok = True
    duplicates_ok = True

    for split in splits:
        for category in CLASSES:
            class_dir = dataset_dir / split / category

            if not class_dir.is_dir():
                minimum_ok = False
                continue

            images = [
                p for p in class_dir.iterdir()
                if p.is_file()
            ]

            counts[split][category] = len(images)

            if len(images) < min_per_class:
                minimum_ok = False

            for path in images:
                try:
                    with Image.open(path) as img:
                        img.load()

                        if img.mode != "RGB":
                            images_ok = False

                        if img.size != (128, 128):
                            images_ok = False

                except (OSError, ValueError):
                    images_ok = False

                # Compare image hashes across splits
                file_hash = hashlib.md5(
                    path.read_bytes()
                ).hexdigest()

                if file_hash in seen_hashes:
                    if seen_hashes[file_hash] != split:
                        duplicates_ok = False
                else:
                    seen_hashes[file_hash] = split

    results["Image format and size"] = images_ok
    results["Minimum images per class"] = minimum_ok
    results["No cross-split duplicates"] = duplicates_ok

    # Check 5: Compare class distributions
    distribution_ok = True

    train_total = sum(counts["training"].values())
    val_total = sum(counts["validation"].values())

    if train_total == 0 or val_total == 0:
        distribution_ok = False
    else:
        for category in CLASSES:
            train_share = (
                counts["training"][category] / train_total
            )

            val_share = (
                counts["validation"][category] / val_total
            )

            if abs(train_share - val_share) > tau:
                distribution_ok = False

    results["Class distribution"] = distribution_ok

    # Print report
    print("\nFood-11 Dataset Validation")
    print("-" * 35)

    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check}: {status}")

    # Exit with error if any check failed
    if not all(results.values()):
        sys.exit(1)

    print("\nDataset validation successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir")
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    validate_dataset(
        args.dataset_dir,
        args.min_per_class,
        args.tau,
    )
