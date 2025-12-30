import os
import shutil
import random

SOURCE_DIR = "../dataset/BirdIdentification/training_set/training_set"
TARGET_DIR = "../dataset/BirdIdentification/training_data"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)

def split_dataset():
    for split in SPLIT_RATIO:
        os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)

        if not os.path.isdir(class_path):
            continue

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        if len(images) == 0:
            print(f"Skipping empty class: {class_name}")
            continue

        random.shuffle(images)

        total = len(images)
        train_end = int(SPLIT_RATIO["train"] * total)
        val_end = train_end + int(SPLIT_RATIO["val"] * total)

        split_map = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, files in split_map.items():
            split_class_dir = os.path.join(TARGET_DIR, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for file in files:
                src = os.path.join(class_path, file)
                dst = os.path.join(split_class_dir, file)
                shutil.copy2(src, dst)

        print(f"Processed class: {class_name}")

    print("Dataset splitting completed successfully.")

if __name__ == "__main__":
    split_dataset()
