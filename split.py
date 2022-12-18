import pathlib
import shutil
import argparse
import numpy as np


def split(
        image_path, label_path,
        train_image_path="train_image",
        train_label_path="train_label",
        validate_image_path="validate_image",
        validate_label_path="validate_label",
        holdout=0.2):

    print(f"Image path: {image_path}")
    print(f"Label path: {label_path}")

    image_path = pathlib.Path(image_path)
    label_path = pathlib.Path(label_path)
    image_files = np.array(sorted(image_path.glob("*")))
    label_files = np.array(sorted(label_path.glob("*")))

    train_image_path = pathlib.Path(train_image_path)
    train_label_path = pathlib.Path(train_label_path)
    validate_image_path = pathlib.Path(validate_image_path)
    validate_label_path = pathlib.Path(validate_label_path)
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_label_path.mkdir(parents=True, exist_ok=True)
    validate_image_path.mkdir(parents=True, exist_ok=True)
    validate_label_path.mkdir(parents=True, exist_ok=True)

    if len(image_files) != len(label_files):
        raise ValueError("Image and Label file counts must match")

    n_samples = len(image_files)
    n_holdout = int(holdout * n_samples)

    i_random = np.random.choice(n_samples, size=n_samples, replace=False)
    i_train = i_random[n_holdout:]
    i_validate = i_random[:n_holdout]

    for i in i_train:
        print(f"cp {image_files[i]} {train_image_path}")
        shutil.copy(image_files[i], train_image_path)
        print(f"cp {label_files[i]} {train_label_path}")
        shutil.copy(label_files[i], train_label_path)

    for i in i_validate:
        print(f"cp {image_files[i]} {validate_image_path}")
        shutil.copy(image_files[i], validate_image_path)
        print(f"cp {label_files[i]} {validate_label_path}")
        shutil.copy(label_files[i], validate_label_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("label_path", type=str)
    parser.add_argument("--train-image-path", type=str, default="train_image")
    parser.add_argument("--train-label-path", type=str, default="train_label")
    parser.add_argument("--validate-image-path", type=str, default="validate_image")
    parser.add_argument("--validate-label-path", type=str, default="validate_label")
    parser.add_argument("--holdout", type=float, default=0.2)

    args = parser.parse_args()
    split(**vars(args))
