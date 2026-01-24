# gt_loader.py  (GTSRB / "GT" loader for DL)
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class GT(Dataset):
    """
    GTSRB (GT) Dataset for DL.

    Expected structure:
    root/
      Final_Training_Images/
        00000/*.ppm
        00001/*.ppm
        ...
      Final_Test_Images/
        *.ppm
        GT-final_test.csv  (must contain Filename and ClassId)

    Returns:
      img: torch.float32 (3,H,W) in [0,1] (or normalized if normalize=True)
      label: torch.long
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        normalize: bool = False,
        csv_name: str = "GT-final_test.csv",
    ):
        self.root = Path(root)
        self.split = split.lower().strip()
        self.transform = transform
        self.normalize = normalize

        # TODO: have to compute mean!!
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.std  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

        if self.split == "train":
            base = self.root / "Final_Training_Images"
            if not base.exists():
                raise FileNotFoundError(f"Missing folder: {base}")

            paths, labels = [], []
            # folders are typically "00000".."00042"
            for class_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
                class_id = int(class_dir.name)  # robust: use folder name as label
                for p in sorted(class_dir.glob("*.ppm")):
                    paths.append(p)
                    labels.append(class_id)

            self.paths = paths
            self.labels = np.asarray(labels, dtype=np.int64)

        elif self.split == "test":
            base = self.root / "Final_Test_Images"
            csv_path = base / csv_name
            if not base.exists():
                raise FileNotFoundError(f"Missing folder: {base}")
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing CSV: {csv_path}")

            df = pd.read_csv(csv_path, sep=None, engine="python")
            df.columns = [c.strip() for c in df.columns]

            if "Filename" not in df.columns or "ClassId" not in df.columns:
                raise ValueError(f"CSV must contain Filename and ClassId, got {df.columns.tolist()}")

            self.paths = [base / fn for fn in df["Filename"].astype(str).tolist()]
            self.labels = df["ClassId"].astype(int).to_numpy(dtype=np.int64)

        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        y = int(self.labels[idx])

        with Image.open(p) as im:
            im = im.convert("RGB")

            if self.transform is not None:
                img = self.transform(im)  # should produce tensor in [0,1] if ToTensor is included
            else:
                # fallback: tensor in [0,1]
                img = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0

        if self.normalize:
            img = (img - self.mean) / self.std

        return img, torch.tensor(y, dtype=torch.long)


def get_gt_dataloaders(
    root: str,
    img_size=(64, 64),
    batch_size: int = 128,
    num_workers: int = 0,
    normalize: bool = False,
):
    """
    Returns (train_loader, test_loader) for GT (GTSRB).
    Resizing is done here via transforms.Resize(img_size).
    """
    tfms = [transforms.Resize(img_size), transforms.ToTensor()]  # -> float in [0,1]

    train_tfm = transforms.Compose(tfms)
    test_tfm  = transforms.Compose(tfms)

    train_ds = GT(root=root, split="train", transform=train_tfm, normalize=normalize)
    test_ds  = GT(root=root, split="test",  transform=test_tfm,  normalize=normalize)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, test_loader
