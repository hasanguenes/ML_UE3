# gtsrb_loader.py  

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms

def compute_gtsrb_train_mean_std(root: str, img_size=(64, 64)):
    train_dir = Path(root) / "Final_Training_Images"
    if not train_dir.exists():
        raise FileNotFoundError(train_dir)

    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),   # (3,H,W) float in [0,1]
    ])

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0

    # iterate all class folders and ppm files
    for class_dir in sorted([d for d in train_dir.iterdir() if d.is_dir()]):
        for img_path in class_dir.glob("*.ppm"):
            with Image.open(img_path) as im:
                x = tfm(im.convert("RGB"))  # (3,H,W)

            h, w = x.shape[1], x.shape[2]
            num_pixels += h * w
            channel_sum += x.sum(dim=(1, 2))
            channel_sq_sum += (x ** 2).sum(dim=(1, 2))

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sq_sum / num_pixels - mean ** 2)
    return mean, std

# pre-computed mean and std values (depending on image size) manually by helper function shown above (compute_gtsrb_train_mean_std)
GTSRB_MEAN_STD_BY_SIZE  = {
    (32, 32): {
        "mean": [0.3403, 0.3121, 0.3214],  
        "std":  [0.2724, 0.2608, 0.2669],  
    },
    (64, 64): {
        "mean": [0.3403, 0.3122, 0.3215],  
        "std":  [0.2751, 0.2642, 0.2707], 
    },
}

def _lookup_gtsrb_stats(img_size):
    img_size = tuple(img_size)
    if img_size not in GTSRB_MEAN_STD_BY_SIZE:
        raise ValueError(
            f"No mean/std available for img_size={img_size}. "
            f"Available: {list(GTSRB_MEAN_STD_BY_SIZE.keys())}"
        )
    stats = GTSRB_MEAN_STD_BY_SIZE[img_size]
    return stats["mean"], stats["std"]

class GTSRBDataset(Dataset):
    """
    GTSRB Dataset for DL.

    Expected structure:
    root/
      Final_Training_Images/
        00000/*.ppm ...
      Final_Test_Images/
        *.ppm
        GT-final_test.csv  (Filename, ClassId)

    Returns:
      img: torch.float32 (3,H,W) in [0,1] (optionally normalized)
      label: torch.long
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        normalize: bool = False,
        img_size=(64, 64),                 # needed to pick correct mean/std
        csv_name: str = "GT-final_test.csv",
    ):
        self.root = Path(root)
        self.split = split.lower().strip()
        self.transform = transform
        self.normalize = normalize
        self.img_size = tuple(img_size)

        mean, std = _lookup_gtsrb_stats(self.img_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)

        if self.split == "train":
            base = self.root / "Final_Training_Images"
            if not base.exists():
                raise FileNotFoundError(f"Missing folder: {base}")

            paths, labels = [], []
            for class_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
                class_id = int(class_dir.name)
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
                img = self.transform(im)  # should yield tensor in [0,1]
            else:
                img = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0

        if self.normalize:
            img = (img - self.mean) / self.std

        return img, torch.tensor(y, dtype=torch.long)


def get_gtsrb_dataloaders(
    root: str,
    img_size=(64, 64),
    batch_size: int = 128,
    num_workers: int = 0,
    normalize: bool = False,
    debug_fraction: float = 1.0, # to just load a fraction of the data
    seed: int = 42,

):
    """
    Returns (train_loader, test_loader) for GTSRB.

    - normalize=False: tensors in [0,1]
    - normalize=True : (x - mean)/std using the pre-computed stats for the given img_size
    """
    img_size = tuple(img_size)

    tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  # -> float in [0,1]
    ])

    train_dataset = GTSRBDataset(
        root=root, split="train", transform=tfms, normalize=normalize, img_size=img_size
    )
    test_dataset = GTSRBDataset(
        root=root, split="test",  transform=tfms, normalize=normalize, img_size=img_size
    )

    if not (0.0 < debug_fraction <= 1.0):
        raise ValueError("debug_fraction must be in (0, 1].")
    
    if debug_fraction < 1.0:
        # Train subset
        n_train = max(1, int(len(train_dataset) * debug_fraction))
        g = torch.Generator().manual_seed(seed)
        train_idx = torch.randperm(len(train_dataset), generator=g)[:n_train].tolist()
        train_dataset = Subset(train_dataset, train_idx)

        # Test subset 
        n_test = max(1, int(len(test_dataset) * debug_fraction))
        g = torch.Generator().manual_seed(seed + 1) # +1 to avoid same seed as for train
        test_idx = torch.randperm(len(test_dataset), generator=g)[:n_test].tolist()
        test_dataset = Subset(test_dataset, test_idx)


    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, test_loader