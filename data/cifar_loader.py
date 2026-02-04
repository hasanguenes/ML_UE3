import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def unpickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")

# helper function to compute mean and std
@torch.no_grad()
def compute_mean_std_from_loader(loader):
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0

    for xb, _ in loader:  # xb: (B,3,H,W) in [0,1]
        b, c, h, w = xb.shape
        num_pixels += b * h * w
        channel_sum += xb.sum(dim=(0, 2, 3))
        channel_sq_sum += (xb ** 2).sum(dim=(0, 2, 3))

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sq_sum / num_pixels - mean ** 2)
    return mean, std

# Stats must match the pipeline: tensor in [0,1] AFTER resizing to img_size
CIFAR10_MEAN_STD_BY_SIZE = {
    (32, 32): {
        # Source: https://www.ricardodecal.com/guides/use-these-normalization-values-for-torchvision-datasets
        "mean": [0.4914, 0.48216, 0.44653],
        "std":  [0.2022, 0.19932, 0.20086],
    },
    (64, 64): {
        # manually pre-computed values by usage of helper function above () 
        "mean": [0.4914,  0.48216, 0.44653],
        "std":  [0.24043, 0.2370, 0.25569],
    },
}


def _lookup_cifar10_stats(img_size):
    img_size = tuple(img_size)
    if img_size not in CIFAR10_MEAN_STD_BY_SIZE:
        raise ValueError(
            f"No mean/std for img_size={img_size}. "
            f"Available: {list(CIFAR10_MEAN_STD_BY_SIZE.keys())}"
        )
    s = CIFAR10_MEAN_STD_BY_SIZE[img_size]
    return s["mean"], s["std"]


class CIFAR10(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        normalize: bool = False,
        img_size=(32, 32),
    ):
        self.root = root
        self.split = split.lower().strip()
        self.transform = transform
        self.normalize = normalize
        self.img_size = tuple(img_size)

        cifar_dir = os.path.join(root, "cifar-10-batches-py")
        if not os.path.isdir(cifar_dir):
            raise FileNotFoundError(f"Folder not found: {cifar_dir}")

        data, labels = [], []

        if self.split == "train":
            for i in range(1, 6):
                batch = unpickle(os.path.join(cifar_dir, f"data_batch_{i}"))
                data.append(batch[b"data"])
                labels.extend(batch[b"labels"])
            self.x = np.vstack(data).astype(np.uint8)   # (50000, 3072)
            self.y = np.array(labels, dtype=np.int64)   # (50000,)
        elif self.split == "test":
            batch = unpickle(os.path.join(cifar_dir, "test_batch"))
            self.x = batch[b"data"].astype(np.uint8)    # (10000, 3072)
            self.y = np.array(batch[b"labels"], dtype=np.int64)
        else:
            raise ValueError("split must be 'train' or 'test'")

        # mean/std depend on the resized resolution
        mean, std = _lookup_cifar10_stats(self.img_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        flat = self.x[idx]  # (3072,)
        img = torch.from_numpy(flat).view(3, 32, 32).float() / 255.0  # [0,1]

        # we just use it with 32x32
        # optional resize to img_size (e.g., 64x64)
        if self.img_size != (32, 32):
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.img_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        # optional extra transforms (should expect a tensor)
        if self.transform is not None:
            img = self.transform(img)

        if self.normalize:
            img = (img - self.mean) / self.std

        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return img, label


def get_cifar10_dataloaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 0,
    normalize: bool = False,
    img_size=(32, 32),
    train_transform=None,
    test_transform=None,
):
    train_ds = CIFAR10(
        root=root, split="train",
        transform=train_transform, normalize=normalize, img_size=img_size
    )
    test_ds = CIFAR10(
        root=root, split="test",
        transform=test_transform, normalize=normalize, img_size=img_size
    )

    pin = torch.cuda.is_available() or torch.backends.mps.is_available()  # <-- MPS hinzugefügt

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    return train_loader, test_loader
