import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def unpickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")

class CIFAR10(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, normalize: bool = False):
        self.root = root
        self.split = split.lower().strip()
        self.transform = transform
        self.normalize = normalize

        cifar_dir = os.path.join(root, "cifar-10-batches-py")
        if not os.path.isdir(cifar_dir):
            raise FileNotFoundError(f"Folder not found: {cifar_dir}")

        data, labels = [], []

        if self.split == "train":
            for i in range(1, 6):
                batch = unpickle(os.path.join(cifar_dir, f"data_batch_{i}"))
                data.append(batch[b"data"])
                labels.extend(batch[b"labels"])
            self.x = np.vstack(data).astype(np.uint8)  # (50000, 3072)
            self.y = np.array(labels, dtype=np.int64)  # (50000,)
        elif self.split == "test":
            batch = unpickle(os.path.join(cifar_dir, "test_batch"))
            self.x = batch[b"data"].astype(np.uint8)   # (10000, 3072)
            self.y = np.array(batch[b"labels"], dtype=np.int64)
        else:
            raise ValueError("split must be 'train' or 'test'")

        # Source https://www.ricardodecal.com/guides/use-these-normalization-values-for-torchvision-datasets
        self.mean = torch.tensor([0.4914, 0.48216, 0.44653]).view(3,1,1)
        self.std  = torch.tensor([0.2022, 0.19932, 0.20086]).view(3,1,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        flat = self.x[idx]  # (3072,)
        img = torch.from_numpy(flat).view(3, 32, 32).float() / 255.0

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
    train_transform=None,
    test_transform=None,
):
    train_ds = CIFAR10(root=root, split="train", transform=train_transform, normalize=normalize)
    test_ds  = CIFAR10(root=root, split="test",  transform=test_transform,  normalize=normalize)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader