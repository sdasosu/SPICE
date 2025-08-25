"""
Dataset handling for pruning with compatibility for existing data module
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from data.data import SegmentationDataset, get_loaders


class PruningDataHandler:
    """Handles data loading for pruning process"""

    def __init__(
        self, data_root: str, batch_size: int = 4, num_workers: int = 2, seed: int = 42
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return get_loaders(
            data_root=self.data_root,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
        )

    def get_train_val_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader, val_loader, _ = self.get_dataloaders()
        return train_loader, val_loader

    def get_calibration_dataloader(self) -> DataLoader:
        from data.data import worker_init_fn

        train_set = SegmentationDataset(self.data_root, split="train")
        pin = torch.cuda.is_available()
        generator = torch.Generator()
        generator.manual_seed(42)

        calibration_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

        return calibration_loader
