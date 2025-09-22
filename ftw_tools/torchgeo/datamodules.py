"""FTW datamodule."""

from typing import Any, Optional

import kornia
import kornia.augmentation as K
import kornia.constants
import torch
from lightning import LightningDataModule
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from ftw_tools.torchgeo.datasets import FTW


def preprocess(sample):
    sample["image"] = sample["image"] / 3000
    return sample


def randomChannelShuffle(x):
    if torch.rand(1) < 0.5:
        return x
    return torch.cat([x[:, 4:8], x[:, :4]], dim=1)


class FTWDataModule(LightningDataModule):
    """LightningDataModule implementation for the FTW dataset."""

    mean = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    std = torch.tensor([3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000])

    def __init__(
        self,
        root: str = "data/ftw/",
        batch_size: int = 64,
        num_workers: int = 0,
        train_countries: list[str] = ["france"],
        val_countries: list[str] = ["france"],
        test_countries: list[str] = ["france"],
        temporal_options: str = "stacked",
        num_samples: int = -1,
        random_shuffle: bool = False,
        resize_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new FTWDataModule instance.

        Note: you can pass train_batch_size, val_batch_size, test_batch_size to
            control the batch sizes of each DataLoader independently.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            train_countries: List of countries to use training splits from
            val_countries: List of countries to use validation splits from
            test_countries: List of countries to use test splits from
            random_shuffle: Whether to use random channel shuffle augmentation
            resize_factor: Optional resize factor to upsample the images
            **kwargs: Additional keyword arguments passed to
                :class:`~src.datasets.FTW`.
        """
        super().__init__()
        if "split" in kwargs:
            raise ValueError("Cannot specify split in FTWDataModule")

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_countries = train_countries
        self.val_countries = val_countries
        self.test_countries = test_countries
        self.load_boundaries = kwargs.pop("load_boundaries", False)
        self.temporal_options = temporal_options
        self.num_samples = num_samples
        self.ignore_sample_fn = kwargs.pop("ignore_sample_fn", None)

        # for the temporal option windowA, windowB and median we will have 4 channel input
        if self.temporal_options in ("windowA", "windowB", "median", "random_window"):
            self.mean = torch.tensor([0, 0, 0, 0])
            self.std = torch.tensor([3000, 3000, 3000, 3000])
        elif (
            self.temporal_options == "rgb"
        ):  # for the rgb temporal option we are just selecting these 3 channls from both window_a and window_b images
            self.mean = torch.tensor([0, 0, 0, 0, 0, 0])
            self.std = torch.tensor([3000, 3000, 3000, 3000, 3000, 3000])

        print("Loaded datamodule with:")
        print(f"Train countries: {self.train_countries}")
        print(f"Val countries: {self.val_countries}")
        print(f"Test countries: {self.test_countries}")
        print(f"Number of samples: {self.num_samples}")

        augs = [
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
        ]
        if random_shuffle:
            print("Using random channel shuffle augmentation")
            augs.append(kornia.contrib.Lambda(randomChannelShuffle))

        if resize_factor is not None:
            if resize_factor < 1:
                raise ValueError("Resize factor must be >= 1")
            print(f"Using resize factor of {resize_factor}")
            augs.append(
                K.Resize(
                    (int(256 * resize_factor), int(256 * resize_factor)),
                    resample=kornia.constants.Resample.BILINEAR,
                    antialias=True,
                )
            )

        print("Augmentations:")
        for aug in augs:
            print(aug)

        self.train_aug = K.AugmentationSequential(*augs, data_keys=None)
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None
        )

    def setup(self, stage: str):
        if stage in ["fit"]:
            self.train_dataset = FTW(
                root=self.root,
                countries=self.train_countries,
                split="train",
                load_boundaries=self.load_boundaries,
                temporal_options=self.temporal_options,
                num_samples=self.num_samples,
                ignore_sample_fn=self.ignore_sample_fn,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = FTW(
                root=self.root,
                countries=self.val_countries,
                split="val",
                load_boundaries=self.load_boundaries,
                temporal_options=self.temporal_options,
                num_samples=self.num_samples,
            )
        if stage == "test":
            self.test_dataset = FTW(
                root=self.root,
                countries=self.test_countries,
                split="test",
                load_boundaries=self.load_boundaries,
                temporal_options=self.temporal_options,
                num_samples=self.num_samples,
            )

    def train_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def on_after_batch_transfer(self, batch: dict[str, Tensor], dataloader_idx: int):
        if self.trainer:
            if self.trainer.training:
                batch = self.train_aug(batch)
            else:
                batch = self.aug(batch)
        return batch

    def plot(self, *args: Any, **kwargs: Any):
        fig: Figure | None = None
        dataset = self.val_dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        if dataset is not None:
            if hasattr(dataset, "plot"):
                fig = dataset.plot(*args, **kwargs)
        return fig
