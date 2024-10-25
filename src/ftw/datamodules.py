"""FTW datamodule."""

from typing import Any, Optional

import kornia.augmentation as K
import torch
from matplotlib.figure import Figure
from torch.utils.data import Subset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from .datasets import FTW


def preprocess(sample):
    sample["image"] = sample["image"] / 3000
    return sample


class FTWDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the FTW dataset."""

    mean = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    std = torch.tensor([3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000])

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        train_countries: list[str] = ["france"],
        val_countries: list[str] = ["france"],
        test_countries: list[str] = ["france"],
        temporal_options: str = "stacked" ,
        num_samples: int = -1,
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
            **kwargs: Additional keyword arguments passed to
                :class:`~src.datasets.FTW`.
        """
        if "split" in kwargs:
            raise ValueError("Cannot specify split in FTWDataModule")

        self.train_countries = train_countries
        self.val_countries = val_countries
        self.test_countries = test_countries
        self.load_boundaries = kwargs.pop("load_boundaries", False)
        self.temporal_options = temporal_options
        self.num_samples = num_samples

        # for the temporal option windowA, windowB and median we will have 4 channel input 
        if self.temporal_options in ("windowA", "windowB" , "median"):
            self.mean = torch.tensor([0, 0, 0, 0]) 
            self.std = torch.tensor([3000, 3000, 3000, 3000])
        elif self.temporal_options == "rgb": # for the rgb temporal option we are just selecting these 3 channls from both window_a and window_b images
            self.mean = torch.tensor([0, 0, 0, 0, 0, 0]) 
            self.std = torch.tensor([3000, 3000, 3000, 3000,  3000, 3000])

        print("Loaded datamodule with:")
        print(f"Train countries: {self.train_countries}")
        print(f"Val countries: {self.val_countries}")
        print(f"Test countries: {self.test_countries}")
        print(f"Number of samples: {self.num_samples}")

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            data_keys=["image", "mask"],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )
        super().__init__(FTW, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = FTW(
                countries=self.train_countries,
                split="train",
                load_boundaries=self.load_boundaries,
                temporal_options=self.temporal_options,
                num_samples=self.num_samples,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = FTW(
                countries=self.val_countries,
                split="val",
                load_boundaries=self.load_boundaries,
                temporal_options=self.temporal_options,
                num_samples=self.num_samples,
                **self.kwargs,
            )
        if stage == "test":
            self.test_dataset = FTW(
                countries=self.test_countries,
                split="test",
                load_boundaries=self.load_boundaries,
                temporal_options=self.temporal_options,
                num_samples=self.num_samples,
                **self.kwargs,
            )

    # NOTE: can get rid of this when https://github.com/microsoft/torchgeo/pull/2003 is merged and released
    # TODO: remove this method, test, then pin torchgeo>=0.6
    def plot(self, *args: Any, **kwargs: Any) -> Optional[Figure]:
        """Run the plot method of the validation dataset if one exists.

        Should only be called during 'fit' or 'validate' stages as ``val_dataset``
        may not exist during other stages.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A matplotlib Figure with the image, ground truth, and predictions.
        """
        fig: Optional[Figure] = None
        dataset = self.train_dataset or self.val_dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        if dataset is not None:
            if hasattr(dataset, "plot"):
                fig = dataset.plot(*args, **kwargs)
        return fig
