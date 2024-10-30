"""FTW dataset."""

import os
import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import NonGeoDataset, RasterDataset

from .utils import validate_checksums


class SingleRasterDataset(RasterDataset):
    """A torchgeo dataset that loads a single raster file."""

    def __init__(self, fn: str, transforms: Optional[Callable] = None):
        """Initialize the SingleRasterDataset class.

        Args:
            fn (str): The path to the raster file.
            transforms (Optional[Callable], optional): The transforms to apply to the
                raster file. Defaults to None.
        """
        path = os.path.abspath(fn)
        self.filename_regex = os.path.basename(path)
        super().__init__(paths=os.path.dirname(path), transforms=transforms)


class FTW(NonGeoDataset):

    valid_countries = [
        "austria",  
        "belgium",  
        "brazil",  
        "cambodia",  
        "corsica",  
        "croatia",  
        "denmark",  
        "estonia",  
        "finland",  
        "france",  
        "germany",  
        "india",  
        "kenya",  
        "latvia",  
        "lithuania",  
        "luxembourg",  
        "netherlands",  
        "portugal",  
        "rwanda",  
        "slovakia",  
        "slovenia",  
        "south_africa",  
        "spain",  
        "sweden",  
        "vietnam"
    ]
    
    valid_splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = "data/ftw",
        countries: Union[Sequence[str], str] = None,
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
        load_boundaries: bool = False,
        temporal_options: str = "stacked",
        num_samples: int = -1,
    ) -> None:
        """Initialize a new FTW dataset instance.

        Args:
            root: root directory where dataset can be found, this should contain the
                country folder
            countries: the countries to load the dataset from, e.g. "france"
            split: string specifying what split to load (e.g. "train", "val", "test")
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            load_boundaries: if True, load the 3 class masks with boundaries
            temporal_options : for abalation study, valid option are (stacked, windowA, windowB, median, rgb)
        Raises:
            AssertionError: if ``countries`` argument is invalid
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.root = root

        if countries is None:
            raise ValueError("Please specify the countries to load the dataset from")
        
        if isinstance(countries, str):
            countries = [countries]
        countries = [country.lower() for country in countries]
        for country in countries:
            assert country in self.valid_countries, f"Invalid country {country}"
        
        self.countries = countries
        assert split in self.valid_splits
        self.transforms = transforms
        self.checksum = checksum
        self.load_boundaries = load_boundaries
        self.temporal_options = temporal_options
        self.num_samples = num_samples

        if self.load_boundaries:
            print("Loading 3 Class Masks, with Boundaries")
        else:
            print("Loading 2 Class Masks, without Boundaries")

        print("Temporal option: ", temporal_options)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found at root directory or corrupted. "
                + "You can use download=True to download it"
            )

        if checksum:
            assert self._checksum(), "Checksum of dataset does not match"

        self.filenames = []
        all_filenames = []

        for country in self.countries:
            country_root: str = os.path.join(self.root, country)
            chips_fn = os.path.join(country_root, f"chips_{country}.parquet")
            chips_df = gpd.read_parquet(str(chips_fn))
            chips_df = chips_df[chips_df["split"] == split]
            aoi_ids = chips_df["aoi_id"].values

            for idx in aoi_ids:
                window_b_fn = Path(os.path.join( country_root, "s2_images/window_b", f"{idx}.tif"))
                window_a_fn =  Path(os.path.join( country_root, "s2_images/window_a", f"{idx}.tif"))
                masks_2c_fn =  Path(os.path.join( country_root, "label_masks/semantic_2class", f"{idx}.tif"))
                masks_3c_fn =  Path(os.path.join( country_root, "label_masks/semantic_3class", f"{idx}.tif"))

                # Skip the image AOI's which does not have all four corresponding files 
                if not (window_b_fn.exists() and window_a_fn.exists() and masks_2c_fn.exists() and masks_3c_fn.exists()):
                    continue

                if self.load_boundaries:
                    mask_fn = os.path.join(country_root, "label_masks/semantic_3class", f"{idx}.tif")
                else:
                    mask_fn = os.path.join(country_root, "label_masks/semantic_2class", f"{idx}.tif")

                if os.path.exists(mask_fn):
                    all_filenames.append(
                        {
                            "window_b": os.path.join(
                                country_root, "s2_images/window_b", f"{idx}.tif"
                            ),
                            "window_a": os.path.join(
                                country_root, "s2_images/window_a", f"{idx}.tif"
                            ),
                            "mask": mask_fn,
                        }
                    )

        if self.num_samples == -1: # select all samples
            self.filenames = all_filenames
        else:
            self.filenames = random.sample(all_filenames, min(self.num_samples, len(all_filenames)))

        print("Selecting : ",len(self.filenames), " samples")


    def _checksum(self) -> bool:
        """Check the checksum of the dataset.

        Returns:
            True if the checksum matches, else False
        """
        for country in self.valid_countries:
            print(f"Validating checksums for {country}")
            for checksum_file in ["distances_checksums.md5", "masks_checksums.md5", "window_b_checksums.md5", "window_a_checksums.md5"]:
                checksum_file = os.path.join(self.root, country, checksum_file)
                if not os.path.exists(checksum_file):
                    print(f"Checksum file {checksum_file} not found")
                    return False
                if not validate_checksums(checksum_file, self.root):
                    return False
        return True

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """



        for country in self.countries:
            if country not in self.valid_countries:
                print(f"Invalid country {country}")
                return False
            
            country_dir = os.path.join(self.root, country)
            if not os.path.exists(country_dir):
                print(f"Country directory {country_dir} not found")
                return False

            chips_fns = list(Path(country_dir).glob(f"chips_*.parquet"))
            # boundaries_fns = list(Path(country_dir).glob(f"boundaries_*.parquet"))
            if len(chips_fns) != 1:
                print(f"Country {country} does not have chips file")
                return False

            if self.load_boundaries:
                if not all(
                    [
                        os.path.exists(os.path.join(country_dir, "s2_images/window_b")),
                        os.path.exists(os.path.join(country_dir, "s2_images/window_a")),
                        os.path.exists(os.path.join(country_dir, "label_masks/semantic_3class")),
                    ]
                ):
                    print(f"Country {country} does not have all required directories")
                    return False
            else:
                if not all(
                    [
                        os.path.exists(os.path.join(country_dir, "s2_images/window_b")),
                        os.path.exists(os.path.join(country_dir, "s2_images/window_a")),
                        os.path.exists(os.path.join(country_dir, "label_masks/semantic_2class")),
                    ]
                ):
                    print(f"Country {country} does not have all required directories")
                    return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        raise NotImplementedError("Download functionality not implemented yet")

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            dictionary containing "image" and "mask" PyTorch tensors
        """
        filenames = self.filenames[index]

        images = []

        if self.temporal_options in ("stacked", "median", "windowB", "rgb"):
            with rasterio.open(filenames["window_b"]) as f:
                window_b_img = f.read()
                if self.temporal_options ==  "rgb": # select 3 channels only
                    window_b_img = window_b_img[:3]
                images.append(window_b_img)

        if self.temporal_options in ("stacked", "median", "windowA", "rgb"):
            with rasterio.open(filenames["window_a"]) as f:
                window_a_img = f.read()
                if self.temporal_options ==  "rgb": # select 3 channels only
                    window_a_img = window_a_img[:3]
                images.append(window_a_img)

        if self.temporal_options == "median":
            images = np.array(images).astype(np.int32)
            image = np.median(images, axis=0).astype(np.int32)
        else:
            image = np.concatenate(images, axis=0).astype(np.int32)
        
        image = torch.from_numpy(image).float()

        with rasterio.open(filenames["mask"]) as f:
            mask = f.read(1)
        mask = torch.from_numpy(mask).long()

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(self, sample: dict[str, Tensor], suptitle: Optional[str] = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        img1 = sample["image"][0:3].numpy().transpose(1, 2, 0)

        if self.temporal_options in ("stacked", "rgb"): # only two option where we will have more than 4 channels in input image
            img2 = sample["image"][3:6]  if self.temporal_options == "rgb" else sample["image"][4:7]
            img2 = img2.numpy().transpose(1, 2, 0)

        mask = sample["mask"].numpy()
        num_panels = 3 if self.temporal_options in ("stacked", "rgb") else 2
        if "prediction" in sample:
            num_panels +=1
            predictions = sample["prediction"].numpy()

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 5, 8))
        axs = axs.flatten()
        axs[0].imshow(np.clip(img1, 0, 1))
        axs[0].axis("off")

        panel_id = 1
        if self.temporal_options in ("stacked", "rgb"):
            axs[panel_id].imshow(np.clip(img2, 0, 1))
            axs[panel_id].axis("off")
            axs[panel_id+1].imshow(mask, vmin=0, vmax=2, cmap="gray")
            axs[panel_id+1].axis("off")
            panel_id+=2
        else:
            axs[panel_id].imshow(mask, vmin=0, vmax=2, cmap="gray")
            axs[panel_id].axis("off")
            panel_id +=1

        if "prediction" in sample:
            axs[panel_id].imshow(predictions)
            axs[panel_id].axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
