import os
from typing import Literal

import geopandas as gpd
import rasterio
import shapely.geometry
import shapely.ops
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import ultralytics
from ultralytics.engine.results import Results

# torchvision.ops.nms is not supported on MPS yet
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class DelineateAnything:
    """DelineateAnything model for delineating fields in satellite imagery."""

    checkpoints = {
        "DelineateAnything-S": "https://hf.co/torchgeo/delineate-anything-s/resolve/69cd440b0c5bd450ced145e68294aa9393ddae05/delineate_anything_s_rgb_yolo11n-b879d643.pt",
        "DelineateAnything": "https://hf.co/torchgeo/delineate-anything/resolve/60bea7b2f81568d16d5c75e4b5b06289e1d7efaf/delineate_anything_rgb_yolo11x-88ede029.pt",
    }

    def __init__(
        self,
        model: Literal[
            "DelineateAnything-S", "DelineateAnything"
        ] = "DelineateAnything-S",
        patch_size: tuple[int, int] | int = 256,
        resize_factor: int = 2,
        max_detections: int = 100,
        iou_threshold: float = 0.3,
        conf_threshold: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize the DelineateAnything model.

        Args:
            model: The model variant to use, either "DelineateAnything-S" or "DelineateAnything".
            patch_size: The size of the input images. If an int is provided, it will be used for both width and height.
            resize_factor: The factor to resize the input images by.
            max_detections: Maximum number of detections per image.
            iou_threshold: Intersection over Union threshold for filtering predictions.
            conf_threshold: Confidence threshold for filtering predictions.
            device: Device to run the model on, either "cuda" or "cpu".
        """
        super().__init__()
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.image_size = (
            self.patch_size[0] * resize_factor,
            self.patch_size[1] * resize_factor,
        )
        self.max_detections = max_detections
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = ultralytics.YOLO(self.checkpoints[model]).to(device)
        self.model.eval()
        self.model.fuse()
        self.transforms = nn.Sequential(
            T.Lambda(lambda x: x.unsqueeze(dim=0) if x.ndim == 3 else x),
            T.Lambda(lambda x: x[:, :3, ...]),
            T.Lambda(lambda x: x / 3000.0),
            T.Lambda(lambda x: x.clip(0.0, 1.0)),
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ConvertImageDtype(torch.float32),
        ).to(device)

    @staticmethod
    def polygonize(
        result: Results, transform: rasterio.Affine, crs=rasterio.CRS
    ) -> gpd.GeoDataFrame | None:
        """Convert the model predictions to a GeoDataFrame of georeferenced polygons.

        Args:
            result: The results from the model prediction.
            transform: The affine transformation to convert pixel coordinates to geographic coordinates.
            crs: The coordinate reference system of the output GeoDataFrame.

        Returns:
            A GeoDataFrame containing the polygons of the delineated fields.
        """

        def pixel_to_geo(x, y, z=None):
            return transform * (x, y)

        df = result.to_df()
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        if len(df) == 0:
            return None

        df["geometry"] = df["segments"].apply(
            lambda x: shapely.geometry.Polygon(zip(x["x"], x["y"]))
            if len(x["x"]) >= 3
            else None
        )
        df.dropna(subset=["geometry"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["geometry"] = df["geometry"].apply(
            lambda geom: shapely.ops.transform(pixel_to_geo, geom)
        )
        df.drop(["name", "class", "box", "segments"], axis=1, inplace=True)
        return gpd.GeoDataFrame(df, geometry=df["geometry"], crs=crs)

    def __call__(self, image: torch.Tensor) -> list[Results]:
        """Forward pass through the model.

        Args:
            image: The input image tensor, expected to be in the format (B, C, H, W).

        Returns:
            A list of results containing the model predictions.
        """
        image = self.transforms(image.to(self.device))
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            max_det=self.max_detections,
            iou=self.iou_threshold,
            device=self.device,
            half=False,
            verbose=False,
        )
        # Rescale masks and boxes to original patch size
        for result in results:
            if result.masks is not None:
                result.masks.orig_shape = self.patch_size
            if result.boxes is not None:
                result.boxes.orig_shape = self.patch_size
        return results
