from typing import Literal

import geopandas as gpd
import rasterio
import shapely.geometry
import shapely.ops
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

try:
    import ultralytics
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "ultralytics is not installed. Please install it with 'pip install ultralytics'."
    )

try:
    import huggingface_hub
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "huggingface_hub is not installed. Please install it with 'pip install huggingface_hub'."
    )


class DelineateAnything:
    """DelineateAnything model for delineating fields in satellite imagery."""

    repo = "MykolaL/DelineateAnything"
    revision = "c218e08349039afadc9bed3ff46f5cc7f69d9aa9"
    transforms = nn.Sequential(
        T.Lambda(lambda x: x[:, :3, ...]),
        T.ToDtype(torch.float),
        T.Lambda(lambda x: x / 3000.0),
        T.Lambda(lambda x: x.clip(0, 1)),
        T.ConvertImageDtype(torch.float32),
    )

    def __init__(
        self,
        model: Literal[
            "DelineateAnything-S", "DelineateAnything"
        ] = "DelineateAnything-S",
        image_size: tuple[int, int] | int = 320,
        max_detections: int = 50,
        iou_threshold: float = 0.6,
        conf_threshold: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize the DelineateAnything model.

        Args:
            model (str): The model variant to use, either "DelineateAnything-S" or "DelineateAnything".
            image_size (tuple[int, int] | int): The size of the input images. If an int is provided, it will be used for both width and height.
            max_detections (int): Maximum number of detections per image.
            iou_threshold (float): Intersection over Union threshold for filtering predictions.
            conf_threshold (float): Confidence threshold for filtering predictions.
            device (str): Device to run the model on, either "cuda" or "cpu".
        """
        super().__init__()
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        self.max_detections = max_detections
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.device = device

        self.checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id=self.repo, revision=self.revision, filename=f"{model}.pt"
        )
        self.model = ultralytics.YOLO(self.checkpoint_path).to(device)
        self.model.eval()
        self.transforms.eval()
        self.transforms = self.transforms.to(device)

    def polygonize(
        self,
        result: ultralytics.engine.results.Results,
        transform: rasterio.Affine,
        crs=rasterio.CRS,
    ) -> gpd.GeoDataFrame:
        """Convert the model predictions to a GeoDataFrame of georeferenced polygons.

        Args:
            result (ultralytics.engine.results.Results): The results from the model prediction.
            transform (rasterio.Affine): The affine transformation to convert pixel coordinates to geographic coordinates.
            crs (rasterio.CRS): The coordinate reference system of the output GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the polygons of the delineated fields.
        """

        def pixel_to_geo(x, y, z=None):
            return transform * (x, y)

        df = result.to_df()
        df["geometry"] = df["segments"].apply(
            lambda x: shapely.geometry.Polygon(zip(x["y"], x["x"]))
        )
        df["geometry"] = df["geometry"].apply(
            lambda geom: shapely.ops.transform(pixel_to_geo, geom)
        )
        df.drop(["name", "class", "box", "segments"], axis=1, inplace=True)
        return gpd.GeoDataFrame(df, geometry=df["geometry"], crs=crs)

    def __call__(self, image: torch.Tensor) -> list[ultralytics.engine.results.Results]:
        """Forward pass through the model.

        Args:
            image (torch.Tensor): The input image tensor, expected to be in the format [C, H, W] and normalized.

        Returns:
            list[ultralytics.engine.results.Results]: A list of results containing the model predictions.
        """
        image = self.transforms(image.to(self.device))
        results = self.model.predict(
            image,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            max_det=self.max_detections,
            iou=self.iou_threshold,
            device=self.device,
            half=True,
            verbose=False,
        )
        return results
