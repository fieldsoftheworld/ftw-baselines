import os
from typing import Literal

import geopandas as gpd
import rasterio
import segmentation_models_pytorch as smp
import shapely.geometry
import shapely.ops
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch import Tensor
from torchgeo.models import FCN, FCSiamConc, FCSiamDiff

# torchvision.ops.nms is not supported on MPS yet
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def load_model_from_checkpoint(path: str) -> tuple[nn.Module, str]:
    """Load a model from a checkpoint file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    model_type = hparams["model"]
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}

    if "criterion.weight" in state_dict:
        del state_dict["criterion.weight"]
    if "ce_loss.weight" in state_dict:
        del state_dict["ce_loss.weight"]

    if model_type == "unet":
        model = smp.Unet(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"],
            classes=hparams["num_classes"],
        )
    elif model_type == "unet_r":
        model = smp.Unet(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            decoder_channels=(16, 32, 64, 128, 256),
            in_channels=hparams["in_channels"],
            classes=hparams["num_classes"],
        )
    elif model_type == "upernet":
        model = smp.UPerNet(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"],
            classes=hparams["num_classes"],
        )
    elif model_type == "fcn":
        model = FCN(
            in_channels=hparams["in_channels"],
            classes=hparams["num_classes"],
            num_filters=hparams["num_filters"],
        )
    elif model_type == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"],
            classes=hparams["num_classes"],
        )
    elif model_type == "fcsiamdiff":
        model = FCSiamDiff(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"] // 2,
            classes=hparams["num_classes"],
        )
    elif model_type == "fcsiamconc":
        model = FCSiamConc(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"] // 2,
            classes=hparams["num_classes"],
        )
    elif model_type == "fcsiamavg":
        model = FCSiamAvg(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"] // 2,
            classes=hparams["num_classes"],
        )
    model.load_state_dict(state_dict, strict=True)
    return model, model_type


class FCSiamAvg(FCSiamDiff):
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input images of shape (b, t, c, h, w)

        Returns:
            predicted change masks of size (b, classes, h, w)
        """
        x1, x2 = x[:, 0], x[:, 1]
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [(features2[i] + features1[i]) / 2 for i in range(1, len(features1))]
        features.insert(0, features2[0])
        decoder_output = self.decoder(features)
        masks: Tensor = self.segmentation_head(decoder_output)
        return masks


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
        percentile_low: float | None = 0.01,
        percentile_high: float | None = 0.99,
        norm_constant: float | None = None,
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
            percentile_low: Lower percentile for per-channel normalization (default: 0.01).
                Mutually exclusive with norm_constant.
            percentile_high: Upper percentile for per-channel normalization (default: 0.99).
                Mutually exclusive with norm_constant.
            norm_constant: If provided, divide by this value and clip to [0, 1] instead of
                using percentile normalization. Mutually exclusive with percentile_low/percentile_high.
            device: Device to run the model on, either "cuda" or "cpu".

        Raises:
            ValueError: If both norm_constant and percentile options are specified.
        """
        super().__init__()

        # Validate mutually exclusive normalization options
        if norm_constant is not None and (percentile_low is not None or percentile_high is not None):
            raise ValueError(
                "norm_constant is mutually exclusive with percentile_low/percentile_high. "
                "Set percentile_low=None and percentile_high=None when using norm_constant."
            )

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
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.norm_constant = norm_constant
        self.device = device
        from ultralytics import YOLO

        self.model = YOLO(self.checkpoints[model]).to(device)
        self.model.eval()
        self.model.fuse()
        self.transforms = nn.Sequential(
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ConvertImageDtype(torch.float32),
        ).to(device)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization based on configured method.

        Uses either constant normalization (if norm_constant is set) or
        per-channel percentile normalization (if percentile_low/high are set).

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Normalized tensor clipped to [0, 1].
        """
        if self.norm_constant is not None:
            # Simple constant normalization: divide and clip
            return (x / self.norm_constant).clip(0.0, 1.0)

        # Per-channel percentile normalization
        # At this point percentile_low and percentile_high are guaranteed to be non-None
        assert self.percentile_low is not None and self.percentile_high is not None

        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        # Flatten spatial dimensions for percentile computation
        x_flat = x.view(B, C, -1)  # (B, C, H*W)

        # Compute percentiles per channel per batch
        p_low = torch.quantile(x_flat.float(), self.percentile_low, dim=2, keepdim=True)  # (B, C, 1)
        p_high = torch.quantile(x_flat.float(), self.percentile_high, dim=2, keepdim=True)  # (B, C, 1)

        # Reshape for broadcasting
        p_low = p_low.view(B, C, 1, 1)
        p_high = p_high.view(B, C, 1, 1)

        # Avoid division by zero
        denom = p_high - p_low
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        # Normalize and clip
        x_norm = (x - p_low) / denom
        return x_norm.clip(0.0, 1.0)

    @staticmethod
    def polygonize(
        result: "ultralytics.engine.results.Results",
        transform: rasterio.Affine,
        crs=rasterio.CRS,
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
            lambda x: (
                shapely.geometry.Polygon(zip(x["x"], x["y"]))
                if len(x["x"]) >= 3
                else None
            )
        )
        df.dropna(subset=["geometry"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["geometry"] = df["geometry"].apply(
            lambda geom: shapely.ops.transform(pixel_to_geo, geom)
        )
        df.drop(["name", "class", "box", "segments"], axis=1, inplace=True)
        return gpd.GeoDataFrame(df, geometry=df["geometry"], crs=crs)

    def __call__(
        self, image: torch.Tensor
    ) -> list["ultralytics.engine.results.Results"]:
        """Forward pass through the model.

        Args:
            image: The input image tensor, expected to be in the format (B, C, H, W) or (C, H, W).

        Returns:
            A list of results containing the model predictions.
        """
        # Add batch dimension if needed
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Select first 3 channels (RGB)
        image = image[:, :3, ...]

        # Apply normalization (constant or percentile-based)
        image = self._normalize(image.to(self.device))

        # Apply remaining transforms (resize, dtype conversion)
        image = self.transforms(image)

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
