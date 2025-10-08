"""Trainer for semantic segmentation."""

import warnings
from typing import Any, Optional, Union

import lightning
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from matplotlib.figure import Figure
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.datasets import unbind_samples
from torchgeo.models import FCN, FCSiamConc, FCSiamDiff
from torchgeo.trainers.base import BaseTask
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.models._api import WeightsEnum

from ..postprocess.metrics import get_object_level_metrics
from .losses import PixelWeightedCE
from .models import FCSiamAvg


class CustomSemanticSegmentationTask(BaseTask):
    """Semantic Segmentation.

    This is currently a copy of torchgeo.trainers.SemanticSegmentationTask, but with a
    fix to allow loading of saved weights.
    """

    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        num_filters: int = 3,
        loss: str = "ce",
        class_weights: Optional[list] = None,
        ignore_index: Optional[int] = None,
        pixel_weight_scale: Optional[float] = None,
        lr: float = 1e-3,
        patience: int = 10,
        patch_weights: bool = False,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        model_kwargs: dict[Any, Any] = dict(),
    ) -> None:
        """Inititalize a new SemanticSegmentationTask instance.

        Args:
            model: Name of the
                `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
            backbone: Name of the `timm
                <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or `smp
                <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone to use.
                Note: if using a DPT model, the backbone must be a supported timm encoder
                from the list `here <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__
                such as `tu-resnet50` or `tu-vit_base_patch16_224`.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False or
                None for random weights, or the path to a saved model state dict. FCN
                model does not support pretrained weights. Pretrained ViT weight enums
                are not supported yet.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes.
            num_filters: Number of filters. Only applicable when model='fcn'.
            loss: Name of the loss function, currently supports
                'ce', 'jaccard' or 'focal' loss.
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss.
            ignore_index: Optional integer class index to ignore in the loss and
                metrics.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.
            freeze_decoder: Freeze the decoder network to linear probe
                the segmentation head.
            model_kwargs: Additional keyword arguments to pass to the model

        Warns:
            UserWarning: When loss='jaccard' and ignore_index is specified.

        .. versionchanged:: 0.3
           *ignore_zeros* was renamed to *ignore_index*.

        .. versionchanged:: 0.4
           *segmentation_model*, *encoder_name*, and *encoder_weights*
           were renamed to *model*, *backbone*, and *weights*.

        .. versionadded: 0.5
            The *class_weights*, *freeze_backbone*, and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           The *weights* parameter now supports WeightEnums and checkpoint paths.
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.
        """
        print("Using custom trainer")
        if ignore_index is not None and loss == "jaccard":
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )
        self.class_names = ["background", "field", "boundary", "unknown"]
        self.weights = weights
        super().__init__()
        print(self.hparams)

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        class_weights = None
        if self.hparams["class_weights"] is not None:
            class_weights = torch.tensor(self.hparams["class_weights"])
        pixel_weight_scale = 1.0
        if self.hparams["pixel_weight_scale"] is not None:
            pixel_weight_scale = self.hparams["pixel_weight_scale"]

        if loss == "ce":
            if self.hparams["class_weights"] is not None:
                class_weights = torch.tensor(self.hparams["class_weights"])
            else:
                class_weights = None
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=class_weights
            )
        elif loss == "pixel_weighted_ce":
            self.criterion = PixelWeightedCE(
                kernel_size=7,
                sigma=3.0,
                target_class=2,
                scale=pixel_weight_scale,
                class_weights=class_weights,
                ignore_index=ignore_index,
            )
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        elif loss == "ce+dice":
            self.dice_loss = smp.losses.DiceLoss(
                "multiclass", ignore_index=ignore_index
            )

            if self.hparams["class_weights"] is not None:
                class_weights = torch.tensor(self.hparams["class_weights"])
            else:
                class_weights = None
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.ce_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=class_weights
            )
            self.criterion = lambda y_pred, y_true: self.ce_loss(
                y_pred, y_true
            ) + self.dice_loss(y_pred, y_true)

        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]

        base_metrics = {
            "precision": MulticlassPrecision(
                num_classes, average=None, ignore_index=ignore_index
            ),
            "recall": MulticlassRecall(
                num_classes, average=None, ignore_index=ignore_index
            ),
            "iou": MulticlassJaccardIndex(
                num_classes, average=None, ignore_index=ignore_index
            ),
        }
        self.train_metrics = MetricCollection(base_metrics, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.val_agg = MetricCollection(
            {
                "precision_macro": MulticlassPrecision(
                    num_classes, average="macro", ignore_index=ignore_index
                ),
                "recall_macro": MulticlassRecall(
                    num_classes, average="macro", ignore_index=ignore_index
                ),
                "iou_macro": MulticlassJaccardIndex(
                    num_classes, average="macro", ignore_index=ignore_index
                ),
            },
            prefix="val/",
        )

        self.val_tps = 0
        self.val_fps = 0
        self.val_fns = 0

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]
        num_filters: int = self.hparams["num_filters"]
        model_kwargs: dict[Any, Any] = self.hparams["model_kwargs"]
        patch_weights: bool = self.hparams["patch_weights"]

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
                **model_kwargs,
            )
        elif model == "unet_r":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                decoder_channels=(16, 32, 64, 128, 256),
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
                **model_kwargs,
            )
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        elif model == "upernet":
            self.model = smp.UPerNet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
                **model_kwargs,
            )
        elif model == "segformer":
            self.model = smp.Segformer(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "dpt":
            self.model = smp.DPT(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
                decoder_readout="ignore",
                **model_kwargs,
            )
        elif model == "fcsiamdiff":
            self.model = FCSiamDiff(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels // 2,
                classes=num_classes,
            )
        elif model == "fcsiamconc":
            self.model = FCSiamConc(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels // 2,
                classes=num_classes,
            )
        elif model == "fcsiamavg":
            self.model = FCSiamAvg(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels // 2,
                classes=num_classes,
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+', and 'fcn', 'upernet', 'segformer', and 'dpt'."
            )

        if in_channels < 5 and model in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            raise ValueError("FCSiam models require more than one input image.")

        # Freeze backbone
        if self.hparams["freeze_backbone"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if patch_weights:
            self.transfer_weights(self.model, backbone)

    def _log_per_class(self, metrics_dict, split: str):
        # metrics_dict like {"precision": tensor(C,), "recall": tensor(C,), "iou": tensor(C,)}
        for name, values in metrics_dict.items():
            # values is shape [C]
            for i, v in enumerate(values):
                cname = self.class_names[i]
                if cname == "field":
                    self.log(
                        f"{name}/{cname}",
                        v,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        y = batch["mask"].squeeze(1)

        if self.hparams["model"] in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            y_hat = self(rearrange(x, "b (t c) h w -> b t c h w", t=2))
        else:
            y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_metrics.update(y_hat, y)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"].squeeze(1)

        if self.hparams["model"] in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            y_hat = self(rearrange(x, "b (t c) h w -> b t c h w", t=2))
        else:
            y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y)

        for i in range(y_hat.shape[0]):
            output = y_hat[i].argmax(dim=0).cpu().numpy().astype(np.uint8)
            mask = y[i].cpu().numpy().astype(np.uint8)
            tps, fps, fns = get_object_level_metrics(mask, output, iou_threshold=0.5)
            self.val_tps += tps
            self.val_fps += fps
            self.val_fns += fns

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_metrics.update(y_hat, y)
        self.val_agg.update(y_hat, y)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            datamodule = self.trainer.datamodule
            batch["prediction"] = y_hat.argmax(dim=1)
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Optional[Figure] = None
            fig = datamodule.plot(sample)

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()

            if fig:
                for logger in self.loggers:
                    summary_writer = logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(
                            f"image/{batch_idx}", fig, global_step=self.global_step
                        )
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"].squeeze(1)

        if self.hparams["model"] in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            y_hat = self(rearrange(x, "b (t c) h w -> b t c h w", t=2))
        else:
            y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics.update(y_hat, y)

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"], amsgrad=True)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams["patience"], eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }

    def on_train_epoch_start(self) -> None:
        lr = self.optimizers().param_groups[0]["lr"]
        self.logger.experiment.add_scalar("lr", lr, self.current_epoch)

    def on_train_epoch_end(self):
        computed = self.train_metrics.compute()
        self._log_per_class(computed, "train")
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        object_precision = (
            self.val_tps / (self.val_tps + self.val_fps)
            if (self.val_tps + self.val_fps) > 0
            else 0
        )
        object_recall = (
            self.val_tps / (self.val_tps + self.val_fns)
            if (self.val_tps + self.val_fns) > 0
            else 0
        )
        object_f1 = (
            2 * object_precision * object_recall / (object_precision + object_recall)
            if (object_precision + object_recall) > 0
            else 0
        )
        self.log("val/object_precision", object_precision)
        self.log("val/object_recall", object_recall)
        self.log("val/object_f1", object_f1)

        self.val_tps = 0
        self.val_fps = 0
        self.val_fns = 0

        per_class = self.val_metrics.compute()
        self._log_per_class(per_class, "val")
        self.val_metrics.reset()

        # log aggregates (single scalars)
        agg = self.val_agg.compute()
        self.log_dict(agg, on_step=False, on_epoch=True, sync_dist=True)
        self.val_agg.reset()

    def on_test_epoch_end(self):
        per_class = self.test_metrics.compute()
        self._log_per_class(per_class, "test")
        self.test_metrics.reset()

    def transfer_weights(self, model, backbone):
        base_model = None
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=True)
        elif backbone == "resnet50":
            base_model = models.resnet50(pretrained=True)
        elif backbone == "resnext50_32x4d":
            base_model = models.resnext50_32x4d(pretrained=True)

        if not base_model:
            print(
                "Pretrained weights for ",
                backbone,
                " not found. Unable to patch wieights",
            )
            return
        prefix = "encoder."
        pretrained_weights = base_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        weights_ = 0
        update_weights = True

        for index, layer_key in enumerate(pretrained_weights):
            # TODO : generalizing the patch mapping
            encoder_key = prefix + layer_key
            layer_w = pretrained_weights[layer_key]
            if encoder_key in model_dict:
                if index == 0:  # pacth first conv. layer weights
                    # Extract pre-trained weights for the first convolutional layer
                    pretrained_conv1_weights = layer_w
                    # Retrieve the current conv1 weights
                    new_conv1_weights = model_dict[encoder_key]
                    new_conv1_weights[:, :3, :, :] = pretrained_conv1_weights[
                        :, :3, :, :
                    ]
                    new_conv1_weights[:, 4:7, :, :] = pretrained_conv1_weights[
                        :, :3, :, :
                    ]
                    print(
                        encoder_key,
                        " First layer: ",
                        model_dict[encoder_key].size(),
                        "=>",
                        new_conv1_weights.size(),
                    )
                    pretrained_dict[encoder_key] = new_conv1_weights
                else:
                    if model_dict[encoder_key].size() != layer_w.size():
                        print("Invalid size match for ", encoder_key)
                        update_weights = False
                        break
                    pretrained_dict[encoder_key] = layer_w
                weights_ += 1
        if update_weights:
            print("Updated weights_ count ", weights_)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("Due to mismatch in the Tensor size, unable to patch weights.")
