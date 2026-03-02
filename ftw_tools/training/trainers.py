"""Trainer for semantic segmentation."""

import logging
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
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.models._api import WeightsEnum

from ..inference.models import FCSiamAvg
from .losses import (
    CombinedLoss,
    FtnmtLoss,
    JaccardLoss,
    LocallyWeightedTverskyFocalLoss,
    PixelWeightedCE,
    TverskyFocalCELoss,
    logCoshDice,
    logCoshDiceCE,
)
from .metrics import get_object_level_metrics
from .utils import batch_corner_consensus_from_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        edge_agreement_loss: bool = False,
        pretrained_checkpoint: Optional[str] = None,
        model_kwargs: Optional[dict[Any, Any]] = None,
    ) -> None:
        """Initialize a new SemanticSegmentationTask instance.

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
            edge_agreement_loss: If True, ignore non-edge pixels by remapping them to
                the reserved "unknown" class index before loss computation.
            pretrained_checkpoint: Path to a checkpoint file from which to load
                encoder and decoder weights. This is used for transfer learning from
                edge pre-training. If provided, weights are loaded after model
                initialization, overriding ImageNet or random weights.
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
        if ignore_index is not None and loss == "jaccard":
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )
        self.class_names = ["background", "field", "boundary", "unknown"]
        self.weights = weights
        self.edge_agreement_loss = edge_agreement_loss
        super().__init__()

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
            ignore_index = self.hparams.get("ignore_index", None)

            base_jaccard = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
            if ignore_index is not None:
                self.criterion = JaccardLoss(base_jaccard, ignore_index)
            else:
                self.criterion = base_jaccard

        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        elif loss == "tversky":
            self.criterion = smp.losses.TverskyLoss(
                "multiclass", ignore_index=ignore_index
            )
        elif loss == "dice":
            self.criterion = smp.losses.DiceLoss(
                "multiclass", ignore_index=ignore_index
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

        elif loss == "logcoshdice":
            self.criterion = logCoshDice(
                mode="multiclass",
                classes=self.hparams["num_classes"],
                class_weights=class_weights,
                ignore_index=ignore_index,
            )
        elif loss == "logcoshdice+ce":
            self.criterion = logCoshDiceCE(
                weight_ce=0.5,
                weight_dice=0.5,
                mode="multiclass",
                classes=self.hparams["num_classes"],
                class_weights=class_weights,
                ignore_index=ignore_index,
            )

        elif loss == "ftnmt":
            self.criterion = FtnmtLoss(
                num_classes=self.hparams["num_classes"],
                loss_depth=self.hparams.get("loss_depth", 5),
                smooth=1e-5,
                ignore_index=ignore_index,
                class_weights=class_weights,
            )

        elif loss == "ce+ftnmt":
            # ce_weight = self.ce_weight
            # ftnmt_weight = self.ftnmt_weight
            ce_weight = getattr(self, "ce_weight", 0.5)
            ftnmt_weight = getattr(self, "ftnmt_weight", 0.5)
            ignore_value = -1000 if ignore_index is None else ignore_index

            ce_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=class_weights
            )
            ftnmt_loss = FtnmtLoss(
                num_classes=self.hparams["num_classes"],
                loss_depth=self.hparams.get("loss_depth", 5),
                smooth=1e-5,
                ignore_index=ignore_index,
                class_weights=class_weights,
            )

            self.criterion = CombinedLoss(ce_loss, ftnmt_loss, ce_weight, ftnmt_weight)

        elif loss == "localtversky":
            self.criterion = LocallyWeightedTverskyFocalLoss(ignore_index=ignore_index)

        elif loss == "tversky_ce":
            self.criterion = TverskyFocalCELoss(
                loss_weight=class_weights,
                tversky_weight=self.hparams.get("tversky_weight", 0.5),
                tversky_smooth=1,
                tversky_alpha=0.7,
                tversky_gamma=1.33,
                ignore_index=ignore_index,
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently supported: 'ce', 'pixel_weighted_ce', 'jaccard', "
                "'focal', 'dice', 'ce+dice', 'logcoshdice', 'ftnmt', "
                "'ce+ftnmt', 'localtversky', or 'tversky_ce'."
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
        # consensus accumulators
        self.val_consensus_sum = 0.0
        self.val_consensus_count = 0

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
        model_kwargs: dict[Any, Any] = self.hparams["model_kwargs"] or {}
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
            raise ValueError(f"Model type '{model}' is not valid.")

        if in_channels < 5 and model in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            raise ValueError("FCSiam models require more than one input image.")

        # Load encoder and decoder weights from a pre-trained checkpoint (e.g., edge pre-training)
        pretrained_checkpoint = self.hparams.get("pretrained_checkpoint")
        if pretrained_checkpoint is not None:
            logger.info("Loading from checkpoint: %s", pretrained_checkpoint)
            self._load_pretrained_weights(pretrained_checkpoint)

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

    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        """Load encoder and decoder weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.ckpt).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Extract encoder weights (keys start with "model.encoder.")
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("model.encoder."):
                new_key = key.replace("model.encoder.", "")
                encoder_state[new_key] = value

        # Extract decoder weights (keys start with "model.decoder.")
        decoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("model.decoder."):
                new_key = key.replace("model.decoder.", "")
                decoder_state[new_key] = value

        if not encoder_state and not decoder_state:
            raise ValueError(
                f"No encoder or decoder weights found in checkpoint {checkpoint_path}. "
                "Expected keys starting with 'model.encoder.' or 'model.decoder.'"
            )

        # Load encoder weights
        if encoder_state:
            result = self.model.encoder.load_state_dict(encoder_state, strict=False)
            if result is None:
                logger.info("Loaded encoder weights from %s.", checkpoint_path)
            else:
                missing, unexpected = result
                logger.info(
                    "Loaded encoder weights from %s. Missing: %d, Unexpected: %d",
                    checkpoint_path,
                    len(missing),
                    len(unexpected),
                )

        # Load decoder weights
        if decoder_state:
            result = self.model.decoder.load_state_dict(decoder_state, strict=False)
            if result is None:
                logger.info("Loaded decoder weights from %s.", checkpoint_path)
            else:
                missing, unexpected = result
                logger.info(
                    "Loaded decoder weights from %s. Missing: %d, Unexpected: %d",
                    checkpoint_path,
                    len(missing),
                    len(unexpected),
                )

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

        if self.edge_agreement_loss:
            y = y.clone()
            edges = batch["edge"].squeeze(1)
            edge_mask = edges > 0
            y = y.masked_fill(~edge_mask, 3)

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

        if len(x.shape) == 4:
            # corner consensus metric accumulation (re-run model on corner crops to capture context truncation)
            fcsiam_mode = self.hparams["model"] in [
                "fcsiamdiff",
                "fcsiamconc",
                "fcsiamavg",
            ]
            if fcsiam_mode:
                x_proc = rearrange(x.detach(), "b (t c) h w -> b t c h w", t=2)
            else:
                x_proc = x.detach()
            consensus_scores = batch_corner_consensus_from_model(
                self.model, x_proc, size=128, padding=64, fcsiam_mode=fcsiam_mode
            )
            for cs in consensus_scores:
                if cs is not None:
                    self.val_consensus_sum += cs
                    self.val_consensus_count += 1

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
            and len(x.shape) == 4
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
        # log consensus if any samples valid
        if self.val_consensus_count > 0:
            avg_consensus = self.val_consensus_sum / self.val_consensus_count
            self.log(
                "val/corner_consensus",
                avg_consensus,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.val_consensus_sum = 0.0
        self.val_consensus_count = 0

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


class EdgePretrainingTask(BaseTask):
    """Pre-training task for edge prediction.

    This task trains a segmentation model to predict binary edge masks from
    satellite imagery. The pre-trained encoder can then be used as initialization
    for the main segmentation task.
    """

    def __init__(
        self,
        model: str = "unet",
        backbone: str = "efficientnet-b3",
        weights: Optional[Union[WeightsEnum, str, bool]] = True,
        in_channels: int = 8,
        lr: float = 1e-3,
        patience: int = 100,
        model_kwargs: Optional[dict[Any, Any]] = None,
    ) -> None:
        """Initialize EdgePretrainingTask.

        Args:
            model: Name of the segmentation model (e.g., "unet").
            backbone: Name of the encoder backbone (e.g., "efficientnet-b3").
            weights: Initial encoder weights. True for ImageNet, False/None for random.
            in_channels: Number of input channels.
            lr: Learning rate.
            patience: Patience for cosine annealing scheduler.
            model_kwargs: Additional keyword arguments for the model.
        """
        self.weights = weights
        super().__init__()

    def configure_losses(self) -> None:
        """Initialize the loss criterion (BCE + Dice for edge prediction)."""
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics for binary edge prediction."""
        base_metrics = {
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "iou": BinaryJaccardIndex(),
        }
        self.train_metrics = MetricCollection(base_metrics, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def configure_models(self) -> None:
        """Initialize the segmentation model with 1 output class for binary edges."""
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        model_kwargs: dict[Any, Any] = self.hparams["model_kwargs"] or {}

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=1,
                **model_kwargs,
            )
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=1,
                **model_kwargs,
            )
        else:
            raise ValueError(
                f"Model '{model}' not supported for edge pretraining. Use 'unet' or 'deeplabv3+'."
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(x)

    def _log_edge_visualization(
        self, x: Tensor, edge: Tensor, y_hat: Tensor, batch_idx: int
    ) -> None:
        """Log visualization of edge predictions to TensorBoard.

        Args:
            x: Input image tensor (B, C, H, W).
            edge: Ground truth edge mask (B, H, W).
            y_hat: Raw model output logits (B, H, W).
            batch_idx: Batch index for labeling.
        """
        # Take first sample from batch
        img = x[0].cpu().numpy()
        gt_edge = edge[0].cpu().numpy()
        pred_prob = torch.sigmoid(y_hat[0]).cpu().numpy()
        pred_binary = (pred_prob > 0.5).astype(np.float32)

        # Check if we have stacked temporal windows (8 channels = 2x4 bands)
        num_channels = img.shape[0]
        has_two_windows = num_channels >= 8

        if has_two_windows:
            # Two temporal windows stacked
            img1 = img[:3].transpose(1, 2, 0)
            img2 = img[4:7].transpose(1, 2, 0)
            num_panels = 5
        else:
            # Single window
            img1 = img[:3].transpose(1, 2, 0)
            img2 = None
            num_panels = 4

        fig, axes = plt.subplots(1, num_panels, figsize=(num_panels * 4, 4))

        # Use np.clip like FTW.plot() for consistency
        axes[0].imshow(np.clip(img1, 0, 1))
        axes[0].set_title("Window B")
        axes[0].axis("off")

        panel_idx = 1
        if has_two_windows:
            axes[panel_idx].imshow(np.clip(img2, 0, 1))
            axes[panel_idx].set_title("Window A")
            axes[panel_idx].axis("off")
            panel_idx += 1

        axes[panel_idx].imshow(gt_edge, cmap="gray", vmin=0, vmax=1)
        axes[panel_idx].set_title("Ground Truth Edge")
        axes[panel_idx].axis("off")

        axes[panel_idx + 1].imshow(pred_prob, cmap="gray", vmin=0, vmax=1)
        axes[panel_idx + 1].set_title("Predicted Probability")
        axes[panel_idx + 1].axis("off")

        axes[panel_idx + 2].imshow(pred_binary, cmap="gray", vmin=0, vmax=1)
        axes[panel_idx + 2].set_title("Predicted Edge (>0.5)")
        axes[panel_idx + 2].axis("off")

        plt.tight_layout()

        # Log to all available loggers
        for log in self.loggers:
            if hasattr(log, "experiment") and hasattr(log.experiment, "add_figure"):
                log.experiment.add_figure(
                    f"edge_val/{batch_idx}", fig, global_step=self.global_step
                )

        plt.close(fig)

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute training loss for edge prediction.

        Args:
            batch: Dictionary with "image" and "edge" tensors.
            batch_idx: Index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        # Binarize edges: edge > 0 means edge pixel
        edge = (batch["edge"].squeeze(1) > 0).float()

        y_hat = self(x).squeeze(1)  # Shape: (B, H, W)

        bce = self.bce_loss(y_hat, edge)
        dice = self.dice_loss(y_hat.unsqueeze(1), edge.unsqueeze(1))
        loss = bce + dice

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bce", bce, on_step=False, on_epoch=True)
        self.log("train/dice", dice, on_step=False, on_epoch=True)

        preds = (torch.sigmoid(y_hat) > 0.5).long()
        self.train_metrics.update(preds, edge.long())

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute validation loss for edge prediction."""
        x = batch["image"]
        edge = (batch["edge"].squeeze(1) > 0).float()

        y_hat = self(x).squeeze(1)

        bce = self.bce_loss(y_hat, edge)
        dice = self.dice_loss(y_hat.unsqueeze(1), edge.unsqueeze(1))
        loss = bce + dice

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bce", bce, on_step=False, on_epoch=True)
        self.log("val/dice", dice, on_step=False, on_epoch=True)

        preds = (torch.sigmoid(y_hat) > 0.5).long()
        self.val_metrics.update(preds, edge.long())

        # Validation visualization for first few batches
        if (
            batch_idx < 10
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            self._log_edge_visualization(x, edge, y_hat, batch_idx)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute test loss for edge prediction."""
        x = batch["image"]
        edge = (batch["edge"].squeeze(1) > 0).float()

        y_hat = self(x).squeeze(1)

        bce = self.bce_loss(y_hat, edge)
        dice = self.dice_loss(y_hat.unsqueeze(1), edge.unsqueeze(1))
        loss = bce + dice

        self.log("test/loss", loss)

        preds = (torch.sigmoid(y_hat) > 0.5).long()
        self.test_metrics.update(preds, edge.long())

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize optimizer and learning rate scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"], amsgrad=True)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams["patience"], eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }

    def on_train_epoch_end(self) -> None:
        """Log per-epoch training metrics."""
        computed = self.train_metrics.compute()
        self.log_dict(computed, on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log per-epoch validation metrics."""
        computed = self.val_metrics.compute()
        self.log_dict(computed, on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log per-epoch test metrics."""
        computed = self.test_metrics.compute()
        self.log_dict(computed, on_step=False, on_epoch=True)
        self.test_metrics.reset()
