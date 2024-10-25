"""Trainer for semantic segmentation."""

import warnings
from typing import Any, Optional, Union

import lightning
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
from matplotlib.figure import Figure
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.datasets import unbind_samples
from torchgeo.models import FCN
from torchgeo.trainers.base import BaseTask
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchvision.models._api import WeightsEnum


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
        lr: float = 1e-3,
        patience: int = 10,
        patch_weights : bool = False,
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

        self.weights = weights
        super().__init__(ignore="weights")
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
        if loss == "ce":
            if self.hparams["class_weights"] is not None:
                class_weights = torch.tensor(self.hparams["class_weights"])
            else:
                class_weights = None
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=class_weights
            )
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="micro"
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

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
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

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
        y = batch["mask"]
        y_hat = self(x)
        loss: Tensor = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics)
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
        y = batch["mask"]
        y_hat = self(x)
        loss: Tensor = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics)

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
        y = batch["mask"]
        y_hat = self(x)
        loss: Tensor = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        y_hat: Tensor = self(x).softmax(dim=1)
        return y_hat

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"], amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams["patience"], eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }

    def on_train_epoch_start(self) -> None:
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar("lr", lr, self.current_epoch)

    def transfer_weights(self, model, backbone):
        base_model = None
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=True)
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=True)
        elif backbone == 'resnext50_32x4d':
            base_model = models.resnext50_32x4d(pretrained=True)

        if not base_model :
            print('Pretrained weights for ',backbone, ' not found. Unable to patch wieights')
            return
        prefix = 'encoder.'
        pretrained_weights = base_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict={}
        weights_ = 0
        update_weights = True

        for index, layer_key in enumerate(pretrained_weights):
            # TODO : generalizing the patch mapping
            encoder_key = prefix+layer_key
            layer_w = pretrained_weights[layer_key]
            if encoder_key in model_dict:
                if index == 0: # pacth first conv. layer weights
                    # Extract pre-trained weights for the first convolutional layer
                    pretrained_conv1_weights = layer_w
                    # Retrieve the current conv1 weights
                    new_conv1_weights = model_dict[encoder_key]
                    new_conv1_weights[:, :3, :, :] = pretrained_conv1_weights[:, :3, :, :]
                    new_conv1_weights[:, 4:7, :, :] = pretrained_conv1_weights[:, :3, :, :]
                    print(encoder_key,' First layer: ', model_dict[encoder_key].size(), '=>', new_conv1_weights.size())
                    pretrained_dict[encoder_key] = new_conv1_weights
                else:
                    if model_dict[encoder_key].size() !=  layer_w.size():
                        print('Invalid size match for ', encoder_key)
                        update_weights = False
                        break
                    pretrained_dict[encoder_key] = layer_w
                weights_+=1
        if update_weights:
            print('Updated weights_ count ', weights_)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print('Due to mismatch in the Tensor size, unable to patch weights.')        
