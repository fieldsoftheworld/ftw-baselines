import torch.nn as nn
import torch
from segmentation_models_pytorch.losses import DiceLoss

class Dice(DiceLoss):
    def __init__(self, class_weights=None, use_log_cosh=True, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )
        self.use_log_cosh = use_log_cosh

    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device)
            loss = loss * weights

        loss = loss.mean()

        if self.use_log_cosh:
            loss = torch.log(torch.cosh(loss))
        return loss


class DiceCE(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, 
                 ignore_index=None, class_weights=None, 
                 mode="multiclass", classes=None, use_log_cosh=True):
        super().__init__()
        # cross entropy
        ignore_value = -1000 if ignore_index is None else ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_value,
            weight=class_weights
        )
        # logcosh dice
        self.dice_loss = Dice(mode=mode, classes=classes, 
                              use_log_cosh=use_log_cosh, 
                              class_weights=class_weights,
                              ignore_index=ignore_index)

        # weighting
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        loss_ce = self.ce_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice