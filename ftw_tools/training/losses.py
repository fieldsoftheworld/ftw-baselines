from typing import Optional, Tuple, Union, Dict

import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss as Dice


class PixelWeightedCE(nn.Module):
    """
    Pixel-weighted Cross-Entropy.
    Weights come from blurring a binary map of the target class with a Gaussian.

    Args
    ----
    kernel_size : int or (int, int)
        Gaussian kernel size, e.g. 5 or (5,5).
    sigma : float or (float, float)
        Gaussian sigma in pixels, e.g. 3.0 or (3.0, 3.0).
    scale : float
        Weight multiplier for the target class neighborhood (default 1.0).
    class_weights : Optional[torch.Tensor]
        Per-class weights for cross-entropy (shape [C]).
    target_class : int
        Class index whose neighborhood should be upweighted.
    ignore_index : Optional[int]
        Label to ignore in loss/weight computation (e.g. 255).
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        sigma: Union[float, Tuple[float, float]] = 3,
        scale: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        target_class: int = 2,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        # normalize args to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma), float(sigma))

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.scale = scale
        # Register class_weights as a buffer so it moves with .to(device)
        # If provided, store as float tensor via register_buffer; otherwise None
        if class_weights is not None:
            cw = class_weights.clone().detach().float()
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None
        self.target_class = int(target_class)
        self.ignore_index = ignore_index

        # Build the blur op (created once; runs on whatever device/tensor you pass)
        self.blur = K.filters.GaussianBlur2d(
            kernel_size=self.kernel_size, sigma=self.sigma, border_type="reflect"
        )

    @torch.no_grad()
    def _make_weights(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Build per-pixel weights:
          1) E = 1 where masks == target_class else 0 (ignores are 0)
          2) weights = blur(E)
        Returns weights shaped (N, H, W) on masks.device.
        """
        assert masks.ndim == 3, "masks should be (N, H, W) with class indices"
        dtype = torch.float32

        # Binary map for the chosen class
        E = masks == self.target_class
        if self.ignore_index is not None:
            E = E & (masks != self.ignore_index)

        E = E.to(dtype=dtype).unsqueeze(1)  # (N,1,H,W)
        # Kornia modules track device via input; no manual .to needed
        weights = self.blur(E).squeeze(1)  # (N,H,W)

        # If everything is ignored/absent, avoid all-zero weights
        if weights.sum() <= 0:
            weights = torch.zeros_like(weights)

        return (weights * self.scale) + 1.0  # add 1 to ensure min weight is 1.0

    def forward(self, preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        preds: (N, C, H, W) logits
        masks: (N, H, W)  int64 class indices
        """
        assert preds.ndim == 4 and masks.ndim == 3, "preds (N,C,H,W), masks (N,H,W)"
        assert preds.shape[0] == masks.shape[0] and preds.shape[2:] == masks.shape[1:]

        # Per-pixel CE, no reduction
        ce = F.cross_entropy(
            preds,
            masks,
            weight=self.class_weights,
            reduction="none",
            ignore_index=-100 if self.ignore_index is None else self.ignore_index,
        )  # (N,H,W), float

        # Build weights from blurred target-class mask
        with torch.no_grad():
            w = self._make_weights(masks)  # (N,H,W)

            # Zero out weights on ignore pixels to be safe
            if self.ignore_index is not None:
                w = torch.where(masks == self.ignore_index, torch.zeros_like(w), w)

        # Weighted average: sum(w * ce) / sum(w)
        num = (w * ce).sum()
        den = w.sum().clamp_min(1e-8)
        return num / den


class logCoshDice(Dice):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )
        
    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device)
            loss = (loss * weights)/weights.sum()

        loss = loss.mean()
        loss = torch.log(torch.cosh(loss))
        return loss


class logCoshDiceCE(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, 
                 ignore_index=None, class_weights=None, 
                 mode="multiclass", classes=None):
        super().__init__()
        # cross entropy
        ignore_value = -1000 if ignore_index is None else ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_value,
            weight=class_weights
        )
        # logcosh dice
        self.dice_loss = logCoshDice(mode=mode, classes=classes, 
                              class_weights=class_weights,
                              ignore_index=ignore_index)

        # weighting
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        loss_ce = self.ce_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice

class FtnmtLoss(nn.Module):
    """
    Multi-class Fractal Tanimoto (with dual) loss
    """

    def __init__(self, num_classes, loss_depth=5, smooth=1e-5, ignore_index=None, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.loss_depth = loss_depth
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)

    def tnmt_base(self, preds, labels):
        tpl = torch.sum(preds * labels, dim=(0, 2, 3))
        tpp = torch.sum(preds * preds, dim=(0, 2, 3))
        tll = torch.sum(labels * labels, dim=(0, 2, 3))

        num = tpl + self.smooth
        denum = 0.0

        for d in range(self.loss_depth):
            a = 2. ** d
            b = -(2. * a - 1.)
            denum += torch.reciprocal(a * (tpp + tll) + b * tpl + self.smooth)

        result = (num * denum) / self.loss_depth
        return result

    def forward(self, preds, targets):
        """
        preds: [B, C, H, W] (raw logits)
        targets: [B, H, W] (class indices)
        """
        preds = torch.softmax(preds, dim=1)  # Convert logits to probabilities
        total_loss = 0.0

        for c in range(self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            # One-hot target for class c
            labels_c = (targets == c).float().unsqueeze(1)  # [B, 1, H, W]
            preds_c = preds[:, c:c+1, :, :]

            l1 = self.tnmt_base(preds_c, labels_c)
            l2 = self.tnmt_base(1. - preds_c, 1. - labels_c)
            score = 0.5 * (l1 + l2)

            weight = self.class_weights[c] if self.class_weights is not None else 1.0
            total_loss += weight * (1.0 - score)

        return total_loss / self.num_classes


# Helper for combining losses
class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, weight1=0.5, weight2=0.5, ignore_index=None):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        loss1_val = self.loss1(inputs, targets)

        # Mask out ignored pixels
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            # For 2D/3D inputs, flatten and select only valid pixels
            if inputs.dim() > 2:
                # inputs: [B, C, H, W], targets: [B, H, W], mask: [B, H, W]
                inputs = inputs.permute(0, 2, 3, 1)  # [B, H, W, C]
                inputs = inputs[mask]               # [N, C]
                inputs = inputs.view(-1, inputs.shape[-1])  # [N, C]
                targets = targets[mask]             # [N]
            else:
                inputs = inputs[mask]
                targets = targets[mask]

        loss2_val = self.loss2(inputs, targets)
        return self.weight1 * loss1_val + self.weight2 * loss2_val

def _as_long_index(t: torch.Tensor) -> torch.Tensor:
    """Return a LongTensor suitable for one_hot/indexing."""
    if t.dtype != torch.long:
        return t.long()
    return t

# Binary Tversky Focal Loss
class BinaryTverskyFocalLoss(nn.Module):
    '''
    Pytorch version of Tversky focal loss proposed in
    "A novel focal Tversky loss function and improved Attention U-Net for
    lesion segmentation" (https://arxiv.org/abs/1810.07842)
    '''

    def __init__(self, smooth=1, alpha=0.7, gamma=1.33):
        super(BinaryTverskyFocalLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], \
            "predict & target batch size do not match"

        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)

        num = (predict * target).sum() + self.smooth
        den = (predict * target).sum() + \
            self.alpha * ((1 - predict) * target).sum() + \
            self.beta * (predict * (1 - target)).sum() + self.smooth
        loss = torch.pow(1 - num/den, 1 / self.gamma)

        return loss


# Multiclass Tversky Focal Loss
class TverskyFocalLoss(nn.Module):
    '''
    Tversky focal loss
    '''
    def __init__(self, weight=None, ignore_index=-100, **kwargs):
        super(TverskyFocalLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        device = predict.device
        nclass = predict.shape[1]
        if predict.shape == target.shape:
            target_oh = target
            valid_mask = (target_oh.sum(dim=1, keepdim=True) > 0).squeeze(1)
        elif len(predict.shape) == 4:
            valid_mask = (target != self.ignore_index)
            if isinstance(valid_mask, bool):
                valid_mask = torch.full_like(target, valid_mask, dtype=torch.bool, device=device)

            safe_target = _as_long_index(target.masked_fill(~valid_mask, 0))
            target_oh = (F.one_hot(safe_target, num_classes=nclass)
                         .permute(0, 3, 1, 2)
                         .contiguous())
        else:
            raise ValueError("Incompatible shapes between 'predict' and 'target'.")

        valid_float = valid_mask.unsqueeze(1).float()
        tversky = BinaryTverskyFocalLoss(**self.kwargs)
        total_loss = 0

        if self.weight is None:
            weight = torch.full((nclass,), 1.0 / nclass, dtype=torch.float32, device=device)
        else:
            if isinstance(self.weight, list):
                weight = torch.tensor(self.weight, dtype=torch.float32, device=device)
            elif isinstance(self.weight, torch.Tensor):
                weight = self.weight.to(device=device, dtype=torch.float32)
            else:
                weight = torch.tensor([float(self.weight)] * nclass, dtype=torch.float32, device=device)

        predict = F.softmax(predict, dim=1)

        for i in range(nclass):
            tversky_loss = tversky(predict[:, i] * valid_float, target_oh[:, i] * valid_float)
            assert weight.shape[0] == nclass, \
                f"Expect weight shape [{nclass}], get[{weight.shape[0]}]"
            tversky_loss *= weight[i]
            total_loss += tversky_loss

        return total_loss


# Locally Weighted Tversky Focal Loss
class LocallyWeightedTverskyFocalLoss(nn.Module):
    r"""
    Tversky focal loss weighted by inverse of label frequency calculated
    locally based on the input batch.
    """
    def __init__(self, ignore_index=-100, **kwargs):
        super(LocallyWeightedTverskyFocalLoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index

    def calculate_weights(self, target, num_class):
        if not isinstance(target, torch.Tensor):
            try:
                target = torch.as_tensor(target)
            except Exception:
                return torch.ones(num_class, device="cpu") * 1e-5

        if target.dim() == 4 and target.shape[1] == num_class:
            target_labels = target.argmax(dim=1)
        else:
            target_labels = target

        device = getattr(target_labels, "device", torch.device("cpu"))
        if getattr(self, "ignore_index", None) is None:
            valid = torch.ones_like(target_labels, dtype=torch.bool, device=device)
        else:
            try:
                valid = (target_labels != self.ignore_index)
            except Exception:
                valid = torch.tensor(bool(target_labels != self.ignore_index), dtype=torch.bool, device=device)

        if isinstance(valid, bool):
            valid = torch.tensor(valid, dtype=torch.bool, device=device)

        if valid.numel() == 0 or valid.sum() == 0:
            return torch.ones(num_class, device=device) * 1e-5

        unique, unique_counts = torch.unique(target_labels[valid], return_counts=True)
        if unique.numel() == 0:
            return torch.ones(num_class, device=device) * 1e-5

        in_range_mask = (unique >= 0) & (unique < num_class)
        if getattr(self, "ignore_index", None) is not None:
            in_range_mask &= (unique != self.ignore_index)

        if not in_range_mask.any():
            return torch.ones(num_class, device=device) * 1e-5

        unique = unique[in_range_mask]
        unique_counts = unique_counts[in_range_mask].to(device).float()

        denom = valid.sum().float().to(device)
        ratio = unique_counts / denom.clamp_min(1e-6)
        weight = (1.0 / ratio)
        weight = weight / weight.sum()

        loss_weight = torch.ones(num_class, device=device) * 1e-5
        for i, idx in enumerate(unique):
            loss_weight[int(idx.item())] = weight[i].to(device)

        return loss_weight

    def forward(self, predict, target):
        num_class = predict.shape[1]
        loss_weight = self.calculate_weights(target, num_class)
        if isinstance(loss_weight, torch.Tensor):
            loss_weight = loss_weight.to(device=predict.device, dtype=torch.float32)
        else:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=predict.device)

        loss_fn = TverskyFocalLoss(weight=loss_weight,
                                   ignore_index=self.ignore_index,
                                   **self.kwargs)
        return loss_fn(predict, target)


# Tversky Focal + Cross Entropy Combined Loss
class TverskyFocalCELoss(nn.Module):
    """
    Combination of Tversky focal loss and cross entropy loss.
    """
    def __init__(self, loss_weight=None, tversky_weight=0.5, tversky_smooth=1,
                 tversky_alpha=0.7, tversky_gamma=1.33, ignore_index=-100):
        super(TverskyFocalCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.tversky_weight = tversky_weight
        self.tversky_smooth = tversky_smooth
        self.tversky_alpha = tversky_alpha
        self.tversky_gamma = tversky_gamma
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size do not match"

        tversky = TverskyFocalLoss(
            weight=self.loss_weight, ignore_index=self.ignore_index,
            smooth=self.tversky_smooth, alpha=self.tversky_alpha, gamma=self.tversky_gamma
        )

        ce_weight = None
        if self.loss_weight is not None:
            if isinstance(self.loss_weight, list):
                ce_weight = torch.tensor(self.loss_weight, dtype=torch.float32, device=predict.device)
            elif isinstance(self.loss_weight, torch.Tensor):
                ce_weight = self.loss_weight.to(device=predict.device, dtype=torch.float32)

        ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=self.ignore_index)
        loss = self.tversky_weight * tversky(predict, target) + (1 - self.tversky_weight) * ce(predict, target)

        return loss


# Dice, LogCosh Dice, Jaccard, and Focal Wrappers, handles ignore index (for presence only countries case) which is not handled in smp

class DiceLoss(nn.Module):
    def __init__(self, base_loss, ignore_index):
        super().__init__()
        self.base_loss = base_loss
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        preds = preds * mask.unsqueeze(1)
        targets = targets * mask
        return self.base_loss(preds, targets)


class JaccardLoss(nn.Module):
    def __init__(self, base_loss, ignore_index):
        super().__init__()
        self.base_loss = base_loss
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        preds = preds * mask.unsqueeze(1)
        targets = targets * mask
        return self.base_loss(preds, targets)
