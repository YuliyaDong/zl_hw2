"""
losses.py — 损失函数实现

包含：
  - CrossEntropyLoss（标准交叉熵，忽略 255）
  - DiceLoss（手动实现，多类别，忽略 255）
  - CombinedLoss（CE + Dice）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 8
IGNORE_INDEX = 255


class CrossEntropyLoss(nn.Module):
    """标准多类别交叉熵，忽略 ignore_index=255"""
    def __init__(self, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, H, W)  long，255为忽略
        """
        return self.ce(logits, targets)


class DiceLoss(nn.Module):
    """
    手动实现多类别 Dice Loss

    公式：Dice = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    对每个类别分别计算，取均值（忽略 ignore_index 的像素）
    """
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, smooth=1.0):
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.smooth        = smooth

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, H, W)  long
        """
        # 构造有效像素 mask（排除 ignore_index）
        valid_mask = (targets != self.ignore_index)          # (B, H, W) bool

        # softmax 得到概率
        probs = F.softmax(logits, dim=1)                     # (B, C, H, W)

        # 将 targets 中的 ignore 像素临时置 0（不影响结果，后面会 mask 掉）
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0

        # one-hot 编码
        B, C, H, W = probs.shape
        targets_oh = F.one_hot(targets_clean, num_classes=C)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()   # (B, C, H, W)

        # 应用 valid_mask：ignore 位置置 0
        mask = valid_mask.unsqueeze(1).float()                 # (B, 1, H, W)
        probs      = probs      * mask
        targets_oh = targets_oh * mask

        # 对每个类别计算 Dice
        # 在 (B, H, W) 维度上求和
        intersection = (probs * targets_oh).sum(dim=(0, 2, 3))  # (C,)
        cardinality  = (probs + targets_oh).sum(dim=(0, 2, 3))  # (C,)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """
    组合损失：CE Loss + Dice Loss
    loss = alpha * CE + (1 - alpha) * Dice
    """
    def __init__(self, alpha=0.5, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.alpha    = alpha
        self.ce_loss  = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, targets):
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.alpha * ce + (1 - self.alpha) * dice


def get_loss(loss_type='ce', num_classes=NUM_CLASSES):
    """
    loss_type: 'ce' | 'dice' | 'combined'
    """
    if loss_type == 'ce':
        return CrossEntropyLoss()
    elif loss_type == 'dice':
        return DiceLoss(num_classes=num_classes)
    elif loss_type == 'combined':
        return CombinedLoss(num_classes=num_classes)
    else:
        raise ValueError(f"未知 loss_type: {loss_type}")
