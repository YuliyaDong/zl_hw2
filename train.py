"""
train.py — U-Net 训练主脚本

支持三种 loss 配置：
  - ce       : CrossEntropyLoss
  - dice     : DiceLoss
  - combined : CE + Dice

wandb 可视化，mIoU 评估。

用法示例：
  python train.py --loss ce       --exp_name unet_ce
  python train.py --loss dice     --exp_name unet_dice
  python train.py --loss combined --exp_name unet_combined
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from unet import UNet
from dataset import get_dataloaders
from losses import get_loss

# ─────────────────────────────────────────────
NUM_CLASSES   = 8
IGNORE_INDEX  = 255
DATA_DIR      = '/mnt/tidal-alsh01/usr/dongyujie/homework/zl/hw2_task3/iccv09Data'
WANDB_API_KEY = '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'
# ─────────────────────────────────────────────


def compute_miou(preds, targets, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """
    preds:   (B, H, W) long — 预测类别
    targets: (B, H, W) long — 真实标签（255 忽略）
    返回: mean IoU (float)
    """
    iou_list = []
    valid = (targets != ignore_index)
    preds_v   = preds[valid]
    targets_v = targets[valid]

    for cls in range(num_classes):
        tp = ((preds_v == cls) & (targets_v == cls)).sum().item()
        fp = ((preds_v == cls) & (targets_v != cls)).sum().item()
        fn = ((preds_v != cls) & (targets_v == cls)).sum().item()
        denom = tp + fp + fn
        if denom == 0:
            continue   # 该类在本 batch 中不存在，跳过
        iou_list.append(tp / denom)

    if len(iou_list) == 0:
        return 0.0
    return float(np.mean(iou_list))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    n_batches  = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)                        # (B, C, H, W)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)                # (B, H, W)
        miou  = compute_miou(preds.cpu(), labels.cpu())

        total_loss += loss.item()
        total_miou += miou
        n_batches  += 1

    return total_loss / n_batches, total_miou / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    n_batches  = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        miou   = compute_miou(preds.cpu(), labels.cpu())

        total_loss += loss.item()
        total_miou += miou
        n_batches  += 1

    return total_loss / n_batches, total_miou / n_batches


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ── wandb 初始化 ──
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    wandb.login()
    wandb.init(
        project = 'hw2_task3_unet',
        name    = args.exp_name,
        config  = {
            'loss_type'  : args.loss,
            'epochs'     : args.epochs,
            'batch_size' : args.batch_size,
            'lr'         : args.lr,
            'img_size'   : args.img_size,
            'num_classes': NUM_CLASSES,
        },
    )

    # ── 数据 ──
    img_size = tuple(args.img_size)    # (H, W)
    train_loader, val_loader = get_dataloaders(
        data_dir    = DATA_DIR,
        batch_size  = args.batch_size,
        img_size    = img_size,
        num_workers = args.num_workers,
    )

    # ── 模型、损失、优化器 ──
    model     = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    criterion = get_loss(args.loss, num_classes=NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_miou = 0.0
    save_dir = os.path.join('checkpoints', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_miou   = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_mIoU={train_miou:.4f}  "
              f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  lr={lr_now:.2e}")

        # ── wandb 记录 ──
        wandb.log({
            'train/loss' : train_loss,
            'train/mIoU' : train_miou,
            'val/loss'   : val_loss,
            'val/mIoU'   : val_miou,
            'lr'         : lr_now,
        }, step=epoch)

        # ── 保存最优模型 ──
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            print(f"  => 保存最优模型（val_mIoU={best_val_miou:.4f}）")

    # ── 保存最后一个 epoch ──
    torch.save(model.state_dict(), os.path.join(save_dir, 'last.pth'))
    print(f"\n训练完成！最优 val_mIoU = {best_val_miou:.4f}")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-Net 训练脚本')
    parser.add_argument('--loss',        type=str,   default='ce',
                        choices=['ce', 'dice', 'combined'],
                        help='损失函数类型')
    parser.add_argument('--exp_name',    type=str,   default='unet_ce',
                        help='实验名称（用于 wandb 和 checkpoint 目录）')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--img_size',    type=int,   nargs=2, default=[256, 320],
                        help='输入图像尺寸 H W')
    parser.add_argument('--num_workers', type=int,   default=4)
    args = parser.parse_args()
    main(args)
