"""
dataset.py — Stanford Background Dataset 数据加载

类别（8类）：
  0: sky
  1: tree
  2: road
  3: grass
  4: water
  5: building
  6: mountain
  7: foreground object
  -1: 忽略（映射到255）
"""

import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torch


class StanfordBackgroundDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=(256, 320), augment=False):
        """
        data_dir: iccv09Data 目录路径
        split:    'train' or 'val'
        img_size: (H, W)
        augment:  是否数据增强（仅 train）
        """
        self.img_dir   = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.img_size  = img_size
        self.augment   = augment

        # 获取所有样本 id
        all_imgs = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        all_ids  = [os.path.splitext(os.path.basename(p))[0] for p in all_imgs]
        # 过滤掉没有 regions.txt 的
        all_ids = [i for i in all_ids
                   if os.path.exists(os.path.join(self.label_dir, f'{i}.regions.txt'))]

        # 固定 8:2 划分
        np.random.seed(42)
        idx = np.random.permutation(len(all_ids))
        split_n = int(len(all_ids) * 0.8)
        if split == 'train':
            self.ids = [all_ids[i] for i in idx[:split_n]]
        else:
            self.ids = [all_ids[i] for i in idx[split_n:]]

        # 图像预处理
        self.img_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        # 加载图像
        img = Image.open(os.path.join(self.img_dir, f'{sid}.jpg')).convert('RGB')
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)

        # 加载标注
        label = np.loadtxt(
            os.path.join(self.label_dir, f'{sid}.regions.txt')
        ).astype(np.int64)
        # resize label（最近邻）
        label_img = Image.fromarray(label.astype(np.int32), mode='I')
        label_img = label_img.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        label = np.array(label_img, dtype=np.int64)

        # -1 映射到 255（忽略标签）
        label[label == -1] = 255

        # 数据增强（仅训练集）
        if self.augment:
            # 随机水平翻转
            if np.random.rand() > 0.5:
                img   = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = np.fliplr(label).copy()

        img_tensor   = self.img_transform(img)
        label_tensor = torch.from_numpy(label).long()

        return img_tensor, label_tensor


def get_dataloaders(data_dir, batch_size=8, img_size=(256, 320), num_workers=4):
    train_ds = StanfordBackgroundDataset(data_dir, split='train',
                                          img_size=img_size, augment=True)
    val_ds   = StanfordBackgroundDataset(data_dir, split='val',
                                          img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"训练集: {len(train_ds)} 张，验证集: {len(val_ds)} 张")
    return train_loader, val_loader


if __name__ == "__main__":
    loader, val = get_dataloaders('/mnt/tidal-alsh01/usr/dongyujie/homework/zl/hw2_task3/iccv09Data')
    imgs, labels = next(iter(loader))
    print("图像 shape:", imgs.shape)
    print("标注 shape:", labels.shape)
    print("标注唯一值:", labels.unique())
