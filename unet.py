"""
unet.py — 从零手写 U-Net（无预训练权重）

网络结构：
  编码器（下采样）：4个 DoubleConv + MaxPool
  解码器（上采样）：4个 Up + DoubleConv + Skip Connection 拼接
  输出头：1x1 卷积映射到 num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两次 Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool 下采样 + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """双线性上采样 + Skip Connection 拼接 + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不整除的情况
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)   # Skip Connection 拼接
        return self.conv(x)


class UNet(nn.Module):
    """
    标准 U-Net
    输入：(B, 3, H, W)
    输出：(B, num_classes, H, W)

    通道配置：
      enc1: 3   -> 64
      enc2: 64  -> 128
      enc3: 128 -> 256
      enc4: 256 -> 512
      bottleneck: 512 -> 1024
      dec1: 1024+512 -> 512
      dec2: 512+256  -> 256
      dec3: 256+128  -> 128
      dec4: 128+64   -> 64
      out:  64 -> num_classes
    """
    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        # 编码器
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = Down(64, 128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)
        # 瓶颈层
        self.bottleneck = Down(512, 1024)
        # 解码器
        self.dec1 = Up(1024 + 512, 512)
        self.dec2 = Up(512 + 256, 256)
        self.dec3 = Up(256 + 128, 128)
        self.dec4 = Up(128 + 64, 64)
        # 输出头
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # 权重初始化（Kaiming）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码
        s1 = self.enc1(x)           # (B, 64,   H,   W)
        s2 = self.enc2(s1)          # (B, 128,  H/2, W/2)
        s3 = self.enc3(s2)          # (B, 256,  H/4, W/4)
        s4 = self.enc4(s3)          # (B, 512,  H/8, W/8)
        # 瓶颈
        b  = self.bottleneck(s4)    # (B, 1024, H/16,W/16)
        # 解码 + Skip Connection
        d1 = self.dec1(b,  s4)     # (B, 512,  H/8, W/8)
        d2 = self.dec2(d1, s3)     # (B, 256,  H/4, W/4)
        d3 = self.dec3(d2, s2)     # (B, 128,  H/2, W/2)
        d4 = self.dec4(d3, s1)     # (B, 64,   H,   W)
        return self.out_conv(d4)    # (B, num_classes, H, W)


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=8)
    x = torch.randn(2, 3, 240, 320)
    out = model(x)
    print("输入:", x.shape)
    print("输出:", out.shape)
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total/1e6:.2f}M")
