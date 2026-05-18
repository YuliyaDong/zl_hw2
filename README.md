# U-Net Semantic Segmentation with Loss Function Comparison

基于 U-Net 的户外场景语义分割，对比 Cross-Entropy、Dice Loss 和组合损失函数的性能。

## 项目概述

本项目使用 Stanford Background Dataset（ICCV 2009）进行语义分割，数据集包含 715 张户外场景图像和 8 类语义标签。

**核心贡献**：从零实现标准 U-Net 模型（无预训练权重），对比三种损失函数配置：
- **CE Loss**：标准交叉熵损失
- **Dice Loss**：区域重叠度指标
- **Combined Loss**：CE + Dice 等权重组合

**最佳结果**：Combined Loss 取得最优验证集 mIoU **0.6083**，相比纯 CE 提升 2.1pp，相比纯 Dice 提升 3.2pp。

## 数据集

- **数据集**：Stanford Background Dataset（ICCV 2009）
- **图像总数**：715 张真实户外场景照片
- **语义类别**：8 类（Sky、Tree、Road、Grass、Water、Building、Mountain、Foreground Object）
- **标注格式**：`.regions.txt`（像素级整数标签矩阵）
- **划分比例**：8:2（训练集 572 张，验证集 143 张）

## 模型结构

标准 U-Net 架构：
- **编码器**：4 层下采样 + MaxPool（64 → 128 → 256 → 512 通道）
- **瓶颈层**：1024 通道
- **解码器**：4 层上采样 + Skip Connection（512 → 256 → 128 → 64 通道）
- **输出头**：1×1 卷积映射到 8 类
- **参数量**：约 31.04M
- **权重初始化**：Kaiming Normal（卷积层），BN 层 weight=1, bias=0

## 实验设置

### 超参数
- Batch Size: 8
- Learning Rate: 1e-3
- Optimizer: AdamW (weight_decay=1e-4)
- Scheduler: Cosine Annealing (T_max=50, eta_min=1e-6)
- Epochs: 50
- Input Size: 256×320

### 损失函数

**CE Loss**：标准多类别交叉熵
$$\mathcal{L}_{\text{CE}} = -\sum_{i} y_i \log \hat{p}_i$$

**Dice Loss**：多类别 Dice 系数
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot |P_c \cap Y_c| + \epsilon}{|P_c| + |Y_c| + \epsilon}$$

**Combined Loss**：等权重组合
$$\mathcal{L}_{\text{Combined}} = 0.5 \cdot \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

### 评价指标

Mean Intersection over Union (mIoU)：
$$\text{mIoU} = \frac{1}{C'} \sum_{c \in C'} \frac{TP_c}{TP_c + FP_c + FN_c}$$

## 实验结果

| 损失函数 | Best val mIoU | Final train mIoU | Final val Loss |
|---------|--------------|-----------------|----------------|
| CE      | 0.5878       | 0.6692          | 0.4765         |
| Dice    | 0.5768       | 0.6269          | 0.3323         |
| **Combined** | **0.6083** | **0.6886**      | 0.4178         |

**关键发现**：
- Combined Loss 性能最优，验证了 CE 和 Dice 的互补性
- Dice Loss 单独使用效果最差，缺乏像素级精确监督
- CE Loss 表现稳定但不及组合方案

## 文件结构

```
.
├── train.py          # 训练主脚本
├── unet.py           # U-Net 模型实现
├── dataset.py        # 数据加载与预处理
├── losses.py         # 损失函数实现（CE、Dice、Combined）
├── README.md         # 本文件
├── checkpoints/      # 训练好的模型权重（best.pth）
└── logs/             # 训练日志
```

## 使用方法

### 环境配置

#### 1. 系统要求
- Python >= 3.8
- CUDA 11.8+ (推荐，也支持 CPU)
- 磁盘空间：至少 5GB（用于数据集和权重）

#### 2. 依赖安装

**方式一：使用 pip**
```bash
# 克隆仓库
git clone https://github.com/YuliyaDong/zl_hw2.git
cd zl_hw2

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install wandb pillow numpy
```

**方式二：使用 conda**
```bash
# 创建虚拟环境
conda create -n unet-seg python=3.10
conda activate unet-seg

# 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install wandb pillow
```

#### 3. 数据集配置

**下载 Stanford Background Dataset**
```bash
# 从以下位置下载数据集
# http://dags.stanford.edu/projects/scenedataset.html

# 解压并放在项目目录
unzip iccv09Data.zip
```

**数据集结构**
```
iccv09Data/
├── images/          # 715 张 JPEG 图像
│   ├── a000001.jpg
│   ├── a000002.jpg
│   └── ...
└── labels/          # 对应的标签文件
    ├── a000001.regions.txt
    ├── a000002.regions.txt
    └── ...
```

#### 4. Wandb 配置（可选，用于训练可视化）

```bash
# 登录 Wandb 账号
wandb login

# 输入你的 API key（从 https://wandb.ai/settings/keys 获取）
```

如不使用 wandb，需在 `train.py` 中注释相关代码（见下方训练部分）。

---

### 训练

#### 快速开始

```bash
# 使用 CE Loss（推荐新手）
python train.py --loss ce --exp_name unet_ce

# 使用 Dice Loss
python train.py --loss dice --exp_name unet_dice

# 使用 Combined Loss（推荐，性能最优）
python train.py --loss combined --exp_name unet_combined
```

#### 完整参数配置

```bash
python train.py \
    --loss combined \
    --exp_name unet_combined \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --img_size 256 320 \
    --num_workers 4
```

#### 参数详解

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--loss` | `ce` | 损失函数：`ce`（交叉熵）、`dice`（Dice）、`combined`（组合） |
| `--exp_name` | `unet_ce` | 实验名称，用于 wandb 和 checkpoint 目录命名 |
| `--epochs` | `50` | 训练轮数 |
| `--batch_size` | `8` | 批大小（根据 GPU 显存调整） |
| `--lr` | `1e-3` | 初始学习率 |
| `--img_size` | `256 320` | 输入图像尺寸（高 宽） |
| `--num_workers` | `4` | 数据加载进程数 |

#### 训练输出说明

训练过程会输出以下信息：
```
使用设备: cuda
Epoch   1/50  train_loss=0.5234  train_mIoU=0.3421  val_loss=0.4876  val_mIoU=0.3682  lr=1.00e-03
Epoch   2/50  train_loss=0.4982  train_mIoU=0.4123  val_loss=0.4562  val_mIoU=0.4215  lr=1.00e-03
  => 保存最优模型（val_mIoU=0.4215）
...
训练完成！最优 val_mIoU = 0.6083
```

**输出文件**：
- `checkpoints/<exp_name>/best.pth`：最优验证集性能的模型权重
- `checkpoints/<exp_name>/last.pth`：最后一个 epoch 的模型权重
- `logs/`：训练日志（如已配置）

---

### 测试与推理

#### 1. 加载预训练模型进行推理

```python
import torch
from unet import UNet
from dataset import StanfordBackgroundDataset
from torch.utils.data import DataLoader

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, num_classes=8).to(device)
model.load_state_dict(torch.load('checkpoints/unet_combined/best.pth'))
model.eval()

# 加载验证集
val_dataset = StanfordBackgroundDataset(
    data_dir='iccv09Data',
    split='val',
    img_size=(256, 320),
    augment=False
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 推理单个样本
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)  # (B, 8, H, W)
        preds = logits.argmax(dim=1)  # (B, H, W)
        print(f"预测结果 shape: {preds.shape}")
        break
```

#### 2. 在自定义图像上进行推理

```python
import torch
from PIL import Image
import torchvision.transforms as T
from unet import UNet
import numpy as np

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, num_classes=8).to(device)
model.load_state_dict(torch.load('checkpoints/unet_combined/best.pth'))
model.eval()

# 图像预处理
def preprocess_image(img_path, img_size=(256, 320)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size[1], img_size[0]), Image.BILINEAR)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

# 推理
img_tensor = preprocess_image('path/to/your/image.jpg')
img_tensor = img_tensor.to(device)

with torch.no_grad():
    logits = model(img_tensor)
    pred = logits.argmax(dim=1).cpu().numpy()[0]

# 可视化
class_names = ['Sky', 'Tree', 'Road', 'Grass', 'Water', 'Building', 'Mountain', 'Foreground']
print(f"预测类别分布：")
for class_id in np.unique(pred):
    count = (pred == class_id).sum()
    print(f"  {class_names[class_id]}: {count} 个像素")
```

#### 3. 评估模型性能

```python
import torch
from train import compute_miou
from unet import UNet
from dataset import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, num_classes=8).to(device)
model.load_state_dict(torch.load('checkpoints/unet_combined/best.pth'))
model.eval()

_, val_loader = get_dataloaders(
    data_dir='iccv09Data',
    batch_size=8,
    img_size=(256, 320),
    num_workers=4
)

# 验证集评估
model.eval()
total_miou = 0.0
num_batches = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        miou = compute_miou(preds.cpu(), labels.cpu())
        total_miou += miou
        num_batches += 1

avg_miou = total_miou / num_batches
print(f"验证集平均 mIoU: {avg_miou:.4f}")
```

---

### 常见问题

#### Q: 显存不足怎么办？
**A**: 减小 batch_size 或 img_size
```bash
python train.py --loss combined --batch_size 4 --img_size 224 256
```

#### Q: 数据集下载很慢？
**A**: 可从备份链接下载，或使用镜像源加速

#### Q: 如何禁用 wandb？
**A**: 在 `train.py` 第 117-130 行注释掉 wandb 相关代码：
```python
# os.environ['WANDB_API_KEY'] = WANDB_API_KEY
# wandb.login()
# wandb.init(...)
```

#### Q: 如何改变输入图像尺寸？
**A**: 修改 `--img_size` 参数，但需确保高度和宽度都是 32 的倍数（U-Net 下采样 5 层）

#### Q: 训练时间多长？
**A**: 在 RTX 4090 上，50 epoch 约需 30-40 分钟（batch_size=8）

## 模型权重

训练好的模型权重（best.pth）已保存在 Google Drive：
https://drive.google.com/file/d/15osUwQZFRYiBovdzoLG7Y37btuXlRO6P/view?usp=drive_link

## 实验可视化

使用 Weights & Biases 记录训练过程，包括：
- 训练/验证 loss 曲线
- 训练/验证 mIoU 曲线
- 学习率衰减曲线

项目链接：https://wandb.ai/astridbunny-fudan-university-school-of-management/hw2_task3_unet

## 许可证

MIT