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
```bash
pip install torch torchvision wandb
```

### 训练

```bash
# CE Loss
python train.py --loss ce --exp_name unet_ce

# Dice Loss
python train.py --loss dice --exp_name unet_dice

# Combined Loss
python train.py --loss combined --exp_name unet_combined
```

### 参数说明
- `--loss`：损失函数类型 (`ce`, `dice`, `combined`)
- `--exp_name`：实验名称（用于 wandb 和 checkpoint 目录）
- `--epochs`：训练轮数（默认 50）
- `--batch_size`：批大小（默认 8）
- `--lr`：学习率（默认 1e-3）
- `--img_size`：输入尺寸 H W（默认 256 320）
- `--num_workers`：数据加载进程数（默认 4）

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