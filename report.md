# 实验报告：基于 U-Net 的语义分割与损失函数对比

---

## 一、数据集介绍

本实验使用 **Stanford Background Dataset**（ICCV 2009），这是一个经典的户外场景语义分割数据集。

- **图像总数**：715 张真实户外场景照片（JPEG 格式）
- **标注格式**：每张图像对应一个 `.regions.txt` 文件，记录每个像素的类别 ID（整数矩阵）
- **语义类别**：共 **8 类**，具体如下：

| 类别 ID | 类别名称 |
|--------|---------|
| 0 | Sky（天空） |
| 1 | Tree（树木） |
| 2 | Road（道路） |
| 3 | Grass（草地） |
| 4 | Water（水体） |
| 5 | Building（建筑） |
| 6 | Mountain（山脉） |
| 7 | Foreground Object（前景物体） |

- 标注中值为 `-1` 的像素表示无效区域，训练中映射为 `255` 并忽略。

---

## 二、模型结构介绍

本实验从零实现标准 **U-Net**（无预训练权重），模型结构如下图所示。

### 2.1 总体架构

U-Net 由 **编码器（Encoder）**、**瓶颈层（Bottleneck）** 和 **解码器（Decoder）** 三部分构成，编解码器之间通过 **Skip Connection（跳跃连接）** 拼接对应层的特征图，以保留细节空间信息。

- **输入**：`(B, 3, H, W)`，RGB 图像
- **输出**：`(B, 8, H, W)`，每个像素对应 8 类的 logits

### 2.2 基本模块

**DoubleConv**：两次 `Conv2d(3×3, padding=1) → BatchNorm2d → ReLU`，是所有层的基础单元，使用 `bias=False` 配合 BN。

**Down**：`MaxPool2d(2×2) → DoubleConv`，空间尺寸减半，通道数翻倍。

**Up**：`Bilinear Upsample(×2) → Concat(Skip) → DoubleConv`，空间尺寸翻倍，与编码器对应层拼接。

### 2.3 通道配置

| 模块 | 输入通道 | 输出通道 | 空间尺寸 |
|------|--------|--------|---------|
| enc1 (DoubleConv) | 3 | 64 | H × W |
| enc2 (Down) | 64 | 128 | H/2 × W/2 |
| enc3 (Down) | 128 | 256 | H/4 × W/4 |
| enc4 (Down) | 256 | 512 | H/8 × W/8 |
| bottleneck (Down) | 512 | 1024 | H/16 × W/16 |
| dec1 (Up) | 1024+512 | 512 | H/8 × W/8 |
| dec2 (Up) | 512+256 | 256 | H/4 × W/4 |
| dec3 (Up) | 256+128 | 128 | H/2 × W/2 |
| dec4 (Up) | 128+64 | 64 | H × W |
| out_conv (1×1 Conv) | 64 | 8 | H × W |

- **参数量**：约 31.04M
- **权重初始化**：卷积层使用 Kaiming Normal（`fan_out`, `relu`），BN 层 weight=1, bias=0。

---

## 三、实验设置

### 3.1 训练集 / 测试集划分

使用固定随机种子（`seed=42`）对 715 张图像进行打乱后，按 **8:2** 比例划分：

| 子集 | 数量 |
|------|------|
| 训练集 | 572 张 |
| 验证集 | 143 张 |

训练集开启随机水平翻转数据增强，验证集不做增强。

### 3.2 图像预处理

- **Resize**：统一缩放至 `256 × 320`（H × W）
- **标准化**：ImageNet 均值/方差，`mean=[0.485, 0.456, 0.406]`，`std=[0.229, 0.224, 0.225]`
- **标签插值**：最近邻插值（Nearest Neighbor），保证类别 ID 不失真

### 3.3 网络结构

标准 U-Net，见第二节，从零随机初始化，无预训练。

### 3.4 超参数设置

| 超参数 | 值 |
|-------|-----|
| Batch Size | 8 |
| 初始 Learning Rate | 1e-3 |
| 优化器 | AdamW（weight_decay=1e-4） |
| 学习率调度 | Cosine Annealing（T_max=50, eta_min=1e-6） |
| Epoch | 50 |
| Iteration（每 epoch） | 71（572 / 8，drop_last=True） |
| 总 Iteration | 3550 |
| 输入尺寸 | 256 × 320 |
| 类别数 | 8 |
| 忽略标签 | 255 |

### 3.5 损失函数

本实验对比三种损失函数配置：

**（1）Cross-Entropy Loss（CE）**

$$\mathcal{L}_{CE} = -\sum_{i} y_i \log \hat{p}_i$$

标准多类别交叉熵，忽略 `ignore_index=255` 的像素。直接优化每个像素的分类概率。

**（2）Dice Loss**

$$\mathcal{L}_{Dice} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot |\hat{P}_c \cap Y_c| + \epsilon}{|\hat{P}_c| + |Y_c| + \epsilon}$$

对每个类别分别计算 Dice 系数后取均值，`smooth=1.0`，忽略 255 像素。直接优化区域重叠度，对类别不平衡更鲁棒。

**（3）Combined Loss（CE + Dice）**

$$\mathcal{L}_{Combined} = 0.5 \cdot \mathcal{L}_{CE} + 0.5 \cdot \mathcal{L}_{Dice}$$

等权重组合，同时兼顾像素级分类精度与区域重叠度。

### 3.6 评价指标

使用 **mean Intersection over Union（mIoU）** 作为主要评价指标：

$$\text{mIoU} = \frac{1}{C'} \sum_{c \in C'} \frac{TP_c}{TP_c + FP_c + FN_c}$$

其中 $C'$ 为当前 batch 中实际出现的类别（跳过分母为 0 的类别），忽略标签为 255 的像素。

---

## 四、实验结果

### 4.1 定量结果

三组实验（各独立运行 50 epoch）的最终指标如下：

| 损失函数 | Best val mIoU | Final train mIoU | Final val Loss |
|---------|--------------|-----------------|---------------|
| CE | 0.5878 | 0.6692 | 0.4765 |
| Dice | 0.5768 | 0.6269 | 0.3323 |
| **Combined（CE+Dice）** | **0.6083** | **0.6886** | 0.4178 |

### 4.2 训练曲线（wandb）

> **以下各图请从 wandb 项目页面截图后放入此处：**
> **项目链接**：https://wandb.ai/astridbunny-fudan-university-school-of-management/hw2_task3_unet

**【图 1】train/loss 曲线对比（三种 loss，横轴 epoch 1–50）**

*（此处插入截图）*

---

**【图 2】val/loss 曲线对比**

*（此处插入截图）*

---

**【图 3】train/mIoU 曲线对比**

*（此处插入截图）*

---

**【图 4】val/mIoU 曲线对比**

*（此处插入截图）*

---

**【图 5】学习率（lr）衰减曲线**

*（此处插入截图）*

### 4.3 结果分析

**Combined Loss 效果最佳**，val mIoU 达到 0.6083，比纯 CE（0.5878）提升约 **2.1 个百分点**，比纯 Dice（0.5768）提升约 **3.2 个百分点**。

**Dice Loss 单独使用效果最差**，尽管 val loss 数值最低（0.3323），但 val mIoU 反而不如 CE。这是因为 Dice Loss 的数值尺度与 CE 不同，单独使用时缺乏像素级精确监督，导致边界处分割质量下降。

**CE Loss** 表现中规中矩，训练稳定，mIoU 居中。

**结论**：CE 与 Dice 的组合损失能够互补两者的优势——CE 提供稳定的逐像素监督，Dice 提高对类别不平衡的鲁棒性——因此在语义分割任务上，Combined Loss 是更优的选择。

---

## 五、实现细节与可复现性

- **框架**：PyTorch，CUDA 加速
- **代码文件**：`unet.py`（模型）、`dataset.py`（数据加载）、`losses.py`（损失函数）、`train.py`（训练脚本）
- **实验管理**：Weights & Biases（wandb）记录全程指标
- **随机种子**：数据集划分使用 `numpy.random.seed(42)`，保证可复现
- **运行命令**：
  ```bash
  python train.py --loss ce       --exp_name unet_ce
  python train.py --loss dice     --exp_name unet_dice
  python train.py --loss combined --exp_name unet_combined
  ```
