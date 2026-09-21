---
title: SwinIR
tags:
  - paper
  - image-restoration
  - 2021
---

## Preliminary

Swin Transformer 原本是用于图像分类、目标检测和语义分割等任务的通用 backbone。它在窗口内计算注意力，再通过移动窗口让不同窗口之间交换信息。同时使用 Patch Merging 逐步降低分辨率、增加通道数，得到不同尺度的特征，交给下游任务的头使用。

图像恢复需要保留像素级细节，因此 SwinIR 保留了窗口注意力和移位窗口的设计，但去掉了逐级 Patch Merging，使特征提取过程中空间分辨率保持不变。

## Overview

![Overview](swinir-overview.png)

SwinIR 主要的设计是使用 Swin Transformer 改进图像恢复中的特征提取步骤。先通过卷积提取浅层特征，再使用多个 RSTB（Residual Swin Transformer Block）提取深层特征，最后将两者相加，输入对应任务的重建头，得到恢复后的图像。

$$
I_{LQ}\xrightarrow{\mathrm{Conv}_{3\times3}}F_0
\xrightarrow{\mathrm{RSTB}\times K\,+\,\mathrm{Conv}_{3\times3}}F_{DF},
\qquad F=F_0+F_{DF}
$$

不同任务使用相同的特征提取架构，再根据是否需要上采样配置重建头。每个任务分别训练整套模型，主干和重建头一起端到端训练。

## Shallow Feature Extraction

使用一个 $3\times3$ 卷积，将输入图像映射到 $C$ 通道的特征空间：

$$
F_0=\operatorname{Conv}_{3\times3}(I_{LQ}),\qquad
F_0\in\mathbb R^{H\times W\times C}
$$

卷积提取局部特征，$F_0$ 同时通过长残差连接传到重建头，保留输入中的低频信息，使深层特征提取部分主要学习需要恢复的高频细节。

## Deep Feature Extraction

依次堆叠 $K$ 个 RSTB，最后用一个 $3\times3$ 卷积整合深层特征：

$$
F_i=\operatorname{RSTB}_i(F_{i-1}),\quad i=1,\ldots,K,
\qquad F_{DF}=\operatorname{Conv}_{3\times3}(F_K)
$$

### RSTB

每个 RSTB 先通过若干 Swin Transformer Layer（STL）提取特征，再经过一个 $3\times3$ 卷积，最后加上输入残差：

$$
X_0=F_{i-1},\qquad X_j=\operatorname{STL}_{i,j}(X_{j-1}),\quad j=1,\ldots,L
$$

$$
F_i=X_0+\operatorname{Conv}_{3\times3}(X_L)
$$

STL 通过注意力聚合相关位置的信息，卷积进一步融合局部邻域信息，残差连接则保留输入特征，方便信息在多个 RSTB 之间传递。

计算注意力时，将特征展平为 $HW\times C$ 的 token 序列，每个空间位置对应一个 token；计算卷积时，再 reshape 回 $H\times W\times C$ 的特征图。

### STL

STL 依次使用注意力和 MLP，两者都先做 LayerNorm，再计算并加上残差：

$$
\hat X=X+\operatorname{MSA}(\operatorname{LN}(X)),\qquad
X_{out}=\hat X+\operatorname{MLP}(\operatorname{LN}(\hat X))
$$

MSA 在窗口内聚合不同位置的信息，MLP 则由两层线性层和中间的 GELU 组成，对每个位置的通道特征做变换。相邻 STL 交替使用普通窗口和移位窗口。

### Window Attention

将特征划分成互不重叠的 $M\times M$ 窗口。对一个窗口 $X_w\in\mathbb R^{M^2\times C}$，每个注意力头计算：

$$
Q=X_wW_Q,\qquad K=X_wW_K,\qquad V=X_wW_V
$$

$$
A=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}+B\right),\qquad Y=AV
$$

$d$ 为每个头的维度，$B$ 为可学习的相对位置偏置。多个头的结果拼接后再做线性投影。注意力矩阵大小为 $M^2\times M^2$，表示窗口内空间位置之间的关系。

仅考虑注意力矩阵的计算，复杂度由全局注意力的 $O((HW)^2C)$ 降为 $O(HWM^2C)$；固定窗口大小后，随像素数线性增长。

### Shifted Window

如果每层都使用相同的窗口划分，注意力就只能在各自窗口内计算。为了让相邻窗口交换信息，相邻层交替使用：

1. **W-MSA**：按普通窗口划分，计算窗口内注意力。
2. **SW-MSA**：将窗口划分偏移 $\lfloor M/2\rfloor$，使原本分属相邻窗口的 token 进入同一个窗口。

具体是先循环平移特征，再划分窗口并计算注意力，最后合并窗口并反向平移，恢复原来的位置。循环平移会把图像两端的区域拼到一起，因此需要 attention mask 屏蔽这些原本不相邻的位置。经过多层交替，信息就能逐步传到更远的窗口。

## Image Reconstruction

先通过长残差连接融合浅层和深层特征：

$$
F=F_0+F_{DF}
$$

**超分辨率**：重建头通过卷积和 [PixelShuffle](../concepts/%5Bup%2Cdown%5Dsampling.md#pixelshuffle) 上采样，再输出高分辨率图像。卷积先生成上采样需要的通道，PixelShuffle 再将这些通道重排到空间维度：

$$
\hat I_{HQ}=H_{REC}(F),\qquad
H\times W\times(r^2C')\xrightarrow{\mathrm{PixelShuffle}(r)}rH\times rW\times C'
$$

其中 $r$ 为放大倍数，$C'$ 为重排后的通道数。

例如放大 2 倍时，卷积先输出 $4C'$ 个通道，PixelShuffle 再把每组 4 个通道的值放到一个 $2\times2$ 区域中，使宽高各增加一倍、通道数变成 $C'$。PixelShuffle 本身没有可学习参数，也不做插值；需要填入这些位置的特征由前面的卷积学习。

[PixelUnshuffle](../concepts/%5Bup%2Cdown%5Dsampling.md#pixelunshuffle) 则反过来，将每个 $r\times r$ 区域中的值收进通道，降低空间分辨率。SwinIR 的这条重建流程不使用 PixelUnshuffle，主干中的 token 展平和窗口划分也不是 PixelUnshuffle。

**去噪和 JPEG 去伪影**：保持原分辨率，用卷积预测图像残差，再加回输入：

$$
\hat I_{HQ}=I_{LQ}+\operatorname{Conv}_{3\times3}(F)
$$

这里有两次残差相加：重建前将浅层特征 $F_0$ 加到深层特征上，重建后再将预测的图像残差加到输入图像上。
