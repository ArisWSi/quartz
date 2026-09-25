---
title: DiT
aliases:
  - Scalable Diffusion Models with Transformers
tags:
  - paper
  - diffusion
  - transformer
  - 2023
description: 在潜空间扩散模型中，用 Transformer 替代去噪 UNet。
---

## Overview

[Scalable Diffusion Models with Transformers，ICCV 2023](https://arxiv.org/html/2212.09748v2)

DiT 沿用 LDM 的 autoencoder 和潜空间扩散过程，改的是**预测噪声的网络**：把 UNet 换成处理潜变量 patch 的 Transformer。论文主要做 ImageNet 类别条件生成，不是文本到图像实验。

## Architecture

将加噪潜变量 $z_t\in\mathbb R^{I\times I\times C}$ 切成 $p\times p$ 的 patch，得到 $(I/p)^2$ 个 token；加上位置编码后交给多层 DiT block，最后把 token 还原为空间张量，预测噪声和方差。

时间步和类别标签也要进入网络。作者比较了把它们当额外 token、使用 cross-attention、使用 adaptive layer norm 等方式。主模型采用 **adaLN-Zero**：条件控制归一化的缩放与偏移，也控制残差分支的强度；残差门初始化为零，Transformer block 起初近似恒等映射。

## Experiments & Discussions

作者通过更深/更宽的网络和更小的 patch 增加单次前向计算量，观察到 FID 随模型规模改善。patch 从 4 缩到 2，token 数会变成四倍，但参数量基本不因 patch 数增长。论文把 FLOPs 作为扩展趋势的横轴。

值得警惕的是，**FLOPs 高与 FID 低的相关性，不等于同等训练成本下 DiT 必然优于 UNet**。训练步数、吞吐量、VAE、guidance 和实现效率也会改变结果。adaLN-Zero 在这里胜过 cross-attention，也只是在类别条件设定下的实验结论；长文本条件需要逐 token 交互，不能照搬这个排序。

相关：[[LDM]] · [[StableFlow]]
