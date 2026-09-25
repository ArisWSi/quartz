---
title: FlowEdit
aliases:
  - Inversion-Free Text-Based Editing Using Pre-Trained Flow Models
tags:
  - paper
  - flow
  - image-editing
  - 2025
description: 用源与目标提示词的速度场差直接编辑图像，不做源图反演。
---

## Overview

[FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models，ICCV 2025](https://arxiv.org/html/2412.08629v2)

常见的流模型编辑先把原图反演到噪声，再用目标 prompt 从这份噪声生成新图。即使反演完全准确，共用同一份噪声也不保证源图与目标图形成“只改必要部分”的配对。FlowEdit 从这个配对问题出发，直接构造源图到目标图的编辑轨迹。

## Method

记原图为 $x^{src}$，源和目标提示词分别为 $c_{src},c_{tar}$。每个时间步重新抽噪声 $n_t$，取

$$
\hat z_t^{src}=(1-t)x^{src}+tn_t,\qquad
\hat z_t^{tar}=z_t^{FE}+\hat z_t^{src}-x^{src}.
$$

在这两个点上分别查询预训练流模型的速度场，用目标速度减去源速度，沿所得方向更新 $z_t^{FE}$。直观地说，它在问：**同一局部扰动下，朝目标描述走的方向，比朝原描述走的方向多出什么？**

理论表达式对不同噪声的速度差取期望；实现时用少量采样近似，论文的主要实验每步只用一个噪声样本。它不求原图对应的初始噪声，也不改模型权重或内部注意力，但仍要多次运行流模型。所谓 inversion-free 不等于一步生成。

## Experiments & Discussions

作者先用简单分布展示：经噪声中转可能把源样本配到很远的目标样本；FlowEdit 的配对在例子里运输成本更低。再在 SD3、FLUX 上比较真实图像编辑，重点看目标文本遵循与原图结构保留的平衡。

这里不能把“成本更低”读成“求出了最优运输”。论文自己说明，平均速度场的做法是启发式，未保证精确匹配目标分布。强保留原图也会妨碍大范围的姿态或背景修改；这不是某个指标可以单独说明的好坏，而是编辑强度的取舍。

相关：[[RectifiedFlow]] · [[StableFlow]]
