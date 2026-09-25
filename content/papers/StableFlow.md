---
title: Stable Flow
aliases:
  - Stable Flow: Vital Layers for Training-Free Image Editing
  - StableFlow
tags:
  - paper
  - flow
  - image-editing
  - 2025
description: 找出 DiT 中对生成结果影响大的层，只在这些层注入参考图视觉特征。
---

## Overview

[Stable Flow: Vital Layers for Training-Free Image Editing，CVPR 2025](https://arxiv.org/html/2411.14430v2)

以往在 UNet 中做图像编辑，可以利用它的多尺度结构决定在哪些层保留原图特征。DiT 没有同样清楚的编码器—解码器层次。Stable Flow 先找出对生成结果影响大的层，再只在这些层做参考图特征注入。

## Vital Layers

固定 prompt 和随机种子，生成一张图；然后逐层旁路 DiT 的某一层，重新生成，并用 DINOv2 特征衡量两张图的差异。旁路后变化越大，该层越 “vital”。这些层并不连续地集中在网络前部或后部。

这个实验很有启发，但“旁路一层影响很大”只说明它对**当前模型与生成结果**重要，不能直接解释成它专门存着物体身份、构图或纹理。层的重要性也可能随采样设置、prompt 和相似度指标变化。

## Image Editing

参考图轨迹和编辑轨迹并行生成，编辑轨迹使用新 prompt。在 vital layers 中，用参考轨迹的图像特征干预编辑轨迹的注意力计算；其他层让编辑轨迹自行演化。全部层都注入，容易压住新 prompt；只在选出的层注入，论文观察到较好的保留与修改平衡。

对模型自己生成、且知道 seed 的图像，这样即可编辑。对真实照片，作者仍须先做**反演**，再用 latent nudging 减少反演后的重建误差。因此它与 [[FlowEdit]] 的“无源图反演”是两条不同路线。“training-free”只是说不为编辑更新模型权重。

## Cons

论文展示了物体替换、增加物体等编辑，也给出风格大改、精确移动物体、完整更换背景的失败例子。注入参考特征本来就是为了保留原图；当要求大范围改变时，同一机制可能成为阻碍。我会更想看不同编辑幅度下，固定 vital set 是否仍比按任务选择层更合适。

相关：[[DiT]] · [[FlowEdit]] · [[WSDT]]
