---
title: Geo4D
tags:
  - concept
  - 4d-reconstruction
  - 3d-reconstruction
---

# Geo4D

Geo4D（Jiang et al. 2025；arXiv:2504.07961）把视频生成模型的几何先验用于单目视频的 4D（动态 3D）重建，估计逐帧稠密 3D 运动与结构。其配套的点图压缩编码器可把稠密点图压成 latent。

在 PlayerOne 中：直接借用它的 Point Map Encoder，把 [[CUT3R]] 渲染的点图压成 latent，再接 5 层 3D 卷积 Adapter 与视频 latent 对齐。

相关笔记：[[DUSt3R]]、[[CUT3R]]
