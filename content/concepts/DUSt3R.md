---
title: DUSt3R
tags:
  - concept
  - 3d-reconstruction
---

# DUSt3R

DUSt3R（Wang et al. 2024, CVPR；arXiv:2312.14132）直接从两张**未标定**图像回归稠密 3D 几何：输出两张图各自的稠密点图（pointmap）与置信度，并统一到同一坐标系——端到端，无需相机标定与特征匹配，两两配对即可扩展为多视角重建。

在 PlayerOne 中：出现在 SR 的消融实验里——把点图渲染器 [[CUT3R]] 换成 DUSt3R 结果依然成立，说明该模块对点图来源不敏感。

相关笔记：[[CUT3R]]、[[Geo4D]]
