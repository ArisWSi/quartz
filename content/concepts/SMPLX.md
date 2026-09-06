---
title: SMPLX
aliases:
  - SMPL-X
tags:
  - concept
  - body-model
  - 3d-human
---

# SMPLX

SMPL-X（Pavlakos et al. 2019，Expressive Body Capture）是 [[SMPL]] 的"全身"扩展：在身体之外显式建模**双手**与**面部**，用单个模型联合表达 body + hands + face。网格约 10475 顶点，手部每手 15 个关节，脸部含下颌与眼球，可配合表情参数驱动。

- 把身体/手/脸统一进同一参数空间，是现代全身捕捉（full-body capture）管线的常见输出形式（例如 SMPLest-X 等估计器回归的就是这类参数）。
- 论文中常见用法之一：用 SMPL-X 把估计出的参数重建为稠密 3D 网格，重投影回 2D 图像平面，再与 2D 关键点检测结果（如 [[OpenPose]]）比对，作为估计质量的校验手段。

相关笔记：[[SMPL]]、[[OpenPose]]、[[Playerone]]
