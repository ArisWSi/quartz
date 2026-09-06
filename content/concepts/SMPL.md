---
title: SMPL
aliases:
  - Skinned Multi-Person Linear Model
  - SMPL model
tags:
  - concept
  - body-model
  - 3d-human
---

# SMPL

SMPL（Skinned Multi-Person Linear Model，Loper et al. 2015）是参数化人体网格模型：输入**形状参数 β**（身材，PCA 系数）与**姿态参数 θ**（各关节相对父关节的旋转），经线性 blend skinning 输出带蒙皮的三角网格（约 6890 顶点）与关节位置。整个人体因而可以被压缩成一组紧凑向量来表示。

- "提取 SMPL 参数"通常指用估计器（如 SMPLest-X、4D-Humans）从图像/视频回归 β、θ 与相机参数的过程。
- 相比 2D/3D 关键点，SMPL 自带拓扑与蒙皮，能重建稠密网格，便于重投影回 2D 做一致性校验，也方便驱动虚拟角色。
- 局限：不显式建模手部与面部细节，扩展见 [[SMPLX]]。

## 在 PlayerOne 中的角色

- 数据管线：SMPLest-X 从去背景的第三人称视频回归 SMPL 参数，作为人体运动表示；
- 参数分解：逐帧拆成身体与脚（66 维）、头部朝向（3 维）、双手（各 45 维），分别喂给三个 Motion Encoder；
- 质量校验：SMPL 参数经 [[SMPLX]] 重建网格后重投影，与 [[OpenPose]] 的 2D 关键点比对误差并过滤。

相关笔记：[[SMPLX]]、[[OpenPose]]、[[Playerone]]
