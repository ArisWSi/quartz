---
title: OpenPose
tags:
  - concept
  - pose-estimation
---

# OpenPose

OpenPose（Cao et al. 2017 / TPAMI 2019）是经典的自底向上（bottom-up）2D 多人姿态估计方法：卷积网络先为所有人预测关键点 heatmap 与 **PAFs**（Part Affinity Fields，关键点之间的连接向量场），再通过贪心匹配把关键点连成完整骨架。无需先做人物检测，天然支持多人场景。

- 与自顶向下（先检测人框、再对单人体态估计）路线相对，自底向上把"找关键点"与"连骨架"两件事分开处理。
- 常被当作独立、与模型无关的 2D 关键点检测器，用来校验其他 3D 姿态/网格估计结果。

## 在 PlayerOne 中的角色

作为 2D 关键点的"参考真值"：把 3D SMPL 网格（经 [[SMPLX]] 重建）重投影到 2D 平面后，与 OpenPose 检测的关键点计算重投影误差，剔除误差最大的前 10% 样本，保证运动-视频对数据质量（见 PlayerOne 的 Data Preprocessing）。

相关笔记：[[SMPL]]、[[SMPLX]]、[[Playerone]]
