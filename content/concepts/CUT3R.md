---
title: CUT3R
aliases:
  - Continuous 3D Perception Model with Persistent State
tags:
  - concept
  - 3d-reconstruction
---

# CUT3R

CUT3R（Wang et al. 2025；arXiv:2501.12387）是 [[DUSt3R]] 面向**视频**的扩展：把输入视为按时间顺序的图像流，用 persistent state 累积全局 3D 记忆，逐帧输出与全局坐标系对齐的稠密点图，支持长视频的连续流式 3D 重建。

在 PlayerOne 中：SR 模块用它从训练视频渲染点图序列——第 n 帧的点图由第 1～n 帧重建，作为"4D 场景"的监督信号。

相关笔记：[[DUSt3R]]、[[Geo4D]]
