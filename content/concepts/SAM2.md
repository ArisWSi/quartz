---
title: SAM 2
aliases:
  - SAM2
  - Segment Anything Model 2
tags:
  - concept
  - segmentation
---

# SAM 2

SAM 2（Ravi et al. 2025, ICLR）是 SAM 的图像+视频统一分割模型：用点/框/掩码提示分割目标后，借助 streaming memory 把掩码跨帧传播，可半自动地对视频目标做跟踪式分割，也支持交互修正。

在 PlayerOne 中：从第三人称视角视频中框选/分割出最大的人物并去除背景，交给下游的 SMPLest-X 做姿态估计。

相关笔记：[[SMPL]]、[[Playerone]]
