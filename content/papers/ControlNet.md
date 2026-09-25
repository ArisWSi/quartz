---
title: ControlNet
aliases:
  - Adding Conditional Control to Text-to-Image Diffusion Models
tags:
  - paper
  - diffusion
  - controllable-generation
  - 2023
description: 冻结原文生图模型，在其旁边训练空间条件分支。
---

## Overview

[Adding Conditional Control to Text-to-Image Diffusion Models，ICCV 2023](https://arxiv.org/html/2302.05543v3)

文本能描述“画什么”，但很难精确指定姿态、轮廓或深度。ControlNet 给预训练 Stable Diffusion 增加一条可训练的空间条件分支，让边缘图、人体骨架、深度图等参与去噪，同时保留原模型的生成能力。

## Architecture

原 UNet 保持冻结。旁边复制它的 **12 个编码块和 1 个中间块**，不复制解码器；副本从原模型权重出发训练。条件图先经过一个小型卷积编码器，变成与潜变量分辨率相符的特征，再送入可训练副本。这一条件编码器与 [[LDM]] 中压缩自然图像的 autoencoder 不是同一个东西。

副本的各尺度输出经过 zero convolution，分别加到原 UNet 的 12 条 skip connection 和中间块。输入条件进入副本时也经过 zero convolution。所谓 zero convolution，就是权重和偏置都初始化为零的 $1\times1$ 卷积。开始训练时，副本对原模型的输出贡献为零；输出端 zero convolution 先学会传递信号，此后条件和副本内部参数才逐渐发挥作用。零初始化并不会让训练永远停住。

## Training & Discussions

训练目标仍是预测潜变量中加入的噪声，只更新条件编码器与副本。论文为边缘、深度、姿态等条件分别训练 ControlNet，也展示多个条件组合。它不是拿一张从未训练过的新类型控制图，就能自动知道怎样使用。

我觉得这里的关键取舍是：冻结主干可减少小数据微调时损坏原模型能力的风险，但复制一大段 UNet 也增加显存与推理成本。比较它和直接微调时，不能只展示“更听话”的图，还要同时看文本遵循、原模型能力是否退化，以及新增的计算。

相关：[[LDM]] · [[DiT]]
