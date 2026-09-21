---
title: Genie
aliases:
  - Generative Interactive Environment
tags:
  - paper
  - world-model
  - video-generation
  - generative-model
  - 2024
description: Genie：从无标注网络视频自监督训练的生成式交互环境，由时空视频 tokenizer、隐动作模型与自回归 dynamics model 组成。
---

## Overview
Generative Interactive Environment, trained in unsupervised manner from unlabelled Internet videos.

从无标注的网络视频中，以自监督的方法训练出的 **Gen**erative **I**nteractive **E**nvironment

## 2-phases generation pipeline
- Train the video tokenizer
- Cotrain latent action model(directly from pixels) and the dynamics model (on video tokens)

## Architecture
![architecture overview](assets/genie-overview.png)

由一个时空视频tokenizer (spatialtemporal video tokenizer)、一个自回归动力系统模型、和一个简单可扩展的隐动作模型组成。

### ST-tranformer

![ST transformer](assets/genie-sttransformer.png)

视频造成了四次方的显存压力，换句话说会产生$10^4$的token。

此处采用了一个 memory-efficient ST-transformer architecture.

一个 ST-tranformer block = Spatial attention(reshape to $BT\times HW$) + Temporal attention(reshape to $BHW \times T$) + FFW

理论上是每个attention层之后要加入一个前馈层，但是这里为了scaling性质省略掉了。后来发现这重大的提升了结果水平。

### Video tokenizer



### Latent action model

训练目标：
- 训练$x_{1:t+1} \mapto a_{1:t}$的编码映射
- 训练从前t帧和t帧的latent action repr 到下一帧的解码映射

这里Genie只用到了编码出来的latent action，只是使用AE架构来强化表示，类似[[AML]]中的自编码器架构。具体的采用了一个VQ-VAE-based objective，并限制了VQ码本的大小来取得human playbility和enforce controlbility。

当然解码器只负责提供训练信号，正如[[AML]]中写的那样。

使用因果mask并行的对帧序列和状态动作序列做处理。

### Dynamics model
视频tokens和隐动作输入，使用[[MaskGIT]]的方法自回归预测下一帧。


## Train
