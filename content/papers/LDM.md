---
title: LDM
aliases:
  - High-Resolution Image Synthesis with Latent Diffusion Models
tags:
  - paper
  - diffusion
  - generative-model
  - 2022
description: 在 autoencoder 的潜空间中训练扩散模型，并通过 cross-attention 接入文本等条件。
---

## Overview

[High-Resolution Image Synthesis with Latent Diffusion Models，CVPR 2022](https://arxiv.org/html/2112.10752v2)

Pixel-space diffusion 每一步都处理整张图像，分辨率一高，训练和采样都很贵。LDM 先用 autoencoder 把图像压缩到较小的二维潜空间，再在这个空间里训练 diffusion model。压缩倍率不能一味增大：太大时，解码器已经无法把细节还原出来；太小时，每步去噪仍然昂贵。

## Autoencoder

编码器 $E$ 将图像 $x$ 变成 $z=E(x)$，解码器 $D$ 将 $z$ 还原为图像。训练第一阶段时，作者用感知损失、对抗损失保证重建质量，并比较轻量 KL 正则与 VQ 正则。进入第二阶段后，autoencoder 固定，扩散模型只需学习潜变量的分布。

这个分工很重要：autoencoder 负责**表示与重建**，不负责从随机噪声中决定图像的内容。它如果已经丢掉某些细节，后面的 diffusion 很难再凭空恢复。

## Conditional Diffusion

去噪网络是时间条件 UNet。训练时在 $z_0=E(x)$ 上加噪得到 $z_t$，让网络预测加入的噪声：

$$
\mathcal L=\mathbb E_{x,t,\epsilon}\left[
\left\|\epsilon-\epsilon_\theta(z_t,t,\tau(y))\right\|^2
\right].
$$

$\tau(y)$ 是条件编码器产生的表示。文本条件经 cross-attention 注入 UNet：图像特征提供 query，文本 token 提供 key 和 value。采样时从随机潜变量开始，反复调用 UNet 去噪，最后只调用一次 $D$ 得到像素图像。图像 autoencoder 和文本编码器处理的是两种不同输入，不应合称一个“编码器”。

## 与 Stable Diffusion v1 的关系

**LDM 是方法，Stable Diffusion v1 是后来的一套具体模型配置。** 原论文的文本生成实验使用 BERT tokenizer 和训练的 Transformer 条件编码器；[SD v1 官方实现](https://github.com/CompVis/stable-diffusion/blob/main/README.md)使用冻结的 CLIP ViT-L/14 文本编码器。CLIP 在这里把 prompt 变成供 UNet 查询的文本表示；它不负责解码图像，也不是每一步执行去噪的网络。

我觉得这篇最值得记住的不是“把扩散放进 latent”这一个动作，而是把画质瓶颈分成两处看：一处是 autoencoder 的重建能力，另一处是潜空间生成能力。比较模型时，先检查前者，否则容易把压缩损失误认为去噪模型的问题。

相关：[[DDPM]] · [[DiT]] · [[ControlNet]]
