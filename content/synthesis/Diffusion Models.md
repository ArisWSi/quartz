---
title: Diffusion Models
tags:
  - synthesis
  - diffusion
  - generative-model
description: 扩散模型综述：从 DDPM、LDM、DiT、ControlNet 到 Rectified Flow 等流模型及其应用。
---

## 阅读顺序

先读 [[LDM]]，分清 autoencoder、文本编码器和去噪 UNet；[[DiT]] 改去噪网络，[[ControlNet]] 在原网络旁增加可训练的控制分支。之后再读 [[RectifiedFlow]]，比较 [[FlowEdit]] 与 [[StableFlow]] 两种编辑思路，最后看 [[WSDT]] 如何在去噪时改动风格相关特征。

## from DDPM to ControlNet

[[DDPM]] 在已有 diffusion probabilistic models 的基础上，通过去噪参数化、训练目标与 UNet 实现，展示了高质量图像生成能力。
[[LDM]] 用感知压缩的二维潜空间减少去噪成本，以 cross-attention 接入文本或布局等条件。autoencoder 管表示与重建，条件编码器管提示词，UNet 管潜变量去噪。
[[DiT]] 保留潜空间扩散框架，将去噪 UNet 换成 Transformer，并比较不同 patch 大小、模型规模与条件注入方式的效果。
[[ControlNet]] 冻结原 UNet、训练其编码侧与中间块的副本，经 zero convolution 把各尺度控制特征注入原网络的 skip connection 与中间块。
[[WSDT]] 在推理时对 self-attention 中与风格相关的特征做逐通道分布变换，用于参考图风格生成。

## 从方法改进到问题重构

值得关注的不只是换了什么架构，还包括作者如何识别既有方法中不必保留的限制。DDIM 从加速 DDPM 采样出发，重新审视了训练目标与采样过程必须绑定到什么程度。

去噪训练使用各时刻的 $q(x_t\mid x_0)$，并不直接观察完整加噪轨迹。DDIM 构造保持这些条件边缘分布的非马尔可夫前向过程，使一族生成过程可以共用去噪训练目标，其中包括确定性采样。它保留了训练所需的结构，同时改变了时刻之间的连接方式。[DDIM §3–4](https://arxiv.org/html/2010.02502v4)

DDIM 的 §4.3 已讨论确定性更新的连续极限与 ODE 的联系。Score-SDE 则系统建立了正向 SDE、反向 SDE 和 probability flow ODE；在精确 score 等条件下，后两者具有相同时间边缘分布，但并不具有相同样本轨迹，有限步数值求解也会引入误差。[DDIM §4.3](https://arxiv.org/html/2010.02502v4#S4.SS3) · [Score-SDE](https://arxiv.org/abs/2011.13456)

我的收获是：改进方法也可能包含深刻的问题重构。提出新视角不一定是增加假设，也可以是发现哪些原有条件能够放松。读论文时应追问：目标真正依赖什么，哪些只是当前算法的选择？

这与简化的能力相连：区分人为设计、精确推导、分布近似与经验目标，再分别检查它们的依据和失效条件。具体见 [[DDPM#建模与简化的层次]]；低层视觉中的应用与实验复盘见 [[Low level#实验与复盘：定位推理链中失效的环节]]。

## flow based models and their applications

流模型通过学习速度场与求解 ODE 描述从源分布到目标分布的连续搬运；有些构造借助最优运输思想，但不能把一般流模型等同于已求得最优运输。

[[RectifiedFlow]] 修正流模型属于 flow matching 流派，是比较前沿的少步采样范式。它把生成建模为噪声与数据两个分布之间由 ODE 描述的搬移过程，网络学习一个速度场，使沿该速度场积分即可把噪声样本送入数据分布。它的独特贡献是 reflow 流程：先把上一轮训练好的速度场模型固定下来，用它前向求解 ODE，把重新采样的噪声映射到新的数据端点，组成新的噪声-数据配对，再在这些配对对应的直线插值路径上重新训练，使轨迹不断被“拉直”；反复迭代后逐步逼近最优运输给出的直线位移映射。轨迹越直，离散化误差越小，生成所需的采样步数越少，配合蒸馏甚至能做到一步生成。

[[FlowEdit]] 用源/目标条件速度场之差直接编辑真实图像，避免对源图做 ODE 反演；其配对是启发式，并非严格最优运输。
[[StableFlow]] 先用逐层旁路实验识别对生成结果影响大的 DiT 层，再在这些层注入参考图视觉特征；编辑真实图像时仍需反演并改进其重建。
