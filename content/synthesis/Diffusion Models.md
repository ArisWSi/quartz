---
title: Diffusion Models
tags:
  - synthesis
  - diffusion
  - generative-model
description: 扩散模型综述：从 DDPM、LDM、DiT、ControlNet 到 Rectified Flow 等流模型及其应用。
---

## from DDPM to ControlNet

[[DDPM]] 在已有 diffusion probabilistic models 的基础上，通过去噪参数化、训练目标与 UNet 实现，展示了高质量图像生成能力。
[[LDM]] 基于[[DDPM#Experiments & Discussions]]中的率失真分析，提出了一套二阶段的隐空间生成路线：在感知压缩后的隐空间上，使用diffusion models做进一步的语义压缩。在效率和质量上都取得了很好的结果。在此之外，它还提出了使用cross-attention机制的条件生成方法，可以接受的条件类型广泛
[[DiT]] 是一篇架构改进论文。其将旧的UNet架构替换成Tranformer Block，做了详尽的实验分析以说明计算效率、生成质量，以及scalability。并且尝试了多种条件注入的机制。
[[ControlNet]] 的灵感来源于NLP领域的超网络(HyperNetwork)和模型微调(Finetuning)。在一个原网络的copy上训练，并经过zero-conv作为残差加到解码器的feature上。能够很好的保持spatial信息。

[[WSDT]]

## 从方法改进到问题重构

值得关注的不只是换了什么架构，还包括作者如何识别既有方法中不必保留的限制。DDIM 从加速 DDPM 采样出发，重新审视了训练目标与采样过程必须绑定到什么程度。

去噪训练使用各时刻的 $q(x_t\mid x_0)$，并不直接观察完整加噪轨迹。DDIM 构造保持这些条件边缘分布的非马尔可夫前向过程，使一族生成过程可以共用去噪训练目标，其中包括确定性采样。它保留了训练所需的结构，同时改变了时刻之间的连接方式。[DDIM §3–4](https://arxiv.org/html/2010.02502v4)

DDIM 的 §4.3 已讨论确定性更新的连续极限与 ODE 的联系。Score-SDE 则系统建立了正向 SDE、反向 SDE 和 probability flow ODE；在精确 score 等条件下，后两者具有相同时间边缘分布，但并不具有相同样本轨迹，有限步数值求解也会引入误差。[DDIM §4.3](https://arxiv.org/html/2010.02502v4#S4.SS3) · [Score-SDE](https://arxiv.org/abs/2011.13456)

我的收获是：改进方法也可能包含深刻的问题重构。提出新视角不一定是增加假设，也可以是发现哪些原有条件能够放松。读论文时应追问：目标真正依赖什么，哪些只是当前算法的选择？

这与简化的能力相连：区分人为设计、精确推导、分布近似与经验目标，再分别检查它们的依据和失效条件。具体见 [[DDPM#建模与简化的层次]]；低层视觉中的应用与实验复盘见 [[Low level#实验与复盘：定位推理链中失效的环节]]。

## flow based models and their applications

流模型是近来广受采纳的方案。是将图像生成解释为从分布到分布的搬运，基于最优运输理论使用ODE来描述这个过程。

[[RectifiedFlow]] 修正流模型属于 flow matching 流派，是比较前沿的少步采样范式。它把生成建模为噪声与数据两个分布之间由 ODE 描述的搬移过程，网络学习一个速度场，使沿该速度场积分即可把噪声样本送入数据分布。它的独特贡献是 reflow 流程：先把上一轮训练好的速度场模型固定下来，用它前向求解 ODE，把重新采样的噪声映射到新的数据端点，组成新的噪声-数据配对，再在这些配对对应的直线插值路径上重新训练，使轨迹不断被“拉直”；反复迭代后逐步逼近最优运输给出的直线位移映射。轨迹越直，离散化误差越小，生成所需的采样步数越少，配合蒸馏甚至能做到一步生成。

[[FlowEdit]] 
[[StableFlow]]
