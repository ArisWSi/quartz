---
title: Diffusion Models
tags:
  - synthesis
  - diffusion
  - generative-model
---

## from DDPM to ControlNet

[[DDPM]] 是一篇范式级的论文，提出了diffusion model的数学建模，并使用UNet架构实现了图像生成任务。
[[LDM]] 基于[[DDPM#Experiments & Discussions]]中的率失真分析，提出了一套二阶段的隐空间生成路线：在感知压缩后的隐空间上，使用diffusion models做进一步的语义压缩。在效率和质量上都取得了很好的结果。在此之外，它还提出了使用cross-attention机制的条件生成方法，可以接受的条件类型广泛
[[DiT]] 是一篇架构改进论文。其将旧的UNet架构替换成Tranformer Block，做了详尽的实验分析以说明计算效率、生成质量，以及scalability。并且尝试了多种条件注入的机制。
[[ControlNet]] 的灵感来源于NLP领域的超网络(HyperNetwork)和模型微调(Finetuning)。在一个原网络的copy上训练，并经过zero-conv作为残差加到解码器的feature上。能够很好的保持spatial信息。

[[WSDT]]

## flow based models and their applications

流模型是近来广受采纳的方案。是将图像生成解释为从分布到分布的搬运，基于最优运输理论使用ODE来描述这个过程。

[[RectifiedFlow]] 修正流模型属于 flow matching 流派，是比较前沿的少步采样范式。它把生成建模为噪声与数据两个分布之间由 ODE 描述的搬移过程，网络学习一个速度场，使沿该速度场积分即可把噪声样本送入数据分布。它的独特贡献是 reflow 流程：先把上一轮训练好的速度场模型固定下来，用它前向求解 ODE，把重新采样的噪声映射到新的数据端点，组成新的噪声-数据配对，再在这些配对对应的直线插值路径上重新训练，使轨迹不断被“拉直”；反复迭代后逐步逼近最优运输给出的直线位移映射。轨迹越直，离散化误差越小，生成所需的采样步数越少，配合蒸馏甚至能做到一步生成。

[[FlowEdit]] 
[[StableFlow]]
